"""
Training script for path prediction diffusion model.
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_preprocessing import prepare_datasets
from diffusion_model import (
    PathDiffusionModel, generate_constraint_mask, MIN_CONSTRAINT_SPACING, MODEL_FORMAT_VERSION
)


class AdamAtan2(torch.optim.Optimizer):
    """Adam-atan2 optimizer (Everett et al., 2024), as used by Kimodo.

    Adam variant that replaces the m/sqrt(v) ratio with atan2(m, sqrt(v)),
    bounding the update magnitude to +-pi/2 for improved training stability.
    """

    def __init__(self, params, lr: float = 2e-5, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = exp_avg / (1 - beta1 ** t)
                v_hat = exp_avg_sq / (1 - beta2 ** t)

                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(torch.atan2(m_hat, v_hat.sqrt() + group['eps']), alpha=-group['lr'])

        return loss


class DiffusionTrainer:
    """Trainer for the diffusion model."""
    
    def __init__(
        self,
        model: PathDiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-6,
        ema_decay: float = 0.995,
        ema_interval: int = 10,
        lr_schedule: str = 'constant',
        lr_start: float = 1e-4,
        lr_end: float = 2e-5,
        save_dir: str = './checkpoints_kimodo'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer: each run gets its own subdirectory (run_001, run_002, ...)
        log_root = self.save_dir / 'logs'
        log_root.mkdir(parents=True, exist_ok=True)
        run_num = 1
        for d in log_root.iterdir():
            if d.is_dir() and d.name.startswith('run_'):
                try:
                    run_num = max(run_num, int(d.name[4:]) + 1)
                except ValueError:
                    pass
        self.writer = SummaryWriter(log_dir=log_root / f'run_{run_num:03d}')
        print(f"TensorBoard run: run_{run_num:03d}")

        # Optimizer: Adam-atan2, constant LR by default (Kimodo recipe) or
        # annealed after each epoch by a cosine schedule from lr_start to lr_end
        if lr_schedule not in ('constant', 'cosine'):
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")
        self.lr_schedule = lr_schedule
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.optimizer = AdamAtan2(
            model.parameters(),
            lr=lr_start if lr_schedule == 'cosine' else learning_rate,
            weight_decay=weight_decay
        )

        # Exponential moving average of model weights: used for validation and
        # checkpoints (the EMA weights are what gets deployed at test time)
        self.ema_decay = ema_decay
        self.ema_interval = ema_interval
        self.ema_params = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.next_epoch = 0
        self.best_val_loss = float('inf')
        self.best_ema_snapshot = None

    def _canonicalize(self, target: torch.Tensor) -> torch.Tensor:
        """Translate each sequence so its first point is at the origin.

        During training, additionally applies a random in-plane rotation per
        sequence to both position (x, y) and direction (dx, dy) channels.
        """
        origin = target[:, :1, :2]  # (B, 1, 2): position of the first point
        target = torch.cat([target[..., :2] - origin, target[..., 2:]], dim=-1)

        if self.model.training:
            batch_size = target.shape[0]
            angles = torch.rand(batch_size, device=target.device) * (2 * math.pi)
            cos, sin = torch.cos(angles), torch.sin(angles)
            # Rotation matrices (B, 2, 2): [[cos, -sin], [sin, cos]]
            rot = torch.stack([
                torch.stack([cos, -sin], dim=-1),
                torch.stack([sin, cos], dim=-1)
            ], dim=-2)
            # Group the 4 features per timestep into two 2-vectors (position, direction)
            target = torch.einsum(
                'bij,btfj->btfi', rot, target.view(batch_size, -1, 2, 2)
            ).reshape(target.shape)

        return target

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute clean-trajectory L1 loss over valid (non-padded) positions.

        The first and last valid point of each sequence are always constrained:
        clean features are imputed and their mask is concatenated in the denoiser.
        Random interior frames are also constrained in all four features, with
        at least 60 frames between every pair of adjacent constraints.
        """
        target = batch['target'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)  # (B, T_pred): 1=valid, 0=padding

        # Canonicalize: first point at origin, random rotation during training
        target = self._canonicalize(target)

        batch_size = target.shape[0]
        num_timesteps = self.model.diffusion.num_timesteps

        # Sample random timesteps
        timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.device)

        # Constraint mask: first and last valid position of each sequence
        constraint_mask = generate_constraint_mask(attention_mask)

        # Sample noise and impute clean constraints in the model's training forward
        valid = attention_mask.bool()

        # Forward pass
        clean_pred = self.model(
            target, timesteps, constraint_mask=constraint_mask, attention_mask=attention_mask
        )

        # Masked L1 reconstruction loss: padded positions contribute nothing
        loss = nn.functional.l1_loss(clean_pred[valid], target[valid])

        return loss
    
    @torch.no_grad()
    def _update_ema(self):
        """Update the exponential moving average of model parameters."""
        for name, p in self.model.named_parameters():
            self.ema_params[name].mul_(self.ema_decay).add_(
                p.detach(), alpha=1 - self.ema_decay
            )

    def _epoch_lr(self, epoch: int, num_epochs: int) -> float:
        """Cosine learning rate for a 0-indexed epoch: lr_start at epoch 0, lr_end at the last."""
        progress = epoch / max(1, num_epochs - 1)
        return self.lr_end + (self.lr_start - self.lr_end) * (1 + math.cos(math.pi * progress)) / 2

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            loss = self.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()

            # EMA update every ema_interval steps (Kimodo-style)
            if self.global_step % self.ema_interval == 0:
                self._update_ema()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})
        
        self.next_epoch = self.current_epoch + 1
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model using the EMA weights (as used at test time)."""
        self.model.eval()

        # Swap in EMA weights for evaluation
        raw_params = {
            name: p.detach().clone() for name, p in self.model.named_parameters()
        }
        for name, p in self.model.named_parameters():
            p.copy_(self.ema_params[name])

        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(self.val_loader, desc="Validation"):
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1

        # Restore raw weights
        for name, p in self.model.named_parameters():
            p.copy_(raw_params[name])

        return total_loss / num_batches
    
    def save_checkpoint(self, filename: str = 'best_model.pt', update_latest: bool = True):
        """Atomically save a resumable checkpoint and update latest.pt."""
        numpy_rng = np.random.get_state()
        checkpoint = {
            'format_version': MODEL_FORMAT_VERSION,
            'prediction_type': 'x0',
            'constraint_conditioning': 'clean_imputation_mask',
            'model_config': dict(self.model.config),
            'epoch': self.next_epoch - 1,
            'next_epoch': self.next_epoch,
            'global_step': self.global_step,
            'ema_decay': self.ema_decay,
            'ema_interval': self.ema_interval,
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
            'python_rng_state': random.getstate(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            # EMA weights are saved as model_state_dict: they are what gets
            # deployed at test time
            'model_state_dict': dict(self.ema_params),
            'raw_model_state_dict': self.model.state_dict(),
            'ema_state_dict': dict(self.ema_params),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        paths = [self.save_dir / filename]
        if update_latest:
            paths.append(self.save_dir / 'latest.pt')
        for path in dict.fromkeys(paths):
            temporary_path = path.with_suffix(path.suffix + '.tmp')
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, path)
            print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Restore training state and continue after the last completed epoch."""
        if checkpoint_path == 'latest':
            checkpoint_path = self.save_dir / 'latest.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if PathDiffusionModel.checkpoint_config(checkpoint) != self.model.config:
            raise ValueError('Resume model configuration must match the saved model_config')
        
        required = {'raw_model_state_dict', 'ema_state_dict', 'optimizer_state_dict',
                    'next_epoch', 'global_step', 'best_val_loss', 'ema_decay', 'ema_interval',
                    'torch_rng_state', 'numpy_rng_state', 'python_rng_state', 'cuda_rng_state'}
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f'Checkpoint is missing required fields: {sorted(missing)}')

        # Prefer raw training weights over the EMA inference weights when resuming
        self.model.load_state_dict(checkpoint['raw_model_state_dict'])
        self.ema_params = {
            k: v.to(self.device) for k, v in checkpoint['ema_state_dict'].items()
        }
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.next_epoch = checkpoint['next_epoch']
        self.current_epoch = self.next_epoch
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_ema_snapshot = None
        self.ema_decay = checkpoint['ema_decay']
        self.ema_interval = checkpoint['ema_interval']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        name, keys, position, has_gauss, cached = checkpoint['numpy_rng_state']
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
        random.setstate(checkpoint['python_rng_state'])
        cuda_rng = checkpoint['cuda_rng_state']
        if cuda_rng and torch.cuda.is_available():
            if len(cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
            else:
                print('CUDA device count changed; CUDA RNG state was not restored.')
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming at epoch {self.next_epoch + 1} ({self.next_epoch} completed), step {self.global_step}")
    
    def train(self, num_epochs: int, val_every: int = 1, save_every: int = 50):
        """Train to a total epoch count, saving best, periodic, and final checkpoints."""
        if num_epochs < 0 or val_every < 1 or save_every < 1:
            raise ValueError('num_epochs must be nonnegative; val_every and save_every must be positive')
        print(f"Training to {num_epochs} total epochs ({self.next_epoch} already completed)")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.lr_schedule == 'cosine':
            print(f"Optimizer: Adam-atan2, cosine lr {self.lr_start:.2e} -> {self.lr_end:.2e} over {num_epochs} epochs")
        else:
            print(f"Optimizer: Adam-atan2, constant lr = {self.optimizer.param_groups[0]['lr']}")
        print(f"EMA: decay = {self.ema_decay}, every {self.ema_interval} steps")
        
        for epoch in range(self.next_epoch, num_epochs):
            self.current_epoch = epoch

            if self.lr_schedule == 'cosine':
                self._set_lr(self._epoch_lr(epoch, num_epochs))

            # Train
            train_loss = self.train_epoch()
            self.next_epoch = epoch + 1
            print(f"Epoch {epoch + 1}: Train loss = {train_loss:.6f}")
            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Validate
            if (epoch + 1) % val_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch + 1}: Val loss = {val_loss:.6f}")
                self.writer.add_scalar('loss/val', val_loss, epoch)

                # Snapshot the EMA weights when validation improves; they are
                # written to best_model.pt at the next checkpoint save
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_ema_snapshot = {
                        name: p.detach().clone() for name, p in self.ema_params.items()
                    }

            # Checkpoints are written every save_every completed epochs, plus
            # the final epoch: a numbered file (or latest.pt) and best_model.pt
            # when validation improved since the previous save
            if self.next_epoch % save_every == 0 or self.next_epoch == num_epochs:
                if self.next_epoch % save_every == 0:
                    self.save_checkpoint(f'epoch_{self.next_epoch:04d}.pt')
                else:
                    self.save_checkpoint('latest.pt')
                if self.best_ema_snapshot is not None:
                    ema_params = self.ema_params
                    self.ema_params = self.best_ema_snapshot
                    try:
                        self.save_checkpoint('best_model.pt', update_latest=False)
                    finally:
                        self.ema_params = ema_params
                    self.best_ema_snapshot = None
        
        self.writer.close()
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for path prediction')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--motion_data_dir', type=str, default='../motion_2/data', help='Motion data directory')
    parser.add_argument('--min_prediction_length', type=int, default=120, help='Minimum prediction length (at least 61)')
    parser.add_argument('--max_prediction_length', type=int, default=600, help='Maximum prediction length (padded tensor size)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Total target epochs, including epochs already completed on resume')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--lr_schedule', type=str, default='constant', choices=['constant', 'cosine'],
                        help='Learning rate schedule; cosine anneals after each epoch from lr_start to lr_end')
    parser.add_argument('--lr_start', type=float, default=1e-4, help='Learning rate at the first epoch (cosine schedule)')
    parser.add_argument('--lr_end', type=float, default=2e-5, help='Learning rate at the last epoch (cosine schedule)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_size', type=int, default=None, help='Transformer feedforward dimension (default: 2x hidden_dim)')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path, or latest to load <save_dir>/latest.pt')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=50, help='Save a numbered checkpoint every N completed epochs')
    
    args = parser.parse_args()
    if not MIN_CONSTRAINT_SPACING < args.min_prediction_length <= args.max_prediction_length:
        parser.error('Require 61 <= min_prediction_length <= max_prediction_length')
    if args.num_epochs < 0 or args.val_every < 1 or args.save_every < 1:
        parser.error('num_epochs must be nonnegative; val_every and save_every must be positive')
    if args.lr_start <= 0 or args.lr_end <= 0:
        parser.error('lr_start and lr_end must be positive')
    if args.resume == 'latest':
        args.resume = str(Path(args.save_dir) / 'latest.pt')
    if args.resume:
        PathDiffusionModel.checkpoint_config(torch.load(args.resume, map_location='cpu', weights_only=True))
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, test_loader, norm_stats = prepare_datasets(
        root_features_path=Path(args.data_dir) / 'root_feature_matrix.npy',
        clip_ranges_path=Path(args.motion_data_dir) / 'clip_frame_ranges_world.pkl',
        min_prediction_length=args.min_prediction_length,
        max_prediction_length=args.max_prediction_length,
        batch_size=args.batch_size
    )
    
    # Save normalization stats
    np.save(save_dir / 'normalization_stats.npy', norm_stats)
    
    # Create model
    print("Creating model...")
    max_seq_len = args.max_prediction_length
    
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        pe_dropout=args.dropout,
        norm_first=False,
        max_seq_len=max_seq_len,
        device=device
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        val_every=args.val_every,
        save_every=args.save_every
    )


if __name__ == '__main__':
    main()
