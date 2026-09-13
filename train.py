"""
Training script for path prediction diffusion model.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_preprocessing import prepare_datasets
from diffusion_model import PathDiffusionModel


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
        save_dir: str = './checkpoints'
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

        # Optimizer: Adam-atan2 with constant LR (Kimodo recipe, no scheduler)
        self.optimizer = AdamAtan2(
            model.parameters(),
            lr=learning_rate,
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
        self.best_val_loss = float('inf')

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
        """Compute diffusion loss, masked over valid (non-padded) positions.

        The first and last valid point of each sequence are always constrained:
        no noise is added at those positions (Kimodo-style constraints).
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
        constraint_mask = torch.zeros_like(attention_mask)
        constraint_mask[:, 0] = 1
        last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
        constraint_mask[torch.arange(batch_size, device=self.device), last_idx] = 1

        # Sample noise and enforce constraints (zero noise at constrained positions)
        noise = torch.randn_like(target)
        noise = noise * (1 - constraint_mask.unsqueeze(-1))
        noisy_target = self.model.diffusion.q_sample(target, timesteps, noise)

        # Forward pass
        noise_pred = self.model.denoiser(
            noisy_target, timesteps, attention_mask=attention_mask
        )

        # Masked MSE loss: padded positions contribute nothing
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(noise_pred)
        loss = nn.functional.mse_loss(
            noise_pred * mask_expanded, noise * mask_expanded, reduction='sum'
        ) / mask_expanded.sum()

        return loss
    
    @torch.no_grad()
    def _update_ema(self):
        """Update the exponential moving average of model parameters."""
        for name, p in self.model.named_parameters():
            self.ema_params[name].mul_(self.ema_decay).add_(
                p.detach(), alpha=1 - self.ema_decay
            )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
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
    
    def save_checkpoint(self, filename: str = 'best_model.pt'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            # EMA weights are saved as model_state_dict: they are what gets
            # deployed at test time
            'model_state_dict': dict(self.ema_params),
            'raw_model_state_dict': self.model.state_dict(),
            'ema_state_dict': dict(self.ema_params),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Backward compatible: older checkpoints only have 'model_state_dict'
        self.model.load_state_dict(
            checkpoint.get('raw_model_state_dict', checkpoint['model_state_dict'])
        )
        if 'ema_state_dict' in checkpoint:
            self.ema_params = {
                k: v.to(self.device) for k, v in checkpoint['ema_state_dict'].items()
            }
        else:
            self.ema_params = {
                name: p.detach().clone() for name, p in self.model.named_parameters()
            }
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, num_epochs: int, val_every: int = 1):
        """Train the model for multiple epochs."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Optimizer: Adam-atan2, constant lr = {self.optimizer.param_groups[0]['lr']}")
        print(f"EMA: decay = {self.ema_decay}, every {self.ema_interval} steps")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train loss = {train_loss:.6f}")
            self.writer.add_scalar('loss/train', train_loss, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Validate
            if (epoch + 1) % val_every == 0:
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val loss = {val_loss:.6f}")
                self.writer.add_scalar('loss/val', val_loss, epoch)

                # Save best model only
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
        
        self.writer.close()
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for path prediction')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--motion_data_dir', type=str, default='../motion_2/data', help='Motion data directory')
    parser.add_argument('--min_prediction_length', type=int, default=20, help='Minimum prediction length')
    parser.add_argument('--max_prediction_length', type=int, default=600, help='Maximum prediction length (padded tensor size)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    
    args = parser.parse_args()
    
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
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        val_every=args.val_every
    )


if __name__ == '__main__':
    main()
