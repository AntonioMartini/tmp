"""
Evaluation script for path prediction diffusion model.
Computes metrics and generates comprehensive visualizations.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from data_preprocessing import PathDataset, prepare_datasets, denormalize_features
from diffusion_model import PathDiffusionModel


class PathEvaluator:
    """Evaluator for path prediction model."""
    
    def __init__(
        self,
        model: PathDiffusionModel,
        norm_stats: dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.norm_stats = norm_stats
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def compute_ade(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        valid_lengths: np.ndarray
    ) -> float:
        """
        Compute Average Displacement Error (ADE).
        Average Euclidean distance over valid (non-padded) positions only.
        """
        pred_pos = predictions[..., :2]
        gt_pos = ground_truth[..., :2]
        distances = np.linalg.norm(pred_pos - gt_pos, axis=-1)  # (B, T)
        
        total_error = 0.0
        total_count = 0
        for i in range(len(valid_lengths)):
            length = int(valid_lengths[i])
            total_error += distances[i, :length].sum()
            total_count += length
        
        return total_error / max(total_count, 1)
    
    @torch.no_grad()
    def compute_fde(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        valid_lengths: np.ndarray
    ) -> float:
        """
        Compute Final Displacement Error (FDE).
        Euclidean distance at the last valid position per sample.
        """
        distances = []
        for i in range(len(valid_lengths)):
            last_idx = int(valid_lengths[i]) - 1
            pred_pos = predictions[i, last_idx, :2]
            gt_pos = ground_truth[i, last_idx, :2]
            distances.append(np.linalg.norm(pred_pos - gt_pos))
        
        return np.mean(distances)
    
    @torch.no_grad()
    def compute_direction_error(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        valid_lengths: np.ndarray
    ) -> float:
        """
        Compute average angular error in forward direction over valid positions.
        """
        pred_dir = predictions[..., 2:]  # (forward_x, forward_z)
        gt_dir = ground_truth[..., 2:]
        
        # Normalize directions
        pred_dir = pred_dir / (np.linalg.norm(pred_dir, axis=-1, keepdims=True) + 1e-8)
        gt_dir = gt_dir / (np.linalg.norm(gt_dir, axis=-1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        dot_product = np.sum(pred_dir * gt_dir, axis=-1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Convert to angular error in degrees
        angular_error = np.arccos(dot_product) * 180.0 / np.pi  # (B, T)
        
        total_error = 0.0
        total_count = 0
        for i in range(len(valid_lengths)):
            length = int(valid_lengths[i])
            total_error += angular_error[i, :length].sum()
            total_count += length
        
        return total_error / max(total_count, 1)
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        target: torch.Tensor,
        attention_mask: torch.Tensor,
        actual_lengths: torch.Tensor,
        num_samples: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate on a single batch.
        Metrics are computed over valid (non-padded) positions only.
        The model is unconditional: the first and last valid point are
        constrained to ground truth, matching the training task.
        """
        # Normalize target
        target_np = target.cpu().numpy()
        target_norm = (target_np - self.norm_stats['mean']) / self.norm_stats['std']

        # Convert to tensors
        target_tensor = torch.FloatTensor(target_norm).to(self.device)
        attn_mask_tensor = attention_mask.to(self.device)
        valid_lengths_np = actual_lengths.cpu().numpy()

        # Endpoint constraints: first and last valid point pinned to ground truth
        B = target.shape[0]
        constraint_mask = torch.zeros_like(attn_mask_tensor)
        constraint_mask[:, 0] = 1
        last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
        constraint_mask[torch.arange(B, device=self.device), last_idx] = 1

        # Generate predictions
        samples = self.model.sample(
            prediction_length=target.shape[1],
            num_samples=num_samples,
            constraint_mask=constraint_mask,
            constraint_values=target_tensor,
            attention_mask=attn_mask_tensor,
            device=self.device
        )
        
        # Convert to numpy and denormalize
        samples_np = samples.cpu().numpy()
        samples_denorm = denormalize_features(samples_np, self.norm_stats)
        target_denorm = denormalize_features(target_norm, self.norm_stats)
        
        # Compute metrics for each sample, then average
        ades = []
        fdes = []
        dir_errors = []
        
        for i in range(num_samples):
            pred = samples_denorm[:, i]  # (B, T, 4)
            gt = target_denorm  # (B, T, 4)
            
            ades.append(self.compute_ade(pred, gt, valid_lengths_np))
            fdes.append(self.compute_fde(pred, gt, valid_lengths_np))
            dir_errors.append(self.compute_direction_error(pred, gt, valid_lengths_np))
        
        return {
            'ade': np.mean(ades),
            'fde': np.mean(fdes),
            'direction_error': np.mean(dir_errors)
        }
    
    def evaluate_dataset(
        self,
        dataloader,
        num_batches: int = None,
        num_samples: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate on entire dataset.
        """
        all_metrics = []
        
        num_iter = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
        
        for i, batch in enumerate(tqdm(dataloader, total=num_iter, desc="Evaluating")):
            if i >= num_iter:
                break
            
            metrics = self.evaluate_batch(
                batch['target'],
                attention_mask=batch['attention_mask'],
                actual_lengths=batch['actual_length'],
                num_samples=num_samples
            )
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {
            'ade': np.mean([m['ade'] for m in all_metrics]),
            'fde': np.mean([m['fde'] for m in all_metrics]),
            'direction_error': np.mean([m['direction_error'] for m in all_metrics])
        }
        
        return aggregated


def create_comprehensive_visualization(
    context: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str,
    title: str = "Path Prediction Results"
):
    """
    Create comprehensive visualization with multiple subplots.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Plot 1: Top-down trajectory view
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot context
    ax1.plot(context[:, 0], context[:, 1], 'b-', linewidth=3, label='Context', alpha=0.8)
    ax1.scatter(context[0, 0], context[0, 1], c='blue', s=150, marker='o', label='Start', zorder=5)
    ax1.scatter(context[-1, 0], context[-1, 1], c='green', s=150, marker='s', label='Context End', zorder=5)
    
    # Plot predictions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    for i, pred in enumerate(predictions):
        alpha = 0.7 if i == 0 else 0.4
        linewidth = 2.5 if i == 0 else 1.5
        ax1.plot(pred[:, 0], pred[:, 1], color=colors[i], linewidth=linewidth, 
                alpha=alpha, label=f'Pred {i+1}' if i < 3 else None)
        ax1.scatter(pred[-1, 0], pred[-1, 1], color=colors[i], s=80, marker='^', alpha=alpha, zorder=4)
    
    # Plot ground truth
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', linewidth=2.5, label='Ground Truth')
    ax1.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='red', s=150, marker='*', label='GT End', zorder=5)
    
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Z Position', fontsize=12)
    ax1.set_title('Trajectory (Top View)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: X position over time
    ax2 = fig.add_subplot(gs[0, 1])
    
    context_time = np.arange(len(context))
    total_length = len(context) + len(predictions[0])
    prediction_time = np.arange(len(context), total_length)
    
    ax2.plot(context_time, context[:, 0], 'b-', linewidth=2.5, label='Context', alpha=0.8)
    
    for i, pred in enumerate(predictions):
        alpha = 0.7 if i == 0 else 0.4
        linewidth = 2 if i == 0 else 1
        ax2.plot(prediction_time, pred[:, 0], color=colors[i], linewidth=linewidth, alpha=alpha)
    
    ax2.plot(prediction_time, ground_truth[:, 0], 'k--', linewidth=2.5, label='Ground Truth')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('X Position', fontsize=12)
    ax2.set_title('X Position Over Time', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Z position over time
    ax3 = fig.add_subplot(gs[0, 2])
    
    ax3.plot(context_time, context[:, 1], 'b-', linewidth=2.5, label='Context', alpha=0.8)
    
    for i, pred in enumerate(predictions):
        alpha = 0.7 if i == 0 else 0.4
        linewidth = 2 if i == 0 else 1
        ax3.plot(prediction_time, pred[:, 1], color=colors[i], linewidth=linewidth, alpha=alpha)
    
    ax3.plot(prediction_time, ground_truth[:, 1], 'k--', linewidth=2.5, label='Ground Truth')
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Z Position', fontsize=12)
    ax3.set_title('Z Position Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Forward X over time
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.plot(context_time, context[:, 2], 'b-', linewidth=2.5, label='Context', alpha=0.8)
    
    for i, pred in enumerate(predictions):
        alpha = 0.7 if i == 0 else 0.4
        linewidth = 2 if i == 0 else 1
        ax4.plot(prediction_time, pred[:, 2], color=colors[i], linewidth=linewidth, alpha=alpha)
    
    ax4.plot(prediction_time, ground_truth[:, 2], 'k--', linewidth=2.5, label='Ground Truth')
    ax4.set_xlabel('Time Step', fontsize=12)
    ax4.set_ylabel('Forward X', fontsize=12)
    ax4.set_title('Forward Direction X', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Forward Z over time
    ax5 = fig.add_subplot(gs[1, 1])
    
    ax5.plot(context_time, context[:, 3], 'b-', linewidth=2.5, label='Context', alpha=0.8)
    
    for i, pred in enumerate(predictions):
        alpha = 0.7 if i == 0 else 0.4
        linewidth = 2 if i == 0 else 1
        ax5.plot(prediction_time, pred[:, 3], color=colors[i], linewidth=linewidth, alpha=alpha)
    
    ax5.plot(prediction_time, ground_truth[:, 3], 'k--', linewidth=2.5, label='Ground Truth')
    ax5.set_xlabel('Time Step', fontsize=12)
    ax5.set_ylabel('Forward Z', fontsize=12)
    ax5.set_title('Forward Direction Z', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error over time (for first prediction)
    ax6 = fig.add_subplot(gs[1, 2])
    
    pred0 = predictions[0]
    position_error = np.linalg.norm(pred0[:, :2] - ground_truth[:, :2], axis=1)
    direction_error = np.arccos(np.clip(
        np.sum(pred0[:, 2:] * ground_truth[:, 2:], axis=1) /
        (np.linalg.norm(pred0[:, 2:], axis=1) * np.linalg.norm(ground_truth[:, 2:], axis=1) + 1e-8),
        -1, 1
    )) * 180.0 / np.pi
    
    ax6.plot(prediction_time, position_error, 'r-', linewidth=2, label='Position Error')
    ax6_twin = ax6.twinx()
    ax6_twin.plot(prediction_time, direction_error, 'g-', linewidth=2, label='Direction Error')
    
    ax6.set_xlabel('Time Step', fontsize=12)
    ax6.set_ylabel('Position Error', fontsize=12, color='red')
    ax6_twin.set_ylabel('Direction Error (degrees)', fontsize=12, color='green')
    ax6.set_title('Error Over Time (First Prediction)', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='y', labelcolor='red')
    ax6_twin.tick_params(axis='y', labelcolor='green')
    ax6.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive visualization to {save_path}")
    plt.close()


def create_trajectory_animation(
    context: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    save_path: str
):
    """
    Create an animation showing trajectory evolution.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up plot
    all_x = np.concatenate([context[:, 0], predictions[0, :, 0], ground_truth[:, 0]])
    all_z = np.concatenate([context[:, 1], predictions[0, :, 1], ground_truth[:, 1]])
    
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    z_range = all_z.max() - all_z.min()
    max_range = max(x_range, z_range)
    
    ax.set_xlim(all_x.min() - margin * max_range, all_x.max() + margin * max_range)
    ax.set_ylim(all_z.min() - margin * max_range, all_z.max() + margin * max_range)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_title('Trajectory Prediction Animation')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Initialize plot elements
    context_line, = ax.plot([], [], 'b-', linewidth=3, label='Context')
    pred_lines = [ax.plot([], [], linewidth=2, alpha=0.6)[0] for _ in predictions]
    gt_line, = ax.plot([], [], 'k--', linewidth=2.5, label='Ground Truth')
    
    # Current position markers
    context_marker, = ax.plot([], [], 'bo', markersize=10, zorder=5)
    pred_markers = [ax.plot([], [], '^', markersize=8, zorder=4)[0] for _ in predictions]
    gt_marker, = ax.plot([], [], 'r*', markersize=12, zorder=5)
    
    ax.legend()
    
    total_frames = len(context) + len(predictions[0])
    
    def update(frame):
        # Update context
        if frame < len(context):
            context_line.set_data(context[:frame+1, 0], context[:frame+1, 1])
            context_marker.set_data([context[frame, 0]], [context[frame, 1]])
        else:
            context_line.set_data(context[:, 0], context[:, 1])
            context_marker.set_data([context[-1, 0]], [context[-1, 1]])
        
        # Update predictions
        pred_frame = frame - len(context)
        if pred_frame >= 0:
            for i, pred in enumerate(predictions):
                if pred_frame < len(pred):
                    pred_lines[i].set_data(pred[:pred_frame+1, 0], pred[:pred_frame+1, 1])
                    pred_markers[i].set_data([pred[pred_frame, 0]], [pred[pred_frame, 1]])
                else:
                    pred_lines[i].set_data(pred[:, 0], pred[:, 1])
                    pred_markers[i].set_data([pred[-1, 0]], [pred[-1, 1]])
        
        # Update ground truth
        if pred_frame >= 0:
            if pred_frame < len(ground_truth):
                gt_line.set_data(ground_truth[:pred_frame+1, 0], ground_truth[:pred_frame+1, 1])
                gt_marker.set_data([ground_truth[pred_frame, 0]], [ground_truth[pred_frame, 1]])
            else:
                gt_line.set_data(ground_truth[:, 0], ground_truth[:, 1])
                gt_marker.set_data([ground_truth[-1, 0]], [ground_truth[-1, 1]])
        
        return [context_line, context_marker, gt_line, gt_marker] + pred_lines + pred_markers
    
    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
    
    anim.save(save_path, writer='pillow', fps=20)
    print(f"Saved animation to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate path prediction diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--norm_stats', type=str, required=True, help='Path to normalization stats')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--motion_data_dir', type=str, default='../motion_2/data', help='Motion data directory')
    parser.add_argument('--min_prediction_length', type=int, default=20, help='Minimum prediction length (must match training)')
    parser.add_argument('--max_prediction_length', type=int, default=600, help='Maximum prediction length (must match training)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for evaluation')
    parser.add_argument('--num_eval_batches', type=int, default=None, help='Number of batches to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./evaluation', help='Output directory')
    parser.add_argument('--create_visualizations', action='store_true', help='Create visualizations')
    parser.add_argument('--num_visualizations', type=int, default=5, help='Number of visualizations to create')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load normalization stats
    norm_stats = np.load(args.norm_stats, allow_pickle=True).item()
    print(f"Loaded normalization stats: {norm_stats['method']}")
    
    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    max_seq_len = args.max_prediction_length
    
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        pe_dropout=0.1,
        norm_first=False,
        max_seq_len=max_seq_len,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded successfully")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    # Prepare test data
    print("Preparing test data...")
    _, _, test_loader, _ = prepare_datasets(
        root_features_path=Path(args.data_dir) / 'root_feature_matrix.npy',
        clip_ranges_path=Path(args.motion_data_dir) / 'clip_frame_ranges_world.pkl',
        min_prediction_length=args.min_prediction_length,
        max_prediction_length=args.max_prediction_length,
        batch_size=args.batch_size
    )
    
    # Evaluate
    print("Evaluating model...")
    evaluator = PathEvaluator(model, norm_stats, device)
    
    metrics = evaluator.evaluate_dataset(
        test_loader,
        num_batches=args.num_eval_batches,
        num_samples=args.num_samples
    )
    
    print("\nEvaluation Results:")
    print(f"Average Displacement Error (ADE): {metrics['ade']:.4f}")
    print(f"Final Displacement Error (FDE): {metrics['fde']:.4f}")
    print(f"Average Direction Error: {metrics['direction_error']:.4f} degrees")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"Average Displacement Error (ADE): {metrics['ade']:.4f}\n")
        f.write(f"Final Displacement Error (FDE): {metrics['fde']:.4f}\n")
        f.write(f"Average Direction Error: {metrics['direction_error']:.4f} degrees\n")
    print(f"Saved metrics to {metrics_path}")
    
    # Create visualizations
    if args.create_visualizations:
        print(f"\nCreating {args.num_visualizations} visualizations...")
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(args.num_visualizations):
            batch = next(iter(test_loader))

            # Get first sample from batch
            target = batch['target'][0].cpu().numpy()
            attention_mask = batch['attention_mask'][0].unsqueeze(0).to(device)
            valid_len = int(batch['actual_length'][0])

            # Denormalize (only valid positions)
            target_denorm = denormalize_features(target[:valid_len], norm_stats)
            # The first GT point acts as the "context" anchor for visualization
            context_denorm = target_denorm[:1]

            # Generate predictions: first and last valid point constrained to GT
            target_norm = (target - norm_stats['mean']) / norm_stats['std']
            constraint_mask = torch.zeros(1, target.shape[0], device=device)
            constraint_mask[0, 0] = 1
            constraint_mask[0, valid_len - 1] = 1
            constraint_values = torch.FloatTensor(target_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                samples = model.sample(
                    prediction_length=target.shape[0],
                    num_samples=args.num_samples,
                    constraint_mask=constraint_mask,
                    constraint_values=constraint_values,
                    attention_mask=attention_mask,
                    device=device
                )

            samples_np = samples.cpu().numpy()
            samples_denorm = denormalize_features(samples_np, norm_stats)[0]
            # Trim predictions to valid length
            samples_denorm = samples_denorm[:, :valid_len, :]
            
            # Create visualization
            viz_path = viz_dir / f'visualization_{i:03d}.png'
            create_comprehensive_visualization(
                context_denorm,
                samples_denorm,
                target_denorm,
                str(viz_path),
                title=f"Path Prediction Sample {i+1}"
            )
            
            # Create animation for first sample
            if i == 0:
                anim_path = viz_dir / 'animation.gif'
                create_trajectory_animation(
                    context_denorm,
                    samples_denorm,
                    target_denorm,
                    str(anim_path)
                )
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
