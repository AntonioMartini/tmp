"""
Inference script for path prediction diffusion model.
The model is unconditional: the context only anchors generation through the
first-point constraint (first predicted point = last observed frame).
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from diffusion_model import PathDiffusionModel
from data_preprocessing import denormalize_features


class PathPredictor:
    """Path prediction inference wrapper."""
    
    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str,
        device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load normalization stats
        self.norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
        print(f"Loaded normalization stats: {self.norm_stats['method']}")
        
        # Load model
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model (extract hyperparameters from checkpoint if available, otherwise use defaults)
        model = PathDiffusionModel(
            feature_dim=4,
            num_timesteps=1000,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1,
            pe_dropout=0.1,
            norm_first=False,
            device=self.device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        print(f"Model loaded successfully")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    def predict(
        self,
        context: np.ndarray,
        prediction_length: int = 60,
        num_samples: int = 5,
        clip_denoised: bool = True,
        max_prediction_length: int = None,
        constraint_mask: np.ndarray = None,
        constraint_values: np.ndarray = None,
        goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate path predictions.

        The model is unconditional; the context only anchors generation via the
        first-point constraint (first predicted point is pinned to the last
        observed frame). The last predicted frame must always be constrained
        too — either via `goal` or by constraining index `prediction_length - 1`
        in `constraint_mask`.

        Args:
            context: (T_context, 4) context trajectory
            prediction_length: Number of frames to predict
            num_samples: Number of diverse predictions to generate
            clip_denoised: Whether to clip denoised predictions
            max_prediction_length: Padded length the model was trained with
                                   (if None, defaults to prediction_length)
            constraint_mask: Optional (prediction_length,) mask, 1 at constrained
                             positions. Index 0 is always constrained to the last
                             context frame; additional indices are merged in.
            constraint_values: Optional (prediction_length, 4) values at constrained
                               positions, in the raw (unnormalized) frame.
                               Required if constraint_mask constrains any index
                               other than 0.
            goal: Optional (4,) raw-frame value for the last predicted frame.
                  Convenience wrapper for constraining index
                  `prediction_length - 1`.

        Returns:
            (num_samples, prediction_length, 4) predicted trajectories
        """
        if max_prediction_length is None:
            max_prediction_length = prediction_length

        # Normalize context
        context_normalized = (context - self.norm_stats['mean']) / self.norm_stats['std']

        # Attention mask marks valid positions (padding excluded from attention)
        attention_mask = torch.zeros(max_prediction_length)
        attention_mask[:prediction_length] = 1
        attention_mask = attention_mask.unsqueeze(0).to(self.device)  # (1, max_pred_length)

        # Build constraints in the normalized frame: index 0 pinned to the last
        # observed context frame; user constraints merged on top (raw frame).
        mask = np.zeros(prediction_length)
        mask[0] = 1
        values = np.zeros((prediction_length, 4))
        values[0] = context_normalized[-1]
        if constraint_mask is not None:
            user_mask = np.asarray(constraint_mask).astype(bool)
            mask[user_mask] = 1
            if constraint_values is not None:
                values_norm = (np.asarray(constraint_values)
                               - self.norm_stats['mean']) / self.norm_stats['std']
                values[user_mask] = values_norm[user_mask]
            elif user_mask[1:].any():
                raise ValueError("constraint_values must be provided when "
                                 "constraint_mask constrains positions beyond index 0")
        if goal is not None:
            mask[prediction_length - 1] = 1
            values[prediction_length - 1] = (
                (np.asarray(goal) - self.norm_stats['mean']) / self.norm_stats['std']
            )
        if mask[prediction_length - 1] != 1:
            raise ValueError(
                "The last predicted frame must be constrained (model is trained "
                "with first/last always constrained): pass goal= or constrain "
                "index prediction_length - 1 in constraint_mask"
            )

        constraint_mask_padded = np.zeros(max_prediction_length)
        constraint_mask_padded[:prediction_length] = mask
        constraint_mask_tensor = torch.FloatTensor(constraint_mask_padded).unsqueeze(0).to(self.device)
        values_padded = np.zeros((max_prediction_length, 4))
        values_padded[:prediction_length] = values
        constraint_values_tensor = torch.FloatTensor(values_padded).unsqueeze(0).to(self.device)

        with torch.no_grad():
            samples = self.model.sample(
                prediction_length=max_prediction_length,
                num_samples=num_samples,
                clip_denoised=clip_denoised,
                constraint_mask=constraint_mask_tensor,
                constraint_values=constraint_values_tensor,
                attention_mask=attention_mask,
                device=self.device
            )
        
        # Trim padding, convert to numpy, denormalize
        samples = samples[:, :, :prediction_length, :].cpu().numpy()
        samples_denorm = denormalize_features(samples, self.norm_stats)
        
        return samples_denorm[0]  # Remove batch dimension
    
    def predict_batch(
        self,
        contexts: np.ndarray,
        prediction_lengths,
        goals: np.ndarray,
        num_samples: int = 5,
        clip_denoised: bool = True,
        max_prediction_length: int = None
    ) -> list:
        """
        Generate predictions for multiple contexts with per-context lengths.

        Args:
            contexts: (B, T_context, 4) context trajectories (anchors first point)
            prediction_lengths: (B,) array of prediction lengths, or a single int for all
            goals: (B, 4) raw-frame values for each context's last predicted frame
            num_samples: Number of diverse predictions per context
            clip_denoised: Whether to clip denoised predictions
            max_prediction_length: Padded length the model was trained with
                                   (if None, uses max of prediction_lengths)

        Returns:
            List of B arrays, each (num_samples, pred_len_i, 4)
        """
        B = contexts.shape[0]
        goals = np.asarray(goals)
        assert goals.shape == (B, 4), f"goals must have shape ({B}, 4), got {goals.shape}"
        
        # Normalize prediction_lengths to a (B,) array
        if np.isscalar(prediction_lengths):
            prediction_lengths = np.full(B, prediction_lengths)
        else:
            prediction_lengths = np.asarray(prediction_lengths)
        
        if max_prediction_length is None:
            max_prediction_length = int(prediction_lengths.max())
        
        # Normalize contexts
        contexts_normalized = (contexts - self.norm_stats['mean']) / self.norm_stats['std']

        # Attention masks: 1 up to each context's prediction length, 0 after
        attention_masks = torch.zeros(B, max_prediction_length)
        for i, pred_length in enumerate(prediction_lengths):
            attention_masks[i, :int(pred_length)] = 1
        attention_masks = attention_masks.to(self.device)

        # Endpoint constraints: first point pinned to the last observed frame,
        # last valid point pinned to the given goal (normalized frame)
        constraint_masks = torch.zeros(B, max_prediction_length)
        constraint_masks[:, 0] = 1
        constraint_values = torch.zeros(B, max_prediction_length, 4)
        constraint_values[:, 0] = torch.FloatTensor(contexts_normalized[:, -1])
        goals_norm = (goals - self.norm_stats['mean']) / self.norm_stats['std']
        for i, pred_length in enumerate(prediction_lengths):
            last = int(pred_length) - 1
            constraint_masks[i, last] = 1
            constraint_values[i, last] = torch.FloatTensor(goals_norm[i])
        constraint_masks = constraint_masks.to(self.device)
        constraint_values = constraint_values.to(self.device)

        # Generate samples with zero-padding to max_prediction_length
        with torch.no_grad():
            samples = self.model.sample(
                prediction_length=max_prediction_length,
                num_samples=num_samples,
                clip_denoised=clip_denoised,
                constraint_mask=constraint_masks,
                constraint_values=constraint_values,
                attention_mask=attention_masks,
                device=self.device
            )
        
        # Denormalize the padded batch, then trim each to its actual length
        samples_denorm = denormalize_features(samples.cpu().numpy(), self.norm_stats)
        
        return [
            samples_denorm[i, :, :int(pred_length), :]
            for i, pred_length in enumerate(prediction_lengths)
        ]


def load_context_from_clip(
    root_features_path: str,
    clip_ranges_path: str,
    clip_name: str,
    context_length: int = 30,
    start_offset: int = 0
) -> np.ndarray:
    """
    Load context trajectory from a specific clip.
    
    Args:
        root_features_path: Path to root_feature_matrix.npy
        clip_ranges_path: Path to clip_frame_ranges_world.pkl
        clip_name: Name of the clip to load
        context_length: Number of frames to use as context
        start_offset: Offset from clip start
        
    Returns:
        (context_length, 4) context trajectory
    """
    # Load data
    root_features = np.load(root_features_path)
    
    with open(clip_ranges_path, 'rb') as f:
        clip_ranges = pickle.load(f)
    
    # Get clip range
    if clip_name not in clip_ranges:
        raise ValueError(f"Clip {clip_name} not found in clip ranges")
    
    start, end = clip_ranges[clip_name]
    context_start = start + start_offset
    context_end = context_start + context_length
    
    if context_end > end:
        raise ValueError(f"Context range [{context_start}, {context_end}) exceeds clip range [{start}, {end})")
    
    # Extract context
    context = root_features[context_start:context_end]
    
    return context


def visualize_predictions(
    context: np.ndarray,
    predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize context and predicted trajectories.
    
    Args:
        context: (T_context, 4) context trajectory
        predictions: (num_samples, T_pred, 4) predicted trajectories
        ground_truth: Optional (T_pred, 4) ground truth trajectory
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectory in X-Z plane
    ax1 = axes[0]
    
    # Plot context
    ax1.plot(context[:, 0], context[:, 1], 'b-', linewidth=2, label='Context', alpha=0.7)
    ax1.scatter(context[0, 0], context[0, 1], c='blue', s=100, marker='o', label='Context Start')
    ax1.scatter(context[-1, 0], context[-1, 1], c='green', s=100, marker='s', label='Context End')
    
    # Plot predictions
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions)))
    for i, pred in enumerate(predictions):
        ax1.plot(pred[:, 0], pred[:, 1], color=colors[i], linewidth=1.5, 
                alpha=0.6, label=f'Prediction {i+1}' if i < 3 else None)
        ax1.scatter(pred[-1, 0], pred[-1, 1], color=colors[i], s=50, marker='^', alpha=0.6)
    
    # Plot ground truth if available
    if ground_truth is not None:
        ax1.plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', linewidth=2, label='Ground Truth')
        ax1.scatter(ground_truth[-1, 0], ground_truth[-1, 1], c='red', s=100, marker='*', label='GT End')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Z Position')
    ax1.set_title('Trajectory Prediction (Top View)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Forward direction over time
    ax2 = axes[1]
    
    # Extend time axis for context + predictions
    context_time = np.arange(len(context))
    total_length = len(context) + predictions.shape[1]
    prediction_time = np.arange(len(context), total_length)
    
    # Plot context forward direction
    ax2.plot(context_time, context[:, 2], 'b-', linewidth=2, label='Context Fwd X', alpha=0.7)
    ax2.plot(context_time, context[:, 3], 'b--', linewidth=2, label='Context Fwd Z', alpha=0.7)
    
    # Plot prediction forward directions
    for i, pred in enumerate(predictions):
        ax2.plot(prediction_time, pred[:, 2], color=colors[i], linewidth=1.5, alpha=0.6)
        ax2.plot(prediction_time, pred[:, 3], color=colors[i], linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Plot ground truth if available
    if ground_truth is not None:
        ax2.plot(prediction_time, ground_truth[:, 2], 'k-', linewidth=2, label='GT Fwd X')
        ax2.plot(prediction_time, ground_truth[:, 3], 'k--', linewidth=2, label='GT Fwd Z')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Forward Direction')
    ax2.set_title('Forward Direction Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate path predictions using trained diffusion model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--norm_stats', type=str, required=True, help='Path to normalization stats')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--motion_data_dir', type=str, default='../motion_2/data', help='Motion data directory')
    parser.add_argument('--context_length', type=int, default=30, help='Context frames to load (anchors the first-point constraint)')
    parser.add_argument('--prediction_length', type=int, default=60, help='Prediction sequence length')
    parser.add_argument('--max_prediction_length', type=int, default=None, 
                        help='Maximum prediction length for variable-length model (pads shorter predictions)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of diverse predictions')
    parser.add_argument('--clip_name', type=str, default=None, help='Specific clip to use for context')
    parser.add_argument('--start_offset', type=int, default=0, help='Offset from clip start')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictor
    predictor = PathPredictor(
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        device=args.device
    )
    
    # Load context (and goal = the true future endpoint, if using a clip)
    if args.clip_name:
        print(f"Loading context from clip: {args.clip_name}")
        window = load_context_from_clip(
            root_features_path=Path(args.data_dir) / 'root_feature_matrix.npy',
            clip_ranges_path=Path(args.motion_data_dir) / 'clip_frame_ranges_world.pkl',
            clip_name=args.clip_name,
            context_length=args.context_length + args.prediction_length,
            start_offset=args.start_offset
        )
        # Last frame of the window is the goal; the rest is context
        goal = window[-1]
        context = window[:args.context_length]
    else:
        # Random context; goal defaults to the last context frame
        print("Loading random context...")
        context = np.random.randn(args.context_length, 4)
        goal = context[-1]

    print(f"Context shape: {context.shape}, goal: {goal}")

    # Generate predictions
    print(f"Generating {args.num_samples} predictions...")
    predictions = predictor.predict(
        context=context,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        max_prediction_length=args.max_prediction_length,
        goal=goal
    )
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Save predictions
    pred_path = output_dir / 'predictions.npy'
    np.save(pred_path, predictions)
    print(f"Saved predictions to {pred_path}")
    
    # Also save context
    context_path = output_dir / 'context.npy'
    np.save(context_path, context)
    print(f"Saved context to {context_path}")
    
    # Visualize
    if args.visualize:
        viz_path = output_dir / 'visualization.png'
        visualize_predictions(context, predictions, save_path=str(viz_path))
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
