"""
Inference script for path prediction diffusion model.
The model is constraint-conditioned: context only anchors generation through the
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

from diffusion_model import (
    PathDiffusionModel, generate_constraint_mask, validate_constraint_mask, MIN_CONSTRAINT_SPACING
)
from data_preprocessing import denormalize_features


def build_constraints(anchor, prediction_length, constraint_mask=None, constraint_values=None):
    if not isinstance(prediction_length, (int, np.integer)) or prediction_length <= MIN_CONSTRAINT_SPACING:
        raise ValueError('prediction_length must be an integer of at least 61 frames')
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (4,) or not np.isfinite(anchor).all():
        raise ValueError('anchor must be a finite (4,) position and direction')
    if constraint_values is None:
        raise ValueError('constraint_values must provide the real waypoint values')
    constraint_values = np.asarray(constraint_values, dtype=np.float64)
    if constraint_values.shape != (prediction_length, 4) or not np.isfinite(constraint_values).all():
        raise ValueError('constraint_values must be a finite (prediction_length, 4) array')
    if constraint_mask is not None:
        constraint_mask = np.asarray(constraint_mask)
        if constraint_mask.shape != (prediction_length,) or not np.isin(constraint_mask, [0, 1]).all():
            raise ValueError('constraint_mask must be a binary (prediction_length,) array')

    attention = torch.ones(1, prediction_length)
    mask = (generate_constraint_mask(attention)[0].numpy() if constraint_mask is None
            else constraint_mask.astype(np.float32, copy=True))
    mask[[0, -1]] = 1
    validate_constraint_mask(torch.from_numpy(mask)[None], attention)
    values = constraint_values.copy()
    values[0] = anchor
    values[mask == 0] = 0
    return mask, values


class PathPredictor:
    """Path prediction inference wrapper."""
    
    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str,
        device: str = 'cuda',
        max_seq_len: Optional[int] = None
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load normalization stats
        self.norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
        print(f"Loaded normalization stats: {self.norm_stats['method']}")
        
        # Load model
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        
        # Create the mask-conditioned x0 model using the saved architecture and schedule
        model = PathDiffusionModel.from_checkpoint(checkpoint, device=self.device)
        if max_seq_len is not None and max_seq_len != model.max_seq_len:
            raise ValueError('max_seq_len must match the checkpoint; omit it to use the saved value')
        
        self.model = model
        print(f"Model loaded successfully")
        print(f"Trained for {checkpoint['epoch']} epochs")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    def predict(
        self,
        context: np.ndarray,
        prediction_length: int = 120,
        num_samples: int = 5,
        clip_denoised: bool = True,
        max_prediction_length: int = None,
        enforce_final_constraints: bool = True,
        sampling_steps: int = None,
        constraint_mask: np.ndarray = None,
        constraint_values: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate path predictions.

        The model has no context encoder; context only anchors generation via the
        first-point constraint (first predicted point is pinned to the last
        observed frame). constraint_values supplies real target values for every
        selected waypoint. Without an explicit mask, endpoints and random
        interior frames are selected with at least 60 frames between constraints.
        All four features are pinned.

        Args:
            context: (T_context, 4) context trajectory
            prediction_length: Number of frames to predict
            num_samples: Number of diverse predictions to generate
            clip_denoised: Whether to clip denoised predictions
            max_prediction_length: Padded sequence length to run the model at
                                   (if None, defaults to prediction_length). May
                                   exceed the training max_seq_len; the sinusoidal
                                   positional encoding extends to any length.
            constraint_mask: Optional binary (prediction_length,) mask. If absent,
                             randomly select interior indices. Explicit masks are
                             used as-is apart from always pinning both endpoints;
                             all selected indices must be at least 60 frames apart.
            constraint_values: Required raw (prediction_length, 4) reference path
                               containing real waypoint values. Values at unselected
                               frames are ignored. The first frame always uses
                               context[-1].

        Returns:
            (num_samples, prediction_length, 4) predicted trajectories
        """
        if max_prediction_length is None:
            max_prediction_length = prediction_length
        if not MIN_CONSTRAINT_SPACING < prediction_length <= max_prediction_length:
            raise ValueError('Require 61 <= prediction_length <= max_prediction_length')
        context = np.asarray(context)
        if context.ndim != 2 or context.shape[1] != 4 or len(context) == 0:
            raise ValueError('context must have shape (T_context, 4) with at least one frame')
        raw_mask, raw_values = build_constraints(
            context[-1], prediction_length, constraint_mask, constraint_values
        )

        # Normalize context
        context_normalized = (context - self.norm_stats['mean']) / self.norm_stats['std']

        # Attention mask marks valid positions (padding excluded from attention)
        attention_mask = torch.zeros(max_prediction_length)
        attention_mask[:prediction_length] = 1
        attention_mask = attention_mask.unsqueeze(0).to(self.device)  # (1, max_pred_length)

        # Build constraints in the normalized frame: index 0 pinned to the last
        # observed context frame; user constraints merged on top (raw frame).
        mask = raw_mask
        values = (raw_values - self.norm_stats['mean']) / self.norm_stats['std']
        values[mask == 0] = 0
        values[0] = context_normalized[-1]

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
                device=self.device,
                enforce_final_constraints=enforce_final_constraints,
                sampling_steps=sampling_steps
            )
        
        # Trim padding, convert to numpy, denormalize
        samples = samples[:, :, :prediction_length, :].cpu().numpy()
        samples_denorm = denormalize_features(samples, self.norm_stats)
        
        return samples_denorm[0]  # Remove batch dimension
    
    def predict_batch(
        self,
        contexts: np.ndarray,
        prediction_lengths,
        constraint_values: np.ndarray,
        num_samples: int = 5,
        clip_denoised: bool = True,
        max_prediction_length: int = None
    ) -> list:
        """
        Generate predictions with per-context lengths and random interior masks.
        Waypoint values and directions are always taken from constraint_values.

        Args:
            contexts: (B, T_context, 4) context trajectories (anchors first point)
            prediction_lengths: (B,) array of prediction lengths, or a single int for all
            constraint_values: Raw (B, max_prediction_length, 4) reference paths
                               containing real waypoint values.
            num_samples: Number of diverse predictions per context
            clip_denoised: Whether to clip denoised predictions
            max_prediction_length: Padded sequence length to run the model at
                                   (if None, uses max of prediction_lengths). May
                                   exceed the training max_seq_len; the sinusoidal
                                   positional encoding extends to any length.

        Returns:
            List of B arrays, each (num_samples, pred_len_i, 4)
        """
        contexts = np.asarray(contexts)
        if contexts.ndim != 3 or contexts.shape[1] == 0 or contexts.shape[2] != 4:
            raise ValueError('contexts must have shape (B, T_context, 4)')
        B = contexts.shape[0]
        if B == 0:
            raise ValueError('contexts must contain a nonempty batch')

        # Normalize prediction_lengths to a (B,) array
        if np.isscalar(prediction_lengths):
            prediction_lengths = np.full(B, prediction_lengths)
        else:
            prediction_lengths = np.asarray(prediction_lengths)
        
        if prediction_lengths.shape != (B,) or not np.issubdtype(prediction_lengths.dtype, np.integer):
            raise ValueError('prediction_lengths must contain one integer per context')
        if max_prediction_length is None:
            max_prediction_length = int(prediction_lengths.max())
        if (prediction_lengths <= MIN_CONSTRAINT_SPACING).any() or (prediction_lengths > max_prediction_length).any():
            raise ValueError('Require 61 <= each prediction_length <= max_prediction_length')
        if constraint_values is None:
            raise ValueError('constraint_values must provide the real waypoint values')
        reference_values = np.asarray(constraint_values, dtype=np.float64)
        if (reference_values.shape != (B, max_prediction_length, 4)
                or not np.isfinite(reference_values).all()):
            raise ValueError('constraint_values must be a finite (B, max_prediction_length, 4) array')
        
        # Normalize contexts
        contexts_normalized = (contexts - self.norm_stats['mean']) / self.norm_stats['std']

        # Attention masks: 1 up to each context's prediction length, 0 after
        attention_masks = torch.zeros(B, max_prediction_length)
        for i, pred_length in enumerate(prediction_lengths):
            attention_masks[i, :int(pred_length)] = 1
        attention_masks = attention_masks.to(self.device)

        # Endpoint constraints use the context anchor and supplied waypoint values.
        constraint_masks = torch.zeros(B, max_prediction_length)
        constraint_values = torch.zeros(B, max_prediction_length, 4)
        for i, pred_length in enumerate(prediction_lengths):
            length = int(pred_length)
            reference = reference_values[i, :length]
            mask, values = build_constraints(contexts[i, -1], length, constraint_values=reference)
            normalized = (values - self.norm_stats['mean']) / self.norm_stats['std']
            normalized[mask == 0] = 0
            constraint_masks[i, :length] = torch.from_numpy(mask)
            constraint_values[i, :length] = torch.from_numpy(normalized)
        constraint_values[:, 0] = torch.FloatTensor(contexts_normalized[:, -1])
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
    parser.add_argument('--prediction_length', type=int, default=120, help='Prediction sequence length (at least 61)')
    parser.add_argument('--max_prediction_length', type=int, default=None, 
                        help='Maximum prediction length for variable-length model (pads shorter predictions)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of diverse predictions')
    parser.add_argument('--clip_name', type=str, required=True,
                        help='Clip supplying context and waypoint constraint values')
    parser.add_argument('--start_offset', type=int, default=0, help='Offset from clip start')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    
    args = parser.parse_args()
    if args.prediction_length <= MIN_CONSTRAINT_SPACING:
        parser.error('prediction_length must be at least 61')
    if args.max_prediction_length is not None and args.max_prediction_length < args.prediction_length:
        parser.error('max_prediction_length cannot be smaller than prediction_length')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictor
    predictor = PathPredictor(
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        device=args.device
    )
    
    if not args.clip_name:
        parser.error('--clip_name is required to supply real waypoint constraint values')

    print(f"Loading context from clip: {args.clip_name}")
    window = load_context_from_clip(
        root_features_path=Path(args.data_dir) / 'root_feature_matrix.npy',
        clip_ranges_path=Path(args.motion_data_dir) / 'clip_frame_ranges_world.pkl',
        clip_name=args.clip_name,
        context_length=args.context_length + args.prediction_length - 1,
        start_offset=args.start_offset
    )
    context = window[:args.context_length]
    constraint_values = window[args.context_length - 1:]

    print(f"Context shape: {context.shape}, waypoint values shape: {constraint_values.shape}")

    # Generate predictions
    print(f"Generating {args.num_samples} predictions...")
    predictions = predictor.predict(
        context=context,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        max_prediction_length=args.max_prediction_length,
        constraint_values=constraint_values
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
