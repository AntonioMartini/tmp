"""
Quick test script to verify the diffusion model implementation.
"""

import torch
import numpy as np
from pathlib import Path

from data_preprocessing import prepare_datasets
from diffusion_model import PathDiffusionModel

def test_pipeline():
    """Test the complete pipeline with a small model."""
    print("Testing diffusion model pipeline...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare small dataset for testing
    print("\n1. Preparing data...")
    train_loader, val_loader, test_loader, norm_stats = prepare_datasets(
        root_features_path=Path('./data/root_feature_matrix.npy'),
        clip_ranges_path=Path('../motion_2/data/clip_frame_ranges_world.pkl'),
        min_prediction_length=15,
        max_prediction_length=20,
        batch_size=4
    )
    
    # Create small model for testing
    print("\n2. Creating model...")
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=50,  # Small number for testing
        hidden_dim=64,    # Small hidden dimension
        num_layers=2,     # Few layers
        num_heads=4,      # Few heads
        dropout=0.1,
        pe_dropout=0.1,
        norm_first=False,
        device=device
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    batch = next(iter(train_loader))
    target = batch['target'].to(device)

    print(f"Target shape: {target.shape}")

    attention_mask = batch['attention_mask'].to(device)
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Actual lengths: {batch['actual_length'].tolist()}")

    # Random timesteps
    timesteps = torch.randint(0, 50, (target.shape[0],), device=device)

    # Forward pass
    noise_pred = model(target, timesteps, attention_mask=attention_mask)
    print(f"Noise prediction shape: {noise_pred.shape}")

    # Test sampling (first point constrained to the target's first point,
    # which also anchors the canonicalization frame)
    print("\n4. Testing sampling...")
    model.eval()
    constraint_mask = torch.zeros_like(attention_mask)
    constraint_mask[:, 0] = 1
    last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
    constraint_mask[torch.arange(target.shape[0], device=device), last_idx] = 1
    with torch.no_grad():
        samples = model.sample(
            prediction_length=20,
            num_samples=2,
            constraint_mask=constraint_mask,
            constraint_values=target,
            attention_mask=attention_mask,
            device=device
        )
    
    print(f"Samples shape: {samples.shape}")
    
    # Test denormalization
    print("\n5. Testing denormalization...")
    from data_preprocessing import denormalize_features
    
    test_normalized = np.random.randn(5, 20, 4)
    test_denorm = denormalize_features(test_normalized, norm_stats)
    print(f"Denormalized shape: {test_denorm.shape}")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
