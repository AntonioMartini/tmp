"""
Quick training test to verify the training loop works.
"""

import torch
from pathlib import Path
from train import DiffusionTrainer
from data_preprocessing import prepare_datasets
from diffusion_model import PathDiffusionModel

def test_training():
    """Test training for a few iterations."""
    print("Testing training loop...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Prepare small dataset
    print("\n1. Preparing data...")
    train_loader, val_loader, test_loader, norm_stats = prepare_datasets(
        root_features_path=Path('./data/root_feature_matrix.npy'),
        clip_ranges_path=Path('../motion_2/data/clip_frame_ranges_world.pkl'),
        min_prediction_length=15,
        max_prediction_length=20,
        batch_size=8
    )
    
    # Create small model
    print("\n2. Creating model...")
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        pe_dropout=0.1,
        norm_first=False,
        device=device
    ).to(device)
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        save_dir='./test_checkpoints'
    )
    
    # Test training for 1 epoch
    print("\n4. Training for 1 epoch...")
    train_loss = trainer.train_epoch()
    print(f"Train loss: {train_loss:.6f}")
    
    # Test validation
    print("\n5. Running validation...")
    val_loss = trainer.validate()
    print(f"Val loss: {val_loss:.6f}")
    
    print("\n✅ Training test passed!")
    return True

if __name__ == '__main__':
    try:
        test_training()
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
