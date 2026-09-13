"""
Data preprocessing pipeline for path prediction diffusion model.
Extracts training sequences from root features and clip ranges.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class PathDataset(Dataset):
    """Dataset of variable-length path sequences for an unconditional model.

    Each sample draws a random length in [min_prediction_length,
    max_prediction_length], clamped to the frames available in the sequence.
    Targets are zero-padded to max_prediction_length and an attention_mask
    marks valid positions (1) vs padding (0).
    Set min == max for fixed-length behavior.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        min_prediction_length: int = 20,
        max_prediction_length: int = 600
    ):
        """
        Args:
            sequences: Array of shape (N, T, 4) containing path sequences
            min_prediction_length: Minimum sequence length per sample
            max_prediction_length: Maximum sequence length (padded tensor size)
        """
        self.sequences = sequences
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length

        self.valid_indices = [
            i for i, seq in enumerate(sequences)
            if len(seq) >= 1
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        seq_idx = self.valid_indices[idx]
        full_seq = self.sequences[seq_idx]

        actual_pred_length = np.random.randint(
            self.min_prediction_length,
            self.max_prediction_length + 1
        )
        # Clamp to what the sequence can actually provide
        actual_pred_length = min(actual_pred_length, len(full_seq))

        # Sample window
        if len(full_seq) > actual_pred_length:
            start_idx = np.random.randint(0, len(full_seq) - actual_pred_length + 1)
            window = full_seq[start_idx:start_idx + actual_pred_length]
        else:
            window = full_seq[:actual_pred_length]

        # Pad target to max_prediction_length with zeros
        target_padded = np.zeros((self.max_prediction_length, 4))
        target_padded[:actual_pred_length] = window

        # Attention mask: 1 for real data, 0 for padding
        attention_mask = np.zeros(self.max_prediction_length)
        attention_mask[:actual_pred_length] = 1

        return {
            'target': torch.FloatTensor(target_padded),
            'attention_mask': torch.FloatTensor(attention_mask),
            'actual_length': actual_pred_length
        }


def extract_sequences_from_clips(
    root_features: np.ndarray,
    clip_ranges: dict,
    min_sequence_length: int = 100,
    max_sequence_length: int = 500
) -> np.ndarray:
    """
    Extract sequences from clips for training.
    
    Args:
        root_features: Array of shape (N, 4) containing root features
        clip_ranges: Dictionary mapping clip names to (start, end) frame ranges
        min_sequence_length: Minimum length of a sequence to include
        max_sequence_length: Maximum length - longer clips are split
        
    Returns:
        Array of shape (M, T, 4) containing extracted sequences
    """
    sequences = []
    
    for clip_name, (start, end) in clip_ranges.items():
        clip_features = root_features[start:end]
        clip_length = len(clip_features)
        
        if clip_length < min_sequence_length:
            continue
            
        # Split long clips into smaller sequences
        if clip_length > max_sequence_length:
            num_splits = int(np.ceil(clip_length / max_sequence_length))
            split_length = clip_length // num_splits
            
            for i in range(num_splits):
                split_start = i * split_length
                split_end = min((i + 1) * split_length, clip_length)
                split_features = clip_features[split_start:split_end]
                
                if len(split_features) >= min_sequence_length:
                    sequences.append(split_features)
        else:
            sequences.append(clip_features)
    
    return np.array(sequences, dtype=object)


def normalize_features(sequences: np.ndarray, method: 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize path features.
    
    Args:
        sequences: Array of shape (M, T, 4) containing sequences
        method: Normalization method ('standard' or 'minmax')
        
    Returns:
        Normalized sequences and normalization statistics
    """
    # First convert to regular numpy array if it's object dtype
    if sequences.dtype == object:
        max_len = max(len(seq) for seq in sequences)
        padded = np.zeros((len(sequences), max_len, 4))
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        sequences = padded
    
    # Flatten to compute statistics
    all_features = sequences.reshape(-1, 4)
    
    if method == 'standard':
        mean = all_features.mean(axis=0)
        std = all_features.std(axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        
        normalized = (sequences - mean) / std
        stats = {'mean': mean, 'std': std, 'method': 'standard'}
        
    elif method == 'minmax':
        min_val = all_features.min(axis=0)
        max_val = all_features.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        normalized = (sequences - min_val) / range_val
        stats = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    return normalized, stats


def denormalize_features(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Denormalize features using the provided statistics."""
    if stats['method'] == 'standard':
        return normalized * stats['std'] + stats['mean']
    elif stats['method'] == 'minmax':
        return normalized * (stats['max'] - stats['min']) + stats['min']
    else:
        raise ValueError(f"Unknown normalization method: {stats['method']}")


def prepare_datasets(
    root_features_path: Path,
    clip_ranges_path: Path,
    min_prediction_length: int = 20,
    max_prediction_length: int = 600,
    batch_size: int = 32,
    test_split: float = 0.2,
    val_split: float = 0.1,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Prepare train, validation, and test dataloaders.

    Args:
        root_features_path: Path to root_feature_matrix.npy
        clip_ranges_path: Path to clip_frame_ranges_world.pkl
        min_prediction_length: Minimum prediction length
        max_prediction_length: Maximum prediction length (padded tensor size)
        batch_size: Batch size for dataloaders
        test_split: Fraction of data for testing
        val_split: Fraction of training data for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader, normalization_stats
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Load data
    print("Loading data...")
    root_features = np.load(root_features_path)
    print(f"Root features shape: {root_features.shape}")
    
    with open(clip_ranges_path, 'rb') as f:
        clip_ranges = pickle.load(f)
    print(f"Number of clips: {len(clip_ranges)}")
    
    # Extract sequences
    print("Extracting sequences from clips...")
    sequences = extract_sequences_from_clips(
        root_features,
        clip_ranges,
        min_sequence_length=min_prediction_length,
        max_sequence_length=700
    )
    print(f"Extracted {len(sequences)} sequences")
    
    # Normalize
    print("Normalizing features...")
    sequences, norm_stats = normalize_features(sequences, method='standard')
    print(f"Normalized sequences shape: {sequences.shape}")
    
    # Train/val/test split
    print("Splitting data...")
    train_val_sequences, test_sequences = train_test_split(
        sequences, test_size=test_split, random_state=random_seed
    )
    train_sequences, val_sequences = train_test_split(
        train_val_sequences, test_size=val_split, random_state=random_seed
    )
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    # Create datasets
    print(f"Prediction lengths: {min_prediction_length}-{max_prediction_length}")
    dataset_args = dict(
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length
    )
    train_dataset = PathDataset(train_sequences, **dataset_args)
    val_dataset = PathDataset(val_sequences, **dataset_args)
    test_dataset = PathDataset(test_sequences, **dataset_args)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, norm_stats


if __name__ == '__main__':
    # Test the preprocessing pipeline
    data_dir = Path('/workspace/models/path_prediction/data')
    motion_data_dir = Path('/workspace/models/motion_2/data')
    
    train_loader, val_loader, test_loader, norm_stats = prepare_datasets(
        root_features_path=data_dir / 'root_feature_matrix.npy',
        clip_ranges_path=motion_data_dir / 'clip_frame_ranges_world.pkl',
        min_prediction_length=20,
        max_prediction_length=60,
        batch_size=16
    )

    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"\nBatch target shape: {batch['target'].shape}")
    
    # Save normalization stats
    np.save(data_dir / 'normalization_stats.npy', norm_stats)
    print(f"\nSaved normalization stats to {data_dir / 'normalization_stats.npy'}")
