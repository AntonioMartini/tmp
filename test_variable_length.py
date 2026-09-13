"""
Test script for variable-length path prediction with padding and attention masks.
Uses synthetic data to verify the implementation.
"""

import torch
import numpy as np
from pathlib import Path

from data_preprocessing import PathDataset, denormalize_features
from diffusion_model import PathDiffusionModel


def make_synthetic_sequences(num_sequences=20, min_len=100, max_len=200):
    """Create synthetic trajectory sequences for testing."""
    sequences = []
    for _ in range(num_sequences):
        length = np.random.randint(min_len, max_len)
        # Random walk trajectory
        pos = np.cumsum(np.random.randn(length, 2) * 0.1, axis=0)
        fwd = np.random.randn(length, 2)
        fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-8)
        seq = np.concatenate([pos, fwd], axis=-1)  # (T, 4)
        sequences.append(seq)
    return np.array(sequences, dtype=object)


def test_variable_length_dataset():
    """Test PathDataset with variable-length mode."""
    print("=" * 60)
    print("TEST 1: Variable-length dataset")
    print("=" * 60)

    sequences = make_synthetic_sequences()
    max_pred = 40
    min_pred = 10

    dataset = PathDataset(
        sequences,
        min_prediction_length=min_pred,
        max_prediction_length=max_pred
    )

    print(f"Dataset size: {len(dataset)}")

    lengths_seen = set()
    for i in range(min(20, len(dataset))):
        item = dataset[i]
        target = item['target']
        mask = item['attention_mask']
        actual_len = item['actual_length']

        assert target.shape == (max_pred, 4), f"Bad target shape: {target.shape}"
        assert mask.shape == (max_pred,), f"Bad mask shape: {mask.shape}"
        assert min_pred <= actual_len <= max_pred, f"Bad actual length: {actual_len}"
        assert mask[:actual_len].sum().item() == actual_len, "Mask doesn't match actual length"
        assert mask[actual_len:].sum().item() == 0, "Mask has non-zero in padding region"

        lengths_seen.add(actual_len)

    print(f"Observed actual lengths: {sorted(lengths_seen)}")
    print("PASSED: Variable-length dataset produces correct shapes and masks\n")


def test_fixed_length_dataset():
    """Test PathDataset with min == max (fixed-length as special case)."""
    print("=" * 60)
    print("TEST 2: Fixed-length dataset (min == max)")
    print("=" * 60)

    sequences = make_synthetic_sequences()
    pred_length = 30

    dataset = PathDataset(
        sequences,
        min_prediction_length=pred_length,
        max_prediction_length=pred_length
    )

    print(f"Dataset size: {len(dataset)}")

    item = dataset[0]
    assert item['target'].shape == (pred_length, 4)
    assert item['attention_mask'].shape == (pred_length,)
    assert item['attention_mask'].sum().item() == pred_length
    assert item['actual_length'] == pred_length

    print("PASSED: Fixed-length dataset works correctly\n")


def test_forward_with_attention_mask():
    """Test model forward pass with attention masks."""
    print("=" * 60)
    print("TEST 3: Forward pass with attention mask")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    max_pred = 40
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=100,
        device=device
    ).to(device)
    model.eval()

    B = 4
    target = torch.randn(B, max_pred, 4).to(device)
    timesteps = torch.randint(0, 50, (B,), device=device)

    # Create attention masks with different valid lengths
    actual_lengths = [15, 25, 40, 30]
    attention_mask = torch.zeros(B, max_pred)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = 1
    attention_mask = attention_mask.to(device)

    # Forward pass WITH attention mask
    noise_pred_masked = model(target, timesteps, attention_mask=attention_mask)
    print(f"Masked forward pass output shape: {noise_pred_masked.shape}")
    assert noise_pred_masked.shape == (B, max_pred, 4)

    # Forward pass WITHOUT attention mask
    noise_pred_unmasked = model(target, timesteps)
    print(f"Unmasked forward pass output shape: {noise_pred_unmasked.shape}")
    assert noise_pred_unmasked.shape == (B, max_pred, 4)

    # The outputs should differ (mask affects attention)
    diff = (noise_pred_masked - noise_pred_unmasked).abs().mean().item()
    print(f"Mean abs diff between masked/unmasked: {diff:.6f}")
    assert diff > 0, "Attention mask had no effect on output"

    print("PASSED: Forward pass handles attention masks correctly\n")


def test_masked_loss():
    """Test masked MSE loss computation."""
    print("=" * 60)
    print("TEST 4: Masked loss computation")
    print("=" * 60)

    B, T, F = 4, 40, 4
    noise_pred = torch.randn(B, T, F)
    noise = torch.randn(B, T, F)

    # Create attention mask with different valid lengths
    actual_lengths = [15, 25, 40, 30]
    attention_mask = torch.zeros(B, T)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = 1

    # Masked loss
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(noise_pred)
    masked_loss = torch.nn.functional.mse_loss(
        noise_pred * mask_expanded, noise * mask_expanded, reduction='sum'
    ) / mask_expanded.sum()

    # Unmasked loss
    unmasked_loss = torch.nn.functional.mse_loss(noise_pred, noise)

    # Manual masked loss for verification
    manual_total = 0.0
    manual_count = 0
    for i in range(B):
        length = actual_lengths[i]
        manual_total += ((noise_pred[i, :length] - noise[i, :length]) ** 2).sum().item()
        manual_count += length * F
    manual_loss = manual_total / manual_count

    print(f"Masked loss: {masked_loss.item():.6f}")
    print(f"Manual masked loss: {manual_loss:.6f}")
    print(f"Unmasked loss: {unmasked_loss.item():.6f}")

    assert abs(masked_loss.item() - manual_loss) < 1e-5, \
        f"Masked loss {masked_loss.item():.6f} != manual {manual_loss:.6f}"

    print("PASSED: Masked loss computed correctly\n")


def test_sampling_with_mask():
    """Test generation with attention mask."""
    print("=" * 60)
    print("TEST 5: Sampling with attention mask")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_pred = 40
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=20,  # Small for fast testing
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=100,
        device=device
    ).to(device)
    model.eval()

    B = 2
    num_samples = 3

    # Sample with attention mask
    attention_mask = torch.ones(B, max_pred).to(device)
    attention_mask[0, :20] = 1  # First sample: only first 20 valid
    attention_mask[0, 20:] = 0
    # Second sample: all valid

    # First and last valid frame are always constrained
    constraint_mask = torch.zeros(B, max_pred).to(device)
    constraint_mask[:, 0] = 1
    last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
    constraint_mask[torch.arange(B, device=device), last_idx] = 1
    constraint_values = torch.randn(B, max_pred, 4).to(device)

    with torch.no_grad():
        samples = model.sample(
            prediction_length=max_pred,
            num_samples=num_samples,
            constraint_mask=constraint_mask,
            constraint_values=constraint_values,
            attention_mask=attention_mask,
            device=device
        )

    print(f"Samples shape: {samples.shape}")
    assert samples.shape == (B, num_samples, max_pred, 4)

    # Sampling without constraints is rejected (endpoints are always required)
    try:
        model.sample(prediction_length=max_pred, num_samples=num_samples, device=device)
        raise AssertionError("sample() should require constraints")
    except ValueError:
        pass

    print("PASSED: Sampling with attention mask works\n")


def test_variable_length_sampling():
    """Test sampling different prediction lengths."""
    print("=" * 60)
    print("TEST 6: Variable-length sampling (different prediction lengths)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_pred = 60
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=20,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=100,
        device=device
    ).to(device)
    model.eval()

    # Sample at different prediction lengths (first/last always constrained)
    for pred_len in [15, 30, 60]:
        constraint_mask = torch.zeros(1, pred_len).to(device)
        constraint_mask[0, 0] = 1
        constraint_mask[0, -1] = 1
        constraint_values = torch.randn(1, pred_len, 4).to(device)
        with torch.no_grad():
            samples = model.sample(
                prediction_length=pred_len,
                num_samples=2,
                constraint_mask=constraint_mask,
                constraint_values=constraint_values,
                device=device
            )
        assert samples.shape == (1, 2, pred_len, 4), \
            f"Wrong shape for pred_len={pred_len}: {samples.shape}"
        print(f"  prediction_length={pred_len}: shape {samples.shape} OK")

    print("PASSED: Variable-length sampling works\n")


def test_end_to_end_variable():
    """Test end-to-end: dataset -> model forward -> masked loss."""
    print("=" * 60)
    print("TEST 7: End-to-end variable-length pipeline")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sequences = make_synthetic_sequences(num_sequences=30)
    max_pred = 40

    dataset = PathDataset(
        sequences,
        min_prediction_length=15,
        max_prediction_length=max_pred
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=max_pred,
        device=device
    ).to(device)

    batch = next(iter(loader))
    target = batch['target'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    actual_lengths = batch['actual_length']

    print(f"Batch target: {target.shape}")
    print(f"Batch mask: {attention_mask.shape}")
    print(f"Actual lengths: {actual_lengths.tolist()}")

    # Forward pass
    timesteps = torch.randint(0, 50, (target.shape[0],), device=device)
    noise_pred = model(target, timesteps, attention_mask=attention_mask)

    # Masked loss
    noise = torch.randn_like(target)
    mask_expanded = attention_mask.unsqueeze(-1).expand_as(noise_pred)
    loss = torch.nn.functional.mse_loss(
        noise_pred * mask_expanded, noise * mask_expanded, reduction='sum'
    ) / mask_expanded.sum()

    print(f"Masked loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("PASSED: End-to-end variable-length pipeline works\n")


if __name__ == '__main__':
    tests = [
        test_variable_length_dataset,
        test_fixed_length_dataset,
        test_forward_with_attention_mask,
        test_masked_loss,
        test_sampling_with_mask,
        test_variable_length_sampling,
        test_end_to_end_variable,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
