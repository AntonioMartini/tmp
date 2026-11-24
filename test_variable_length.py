"""
Test script for variable-length path prediction with padding and attention masks.
Uses synthetic data to verify the implementation.
"""

import torch
import numpy as np
from pathlib import Path

from data_preprocessing import PathDataset, denormalize_features
from diffusion_model import PathDiffusionModel, generate_constraint_mask


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
    max_pred = 181
    min_pred = 61

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
    pred_length = 61

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

    max_pred = 181
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=181,
        device=device
    ).to(device)
    model.eval()

    B = 4
    target = torch.randn(B, max_pred, 4).to(device)
    timesteps = torch.randint(0, 50, (B,), device=device)

    # Create attention masks with different valid lengths
    actual_lengths = [61, 121, 181, 150]
    attention_mask = torch.zeros(B, max_pred)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = 1
    attention_mask = attention_mask.to(device)

    # Forward pass WITH attention mask
    clean_pred_masked = model.denoiser(target, timesteps, attention_mask=attention_mask)
    print(f"Masked forward pass output shape: {clean_pred_masked.shape}")
    assert clean_pred_masked.shape == (B, max_pred, 4)

    # Forward pass WITHOUT attention mask
    clean_pred_unmasked = model.denoiser(target, timesteps)
    print(f"Unmasked forward pass output shape: {clean_pred_unmasked.shape}")
    assert clean_pred_unmasked.shape == (B, max_pred, 4)

    # The outputs should differ (mask affects attention)
    diff = (clean_pred_masked - clean_pred_unmasked).abs().mean().item()
    print(f"Mean abs diff between masked/unmasked: {diff:.6f}")
    assert diff > 0, "Attention mask had no effect on output"

    print("PASSED: Forward pass handles attention masks correctly\n")


def test_masked_loss():
    """Test masked L1 clean-trajectory loss computation."""
    print("=" * 60)
    print("TEST 4: Masked loss computation")
    print("=" * 60)

    B, T, F = 4, 181, 4
    clean_pred = torch.randn(B, T, F)
    target = torch.randn(B, T, F)

    # Create attention mask with different valid lengths
    actual_lengths = [61, 121, 181, 150]
    attention_mask = torch.zeros(B, T)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = 1

    # Masked loss
    valid = attention_mask.bool()
    masked_loss = torch.nn.functional.l1_loss(clean_pred[valid], target[valid])

    # Unmasked loss
    unmasked_loss = torch.nn.functional.l1_loss(clean_pred, target)

    # Manual masked loss for verification
    manual_total = 0.0
    manual_count = 0
    for i in range(B):
        length = actual_lengths[i]
        manual_total += (clean_pred[i, :length] - target[i, :length]).abs().sum().item()
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

    max_pred = 181
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=20,  # Small for fast testing
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=181,
        device=device
    ).to(device)
    model.eval()

    B = 2
    num_samples = 3

    # Sample with attention mask
    attention_mask = torch.ones(B, max_pred).to(device)
    attention_mask[0, :121] = 1  # First sample: only first 121 valid
    attention_mask[0, 121:] = 0
    # Second sample: all valid

    # First and last valid frame are always constrained
    constraint_mask = generate_constraint_mask(attention_mask)
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

    max_pred = 181
    model = PathDiffusionModel(
        feature_dim=4,
        num_timesteps=20,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        pe_dropout=0.0,
        max_seq_len=181,
        device=device
    ).to(device)
    model.eval()

    # Sample at different prediction lengths (first/last always constrained)
    for pred_len in [61, 120, 121, 181]:
        constraint_mask = generate_constraint_mask(torch.ones(1, pred_len, device=device))
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
    max_pred = 181

    dataset = PathDataset(
        sequences,
        min_prediction_length=61,
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
    clean_pred = model(target, timesteps, attention_mask=attention_mask)

    # Masked loss
    valid = attention_mask.bool()
    loss = torch.nn.functional.l1_loss(clean_pred[valid], target[valid])

    print(f"Masked loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print("PASSED: End-to-end variable-length pipeline works\n")


def test_random_constraint_masks():
    from diffusion_model import generate_constraint_mask, validate_constraint_mask

    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        lengths = torch.arange(61, 601, device=device)
        attention = (torch.arange(600, device=device)[None] < lengths[:, None]).float()
        torch.manual_seed(42)
        mask = generate_constraint_mask(attention)
        validate_constraint_mask(mask, attention)
        assert mask.device == attention.device
        assert not (mask.bool() & ~attention.bool()).any()
        assert (mask[:, 0] == 1).all()
        assert (mask[torch.arange(len(lengths), device=device), lengths - 1] == 1).all()
        counts = []
        for row, length in zip(mask, lengths.tolist()):
            indices = row.nonzero().flatten()
            assert (indices.diff() >= 60).all()
            assert len(indices) <= (length - 1) // 60 + 1
            if length <= 120:
                assert indices.tolist() == [0, length - 1]
            counts.append(len(indices))
        assert max(counts) > 2
        torch.manual_seed(42)
        assert torch.equal(mask, generate_constraint_mask(attention))
        assert not torch.equal(mask, generate_constraint_mask(attention))
        boundary = generate_constraint_mask(torch.ones(128, 121, device=device))
        assert boundary[:, 60].any()
        assert set(boundary.nonzero()[:, 1].tolist()) == {0, 60, 120}

    for length in [0, 1, 60]:
        with np.testing.assert_raises(ValueError):
            generate_constraint_mask(torch.ones(1, length))
    invalid_attention = torch.ones(1, 121)
    invalid_attention[0, 10] = 0
    with np.testing.assert_raises(ValueError):
        generate_constraint_mask(invalid_attention)
    for indices in [[0, 59, 180], [0, 60, 119, 180], [0, 121, 180], [60, 180], [0, 60]]:
        mask = torch.zeros(1, 181)
        mask[0, indices] = 1
        with np.testing.assert_raises(ValueError):
            validate_constraint_mask(mask, torch.ones_like(mask))


def test_training_random_constraints():
    from unittest.mock import patch
    from train import DiffusionTrainer
    from diffusion_model import validate_constraint_mask

    torch.manual_seed(12)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=241, device='cpu')
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.model = model
    trainer.device = 'cpu'
    lengths = torch.tensor([61, 120, 121, 181, 241] * 4)
    attention = (torch.arange(241)[None] < lengths[:, None]).float()
    batch = {'target': torch.randn(20, 241, 4), 'attention_mask': attention}
    masks = []
    for training in [True, False]:
        model.train(training)
        projected_inputs = []
        outputs = []
        input_hook = model.denoiser.input_linear.register_forward_pre_hook(
            lambda module, args: projected_inputs.append(args[0].detach().clone())
        )
        output_hook = model.denoiser.register_forward_hook(
            lambda module, args, output: outputs.append(output.detach().clone())
        )
        with patch.object(model.diffusion, 'q_sample', wraps=model.diffusion.q_sample) as q_sample:
            loss = trainer.compute_loss(batch)
        input_hook.remove()
        output_hook.remove()
        assert torch.isfinite(loss)
        loss.backward()
        target, timesteps, noise = q_sample.call_args.args
        inputs = projected_inputs[0]
        assert inputs.shape == (20, 241, 8)
        feature_mask = inputs[..., 4:]
        mask = feature_mask[..., 0]
        torch.testing.assert_close(feature_mask, mask[..., None].expand_as(feature_mask))
        validate_constraint_mask(mask, attention)
        assert (mask.sum(dim=1) > 2).any()
        constrained = mask.bool()
        torch.testing.assert_close(inputs[..., :4][constrained], target[constrained])
        noisy = model.diffusion.q_sample(target, timesteps, noise)
        free = attention.bool() & ~constrained
        torch.testing.assert_close(inputs[..., :4][free], noisy[free])
        assert torch.count_nonzero(noise[constrained]) > 0
        expected_loss = (outputs[0] - target).abs()[attention.bool()].mean()
        torch.testing.assert_close(loss.detach(), expected_loss)
        assert model.denoiser.input_linear.weight.grad[:, 4:].abs().sum() > 0
        masks.append(mask)
    assert not torch.equal(*masks)


def test_sampling_constraint_signal():
    from unittest.mock import patch

    torch.manual_seed(42)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=181, device='cpu').eval()
    attention = torch.ones(2, 181)
    attention[0, 121:] = 0
    mask = torch.zeros_like(attention)
    mask[0, [0, 60, 120]] = 1
    mask[1, [0, 60, 120, 180]] = 1
    values = torch.randn(2, 181, 4)
    values[:, -1, :2] = 12
    projected_inputs = []
    hook = model.denoiser.input_linear.register_forward_pre_hook(
        lambda module, args: projected_inputs.append(args[0].detach().clone())
    )
    with patch.object(model.diffusion, 'q_sample', wraps=model.diffusion.q_sample) as q_sample:
        samples = model.sample(181, num_samples=2, constraint_mask=mask.bool(),
                               constraint_values=values, attention_mask=attention, device='cpu')
    hook.remove()
    assert q_sample.call_count == 0
    expected = values.clone()
    expected[..., :2] -= values[:, :1, :2]
    expected = expected.repeat_interleave(2, dim=0)
    selected = mask.repeat_interleave(2, dim=0).bool()
    assert len(projected_inputs) == 4
    for inputs in projected_inputs:
        assert inputs.shape == (4, 181, 8)
        torch.testing.assert_close(inputs[..., :4][selected], expected[selected])
        torch.testing.assert_close(inputs[..., 4:], selected[..., None].expand_as(expected).float())
    for row in range(2):
        for sample in samples[row]:
            torch.testing.assert_close(sample[mask[row].bool()], values[row, mask[row].bool()])
    bad_mask = mask.clone()
    bad_mask[1, 59] = 1
    with np.testing.assert_raises(ValueError):
        model.sample(181, constraint_mask=bad_mask, constraint_values=values,
                     attention_mask=attention, device='cpu')


def test_respaced_ddim_sampling():
    torch.manual_seed(42)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=121, device='cpu').eval()
    mask = torch.zeros(1, 121)
    mask[:, [0, 60, 120]] = 1
    values = torch.randn(1, 121, 4)
    projected_inputs = []
    hook = model.denoiser.input_linear.register_forward_pre_hook(
        lambda module, args: projected_inputs.append(args[0].detach().clone())
    )
    samples = model.sample(121, constraint_mask=mask, constraint_values=values,
                           sampling_steps=2, device='cpu')
    hook.remove()
    assert len(projected_inputs) == 2
    torch.testing.assert_close(samples[0, 0, mask[0].bool()], values[0, mask[0].bool()])
    with np.testing.assert_raises(ValueError):
        model.sample(121, constraint_mask=mask, constraint_values=values,
                     sampling_steps=5, device='cpu')


def test_inference_waypoint_constraints():
    from unittest.mock import patch
    from inference import PathPredictor, build_constraints
    from diffusion_model import validate_constraint_mask

    torch.manual_seed(7)
    predictor = PathPredictor.__new__(PathPredictor)
    predictor.device = 'cpu'
    predictor.norm_stats = {'method': 'standard', 'mean': np.array([20., -30., .1, -.2]),
                            'std': np.array([90., 120., .6, .8])}
    predictor.model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                                         num_timesteps=4, max_seq_len=241, device='cpu').eval()
    anchor = np.array([10., -20., 1., 0.])
    reference = np.tile([40., 25., 0., 1.], (241, 1))
    reference[[60, 120, 240]] = [[50., 30., 0., 1.], [75., 45., 1., 0.], [110., 60., -1., 0.]]
    reference_before = reference.copy()
    mask, values = build_constraints(anchor, 241, constraint_values=reference)
    assert mask[0] == mask[-1] == 1
    interior = mask.astype(bool)
    interior[[0, -1]] = False
    np.testing.assert_array_equal(values[interior], reference[interior])
    np.testing.assert_array_equal(reference, reference_before)
    assert mask.sum() > 2
    explicit = np.zeros(241)
    explicit[[0, 60, 180]] = 1
    mask, values = build_constraints(anchor, 241, explicit, reference)
    with patch.object(predictor.model, 'sample', wraps=predictor.model.sample) as sample:
        predictions = predictor.predict(anchor[None], 241, num_samples=2, sampling_steps=2,
                                        constraint_mask=explicit, constraint_values=reference)
    assert sample.call_args.kwargs['sampling_steps'] == 2
    actual_mask = sample.call_args.kwargs['constraint_mask']
    assert actual_mask[0].nonzero().flatten().tolist() == [0, 60, 180, 240]
    for prediction in predictions:
        np.testing.assert_allclose(prediction[mask.astype(bool)], values[mask.astype(bool)], atol=2e-5)
    assert predictions.shape == (2, 241, 4)
    assert np.isfinite(predictions).all()

    contexts = np.tile(anchor, (3, 1, 1))
    references = np.tile(reference, (3, 1, 1))
    with patch.object(predictor.model, 'sample', wraps=predictor.model.sample) as sample:
        predictions = predictor.predict_batch(contexts, [61, 121, 241], references,
                                              num_samples=2, max_prediction_length=241)
    args = sample.call_args.kwargs
    validate_constraint_mask(args['constraint_mask'], args['attention_mask'])
    raw_values = denormalize_features(args['constraint_values'].numpy(), predictor.norm_stats)
    for row, length in enumerate([61, 121, 241]):
        assert predictions[row].shape == (2, length, 4)
        selected = args['constraint_mask'][row, :length].bool().numpy()
        for prediction in predictions[row]:
            np.testing.assert_allclose(prediction[selected], raw_values[row, :length][selected], atol=2e-5)
            np.testing.assert_allclose(prediction[0], anchor, atol=2e-5)
            np.testing.assert_allclose(prediction[-1], reference[length - 1], atol=2e-5)

    for invalid in [np.zeros(60), np.full(241, .5), np.eye(1, 241, 59)[0]]:
        with np.testing.assert_raises(ValueError):
            predictor.predict(anchor[None], 241, constraint_mask=invalid, constraint_values=reference)
    with np.testing.assert_raises(ValueError):
        predictor.predict(anchor[None], 60, constraint_values=reference[:60])
    with np.testing.assert_raises(ValueError):
        predictor.predict(anchor[None], 241, max_prediction_length=120, constraint_values=reference)
    with np.testing.assert_raises(ValueError):
        predictor.predict(anchor[None], 241)
    with np.testing.assert_raises(ValueError):
        predictor.predict(anchor[None], 241, constraint_mask=explicit)
    with np.testing.assert_raises(ValueError):
        predictor.predict_batch(contexts, [61, 60, 241], references)
    with np.testing.assert_raises(ValueError):
        predictor.predict_batch(contexts, [61, 121, 241], None)


def test_random_sampling_and_evaluation():
    from unittest.mock import patch
    from evaluate import PathEvaluator
    from diffusion_model import validate_constraint_mask

    torch.manual_seed(7)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=241, device='cpu').eval()
    attention = torch.ones(4, 241)
    values = torch.randn(4, 241, 4)
    with patch.object(model.diffusion, 'p_sample_loop', wraps=model.diffusion.p_sample_loop) as loop:
        samples = model.sample(241, constraint_values=values, device='cpu')
    mask = loop.call_args.kwargs['constraint_mask']
    validate_constraint_mask(mask, attention)
    assert (mask.sum(dim=1) > 2).any()
    for row in range(4):
        selected = mask[row].bool()
        torch.testing.assert_close(samples[row, 0, selected], values[row, selected])
    bad = mask.clone()
    bad[0, 180:] = 0
    bad[0, 120] = 1
    bad[0, 240] = 1
    padded_attention = attention.clone()
    padded_attention[0, 121:] = 0
    with np.testing.assert_raises(ValueError):
        model.sample(241, constraint_mask=bad, constraint_values=values,
                     attention_mask=padded_attention, device='cpu')

    stats = {'method': 'standard', 'mean': np.array([20., -30., .1, -.2]),
             'std': np.array([90., 120., .6, .8])}
    evaluator = PathEvaluator(model, stats, device='cpu')
    with patch.object(model, 'sample', wraps=model.sample) as sample:
        metrics = evaluator.evaluate_batch(values, attention, torch.full((4,), 241), num_samples=2)
    torch.testing.assert_close(sample.call_args.kwargs['constraint_values'], values)
    validate_constraint_mask(sample.call_args.kwargs['constraint_mask'], attention)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics['fde'] < 1e-3


def test_kimodo_denoiser_inputs():
    torch.manual_seed(12)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2, dropout=0,
                               pe_dropout=0, num_timesteps=4, max_seq_len=121, device='cpu').eval()
    noisy = torch.randn(1, 121, 4)
    values = torch.randn_like(noisy)
    mask = torch.zeros(1, 121, dtype=torch.bool)
    mask[:, [0, 60, 120]] = True
    values[:, 60] = 0
    projected = []
    hook = model.denoiser.input_linear.register_forward_pre_hook(
        lambda module, args: projected.append(args[0].detach().clone())
    )
    for timestep in [0, 3]:
        model.denoiser(noisy, torch.tensor([timestep]), constraint_mask=mask, constraint_values=values)
    for inputs in projected:
        assert inputs.shape[-1] == 8
        torch.testing.assert_close(inputs[..., :4][mask], values[mask])
        torch.testing.assert_close(inputs[..., :4][~mask], noisy[~mask])
        torch.testing.assert_close(inputs[..., 4:], mask[..., None].expand_as(values).float())
        assert (inputs[:, 60, 4:] == 1).all()
    changed = values.clone()
    changed[~mask] = 1e6
    original = model.denoiser(noisy, torch.tensor([3]), constraint_mask=mask, constraint_values=values)
    modified = model.denoiser(noisy, torch.tensor([3]), constraint_mask=mask, constraint_values=changed)
    torch.testing.assert_close(original, modified)
    model.denoiser(noisy, torch.tensor([3]))
    assert not projected[-1][..., 4:].any()
    hook.remove()
    with np.testing.assert_raises(ValueError):
        model.denoiser(noisy, torch.tensor([3]), constraint_mask=mask)


def test_clean_prediction_posterior():
    from unittest.mock import patch

    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=121, device='cpu').eval()
    noisy = torch.randn(2, 121, 4)
    clean_prediction = torch.full_like(noisy, .75)
    timesteps = torch.tensor([0, 3])
    expected = model.diffusion.q_posterior_mean_variance(clean_prediction, noisy, timesteps)
    with patch.object(model.denoiser, 'forward', return_value=clean_prediction):
        actual = model.diffusion.p_mean_variance(model, noisy, timesteps, clip_denoised=False)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_clip_window_clamped_to_max_seq_len():
    """A clip window clamped to the model limit yields a valid prediction."""
    from inference import PathPredictor

    torch.manual_seed(5)
    predictor = PathPredictor.__new__(PathPredictor)
    predictor.device = 'cpu'
    predictor.norm_stats = {'method': 'standard', 'mean': np.array([20., -30., .1, -.2]),
                            'std': np.array([90., 120., .6, .8])}
    predictor.model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                                         num_timesteps=4, max_seq_len=181, device='cpu').eval()

    clip = np.random.randn(1000, 4).astype(np.float32)
    context = clip[:60]
    ground_truth = clip[59:59 + predictor.model.max_seq_len]
    length = len(ground_truth)
    assert length == 181

    mask = np.zeros(length)
    mask[[0, 120, -1]] = 1
    predictions = predictor.predict(
        context, prediction_length=length, num_samples=2,
        max_prediction_length=length,
        constraint_mask=mask, constraint_values=ground_truth)
    assert predictions.shape == (2, length, 4)
    assert np.isfinite(predictions).all()
    for prediction in predictions:
        np.testing.assert_allclose(prediction[0], context[-1], atol=2e-5)
        np.testing.assert_allclose(prediction[-1], ground_truth[-1], atol=2e-5)
        np.testing.assert_allclose(prediction[120], ground_truth[120], atol=2e-5)


def test_inference_beyond_max_seq_len():
    """Inference is not clamped to the training max_seq_len; the sinusoidal
    positional encoding extends to cover longer sequences."""
    from inference import PathPredictor
    from diffusion_model import PositionalEncoding

    pe = PositionalEncoding(16, 0.0, max_len=10).eval()
    out = pe(torch.zeros(1, 30, 16))
    assert pe.pe.shape[1] >= 30
    assert out.shape == (1, 30, 16)

    torch.manual_seed(11)
    predictor = PathPredictor.__new__(PathPredictor)
    predictor.device = 'cpu'
    predictor.norm_stats = {'method': 'standard', 'mean': np.array([20., -30., .1, -.2]),
                            'std': np.array([90., 120., .6, .8])}
    predictor.model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                                         num_timesteps=4, max_seq_len=181, device='cpu').eval()
    anchor = np.array([10., -20., 1., 0.])
    goal = np.array([110., 60., -1., 0.])
    reference = np.tile([40., 25., 0., 1.], (241, 1))
    reference[-1] = goal
    mask = np.zeros(241)
    mask[[0, 120, -1]] = 1
    predictions = predictor.predict(anchor[None], 241, num_samples=2,
                                    constraint_mask=mask, constraint_values=reference)
    assert predictions.shape == (2, 241, 4)
    assert np.isfinite(predictions).all()
    for prediction in predictions:
        np.testing.assert_allclose(prediction[0], anchor, atol=2e-5)
        np.testing.assert_allclose(prediction[-1], goal, atol=2e-5)
        np.testing.assert_allclose(prediction[120], reference[120], atol=2e-5)

    contexts = np.tile(anchor, (2, 1, 1))
    references = np.tile(reference, (2, 1, 1))
    outs = predictor.predict_batch(contexts, [121, 241], references, num_samples=1)
    assert outs[0].shape == (1, 121, 4)
    assert outs[1].shape == (1, 241, 4)


def test_dataset_constraint_lengths():
    sequences = [np.zeros((length, 4)) for length in [20, 60, 61, 120]]
    dataset = PathDataset(sequences, min_prediction_length=61, max_prediction_length=180)
    assert dataset.valid_indices == [2, 3]
    assert dataset[0]['actual_length'] == 61
    with np.testing.assert_raises(ValueError):
        PathDataset(sequences, min_prediction_length=60, max_prediction_length=180)
    with np.testing.assert_raises(ValueError):
        PathDataset(sequences[:2], min_prediction_length=61, max_prediction_length=180)


def test_waypoint_solver_configuration():
    from waypoint_app import PREDICTION_LENGTH, WAYPOINT_COUNT, WaypointSolver

    solver = WaypointSolver.__new__(WaypointSolver)
    solver.prediction_length = PREDICTION_LENGTH
    solver.waypoint_indices = solver._waypoint_indices()

    assert PREDICTION_LENGTH == 60 * 6
    assert WAYPOINT_COUNT == 2 + 4
    assert solver.waypoint_indices.tolist() == [0, 72, 144, 215, 287, 359]
    assert solver.config() == {
        'waypoint_indices': [0, 72, 144, 215, 287, 359],
        'gaps': [72, 72, 71, 72, 72],
    }
    with np.testing.assert_raises(ValueError):
        solver.solve(np.zeros((5, 2)))


if __name__ == '__main__':
    tests = [
        test_kimodo_denoiser_inputs,
        test_clean_prediction_posterior,
        test_inference_waypoint_constraints,
        test_random_sampling_and_evaluation,
        test_dataset_constraint_lengths,
        test_waypoint_solver_configuration,
        test_clip_window_clamped_to_max_seq_len,
        test_inference_beyond_max_seq_len,
        test_random_constraint_masks,
        test_training_random_constraints,
        test_sampling_constraint_signal,
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
