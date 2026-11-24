"""
Quick training test to verify the training loop works.
"""

import torch
import numpy as np
from pathlib import Path
from train import DiffusionTrainer
from data_preprocessing import prepare_datasets
from diffusion_model import PathDiffusionModel, MODEL_FORMAT_VERSION

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
        min_prediction_length=120,
        max_prediction_length=180,
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

def _save_kimodo_test_checkpoint(save_dir):
    torch.manual_seed(42)
    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2, dropout=0,
                               pe_dropout=0, num_timesteps=4, max_seq_len=600, device='cpu')
    batch = {'target': torch.randn(2, 181, 4), 'attention_mask': torch.ones(2, 181)}
    trainer = DiffusionTrainer(model, [batch], [batch], device='cpu', save_dir=save_dir)
    try:
        assert np.isfinite(trainer.train_epoch())
        trainer.best_val_loss = trainer.validate()
        assert np.isfinite(trainer.best_val_loss)
        trainer.save_checkpoint()
    finally:
        trainer.writer.close()
    stats = {'method': 'standard', 'mean': np.array([20., -30., .1, -.2]),
             'std': np.array([90., 120., .6, .8])}
    np.save(Path(save_dir) / 'normalization_stats.npy', stats)
    return trainer


def test_kimodo_checkpoint_round_trip(tmp_path):
    from inference import PathPredictor, build_constraints

    trainer = _save_kimodo_test_checkpoint(tmp_path)
    checkpoint_path = tmp_path / 'best_model.pt'
    stats_path = tmp_path / 'normalization_stats.npy'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert checkpoint['format_version'] == MODEL_FORMAT_VERSION
    assert checkpoint['prediction_type'] == 'x0'
    assert checkpoint['constraint_conditioning'] == 'clean_imputation_mask'
    predictor = PathPredictor(str(checkpoint_path), str(stats_path), device='cpu')
    assert predictor.model.config == trainer.model.config
    assert predictor.model.denoiser.input_linear.in_features == 8
    assert predictor.model.diffusion.num_timesteps == 4
    for name, param in predictor.model.named_parameters():
        torch.testing.assert_close(param, checkpoint['model_state_dict'][name])
    anchor = np.array([10., -20., 1., 0.])
    values = np.tile([40., 25., 0., 1.], (181, 1))
    values[-1] = [110., 60., 0., 1.]
    mask, values = build_constraints(anchor, 181, constraint_values=values)
    predictions = predictor.predict(anchor[None], 181, num_samples=2, max_prediction_length=600,
                                    constraint_mask=mask, constraint_values=values)
    assert np.isfinite(predictions).all()
    for prediction in predictions:
        np.testing.assert_allclose(prediction[mask.astype(bool)], values[mask.astype(bool)], atol=2e-5)
    with torch.no_grad():
        for param in trainer.model.parameters():
            param.zero_()
    trainer.load_checkpoint(checkpoint_path)
    for name, param in trainer.model.named_parameters():
        torch.testing.assert_close(param, checkpoint['raw_model_state_dict'][name])
    assert trainer.global_step == checkpoint['global_step']
    assert trainer.optimizer.state_dict()['state']
    trainer.model.config['num_timesteps'] = 5
    with np.testing.assert_raises_regex(ValueError, 'configuration must match'):
        trainer.load_checkpoint(checkpoint_path)
    trainer.model.config['num_timesteps'] = 4
    with np.testing.assert_raises_regex(ValueError, 'max_seq_len must match'):
        PathPredictor(str(checkpoint_path), str(stats_path), device='cpu', max_seq_len=601)
    legacy_path = tmp_path / 'legacy.pt'
    torch.save({'model_state_dict': checkpoint['model_state_dict']}, legacy_path)
    with np.testing.assert_raises_regex(ValueError, 'must be retrained'):
        PathPredictor(str(legacy_path), str(stats_path), device='cpu')
    with np.testing.assert_raises_regex(ValueError, 'must be retrained'):
        trainer.load_checkpoint(legacy_path)
    incompatible = dict(checkpoint, prediction_type='epsilon')
    with np.testing.assert_raises_regex(ValueError, 'must be retrained'):
        PathDiffusionModel.from_checkpoint(incompatible)
    incomplete = dict(checkpoint, model_config={})
    with np.testing.assert_raises_regex(ValueError, 'complete model_config'):
        PathDiffusionModel.from_checkpoint(incomplete)


def test_kimodo_notebook(tmp_path):
    import pytest
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')

    _save_kimodo_test_checkpoint(tmp_path)
    project_dir = Path(__file__).resolve().parent
    notebook = nbformat.read(project_dir / 'test_random_constraints.ipynb', as_version=4)
    nbformat.validate(notebook)
    original = "CHECKPOINT_DIR = PROJECT_DIR / 'checkpoints'"
    assert original in notebook.cells[1].source
    notebook.cells[1].source = notebook.cells[1].source.replace(original, f'CHECKPOINT_DIR = Path({str(tmp_path)!r})')
    executed = nbclient.NotebookClient(notebook, timeout=120, kernel_name='python3',
                                       resources={'metadata': {'path': str(project_dir)}}).execute()
    outputs = [output for cell in executed.cells for output in cell.get('outputs', [])]
    assert not any(output.output_type == 'error' for output in outputs)
    assert sum('image/png' in output.get('data', {}) for output in outputs) >= 2
    text = ''.join(output.get('text', '') for output in outputs)
    assert text.count('Validated all four features at') == 4
    assert 'Denoising steps: 4' in text


def test_periodic_and_final_checkpoints(tmp_path):
    from unittest.mock import patch

    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=121, device='cpu')
    trainer = DiffusionTrainer(model, [], [], device='cpu', save_dir=tmp_path)
    epochs = []

    def train_epoch():
        epochs.append(trainer.current_epoch)
        trainer.global_step += 1
        return 1.0

    with patch.object(trainer, 'train_epoch', side_effect=train_epoch), \
            patch.object(trainer, 'validate', return_value=.5):
        trainer.train(105, val_every=7)
    assert epochs == list(range(105))
    assert sorted(path.name for path in tmp_path.glob('epoch_*.pt')) == ['epoch_0050.pt', 'epoch_0100.pt']
    for completed in [50, 100]:
        checkpoint = torch.load(tmp_path / f'epoch_{completed:04d}.pt', weights_only=True)
        assert checkpoint['epoch'] == completed - 1
        assert checkpoint['next_epoch'] == completed
        assert checkpoint['global_step'] == completed
    latest = torch.load(tmp_path / 'latest.pt', weights_only=True)
    best = torch.load(tmp_path / 'best_model.pt', weights_only=True)
    assert latest['next_epoch'] == 105
    assert latest['epoch'] == 104
    assert latest['global_step'] == 105
    assert best['next_epoch'] == 50  # best is written at the first save point after improvement
    assert latest['best_val_loss'] == best['best_val_loss'] == .5


def test_cosine_lr_schedule(tmp_path):
    from unittest.mock import patch

    model = PathDiffusionModel(hidden_dim=16, num_layers=1, num_heads=2,
                               num_timesteps=4, max_seq_len=121, device='cpu')
    trainer = DiffusionTrainer(model, [], [], device='cpu', save_dir=tmp_path,
                               lr_schedule='cosine', lr_start=1e-4, lr_end=2e-5)
    lrs = []

    def train_epoch():
        lrs.append(trainer.optimizer.param_groups[0]['lr'])
        return 1.0

    with patch.object(trainer, 'train_epoch', side_effect=train_epoch), \
            patch.object(trainer, 'validate', return_value=float('inf')):
        trainer.train(5)
    np.testing.assert_allclose(lrs[0], 1e-4)
    np.testing.assert_allclose(lrs[-1], 2e-5)
    assert all(a > b for a, b in zip(lrs, lrs[1:]))

    with np.testing.assert_raises_regex(ValueError, 'Unknown lr_schedule'):
        DiffusionTrainer(model, [], [], device='cpu', save_dir=tmp_path,
                         lr_schedule='linear')


def test_resume_starts_after_saved_epoch(tmp_path):
    from unittest.mock import patch

    trainer = _save_kimodo_test_checkpoint(tmp_path)
    trainer.current_epoch = 49
    trainer.next_epoch = 50
    trainer.global_step = 100
    trainer.save_checkpoint('epoch_0050.pt')
    resumed = DiffusionTrainer(PathDiffusionModel(**trainer.model.config), [], [],
                                device='cpu', save_dir=tmp_path)
    resumed.load_checkpoint('latest')
    assert resumed.current_epoch == 50
    assert resumed.next_epoch == 50
    epochs = []

    def train_epoch():
        epochs.append(resumed.current_epoch)
        resumed.global_step += 2
        return 1.0

    with patch.object(resumed, 'train_epoch', side_effect=train_epoch), \
            patch.object(resumed, 'validate', return_value=float('inf')):
        resumed.train(51)
    assert epochs == [50]
    latest = torch.load(tmp_path / 'latest.pt', weights_only=True)
    assert latest['epoch'] == 50
    assert latest['next_epoch'] == 51
    assert latest['global_step'] == 102
    with patch.object(resumed, 'train_epoch') as train_epoch:
        resumed.train(51)
    train_epoch.assert_not_called()
    older = torch.load(tmp_path / 'epoch_0050.pt', weights_only=True)
    del older['next_epoch']
    torch.save(older, tmp_path / 'older_kimodo.pt')
    with np.testing.assert_raises_regex(ValueError, 'missing required fields'):
        resumed.load_checkpoint(tmp_path / 'older_kimodo.pt')


def test_resume_restores_rng_and_ema_settings(tmp_path):
    import random

    trainer = _save_kimodo_test_checkpoint(tmp_path)
    trainer.ema_decay = .9
    trainer.ema_interval = 3
    trainer.save_checkpoint('latest.pt')
    expected_torch = torch.rand(8)
    expected_numpy = np.random.rand(8)
    expected_python = [random.random() for _ in range(8)]
    expected_cuda = torch.rand(8, device='cuda') if torch.cuda.is_available() else None
    trainer.ema_decay = .1
    trainer.ema_interval = 7
    trainer.load_checkpoint('latest')
    assert trainer.ema_decay == .9
    assert trainer.ema_interval == 3
    torch.testing.assert_close(torch.rand(8), expected_torch, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.rand(8), expected_numpy)
    assert [random.random() for _ in range(8)] == expected_python
    if expected_cuda is not None:
        torch.testing.assert_close(torch.rand(8, device='cuda'), expected_cuda, rtol=0, atol=0)
    trainer.load_checkpoint('latest')
    trainer.train_epoch()
    expected_weights = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
    expected_ema = {name: value.clone() for name, value in trainer.ema_params.items()}
    trainer.load_checkpoint('latest')
    trainer.train_epoch()
    for name, value in trainer.model.named_parameters():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
        torch.testing.assert_close(trainer.ema_params[name], expected_ema[name], rtol=0, atol=0)


def test_failed_checkpoint_write_preserves_latest(tmp_path):
    from unittest.mock import patch

    trainer = _save_kimodo_test_checkpoint(tmp_path)
    original = torch.load(tmp_path / 'latest.pt', weights_only=True)
    trainer.global_step += 1

    def fail_write(checkpoint, path):
        Path(path).write_bytes(b'incomplete checkpoint')
        raise OSError('Simulated failed checkpoint write')

    with patch('train.torch.save', side_effect=fail_write):
        with np.testing.assert_raises_regex(OSError, 'Simulated failed'):
            trainer.save_checkpoint('latest.pt')
    preserved = torch.load(tmp_path / 'latest.pt', weights_only=True)
    assert preserved['global_step'] == original['global_step']
    for name, value in original['raw_model_state_dict'].items():
        torch.testing.assert_close(preserved['raw_model_state_dict'][name], value)


if __name__ == '__main__':
    try:
        test_training()
    except Exception as e:
        print(f"\n❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
