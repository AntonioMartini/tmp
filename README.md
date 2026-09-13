# Path Prediction with Diffusion Models

This project implements a constraint-conditioned diffusion model for predicting character root trajectories (paths) from motion capture data, without text or a context-history encoder. The model learns to generate diverse, plausible future paths, controlled via constraints on generated points.

## Overview

The diffusion model approach treats path prediction as a denoising problem:
- **Forward process**: Gradually add noise to ground truth trajectories
- **Reverse process**: Learn to denoise trajectories
- **Sampling**: Generate diverse predictions by iteratively denoising from random noise

The model has **no context-history input**: generation is controlled by clean
trajectory constraints and an explicit constraint mask. During both training and
every inference step, the denoiser receives the Kimodo-style input
`concat([mask * clean_constraints + (1 - mask) * noisy_path, mask])`.
The frame mask is expanded to all four position/direction features, giving an
8-feature input. Constrained values are not attenuated by the noise schedule.
The model predicts the **clean trajectory (`x0`)** and is trained with L1 loss
over valid frames, including constrained frames. The DDPM posterior uses that
prediction directly; sampling also pins constrained outputs exactly.

This adapts Kimodo's constraint conditioning and clean-motion prediction to the
existing root-only transformer. It does not reproduce Kimodo's full-body,
two-stage architecture, text guidance, or DDIM sampler. Each sequence is still
canonicalized by translating its first point to the origin (plus a random
rotation during training); at inference, the first-point constraint is the anchor.

## Data

The model uses two main data files:
- `root_feature_matrix.npy`: Contains root trajectory features (hip position X/Z, forward direction X/Z) for 4,779,750 frames
- `clip_frame_ranges_world.pkl`: Maps 810 motion clips to their frame ranges

### Feature Representation

Each frame is represented by 4 features:
- `hips_x`: Hip position in world X
- `hips_z`: Hip position in world Z  
- `forward_x`: Forward direction X component (normalized)
- `forward_z`: Forward direction Z component (normalized)

## Model Architecture

The model consists of:

1. **PathDenoiser**: A transformer-based denoising network
   - Sinusoidal time embeddings
   - Temporal self-attention layers
   - Clean constraint imputation and concatenated feature masks, without a context encoder
   - Predicts clean trajectories (`x0`) rather than noise; masked L1 reconstruction loss

2. **GaussianDiffusion**: Implements the diffusion process
   - Linear or cosine noise schedule
   - Forward diffusion: q(x_t | x_{t-1})
   - Reverse diffusion: p(x_{t-1} | x_t)
   - Posterior computation for sampling

3. **PathDiffusionModel**: Complete model combining denoiser and diffusion

## Installation

Requirements:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- scikit-learn
- matplotlib
- tqdm
- tensorboard

Install dependencies:
```bash
pip install torch numpy scikit-learn matplotlib tqdm tensorboard
```

## Usage

### 1. Data Preprocessing

First, prepare the training data:

```bash
python data_preprocessing.py
```

This will:
- Extract sequences from clips
- Normalize features
- Split into train/val/test sets
- Save normalization statistics

### 2. Training

Train the diffusion model:

```bash
python train.py \
    --data_dir ./data \
    --motion_data_dir ../motion_2/data \
    --min_prediction_length 120 \
    --max_prediction_length 600 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 2e-5 \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_heads 8 \
    --num_timesteps 1000 \
    --save_dir ./checkpoints_kimodo
```

Arguments (defaults match the command above):
- `--data_dir`: Directory containing `root_feature_matrix.npy` (default: `./data`)
- `--motion_data_dir`: Directory containing `clip_frame_ranges_world.pkl` (default: `../motion_2/data`)
- `--min_prediction_length`: Minimum prediction length per sample (default: 120; must be at least 61, since endpoint constraints require a 60-frame gap)
- `--max_prediction_length`: Maximum prediction length; sets the padded tensor size and the model's `max_seq_len` (default: 600)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Total target epochs (default: 100)
- `--learning_rate`: Constant Adam-atan2 learning rate (default: 2e-5)
- `--hidden_dim`: Model hidden dimension (default: 256)
- `--num_layers`: Number of transformer layers (default: 6)
- `--num_heads`: Number of attention heads (default: 8)
- `--dropout`: Dropout for the transformer layers and positional encoding (default: 0.1)
- `--num_timesteps`: Diffusion timesteps (default: 1000)
- `--device`: `cuda` or `cpu` (default: `cuda`, falls back to CPU if unavailable)
- `--save_dir`: Directory for checkpoints and normalization stats (default: `./checkpoints_kimodo`, created if missing)
- `--val_every`: Run validation every N epochs (default: 1)
- `--save_every`: Write checkpoints (numbered epoch file, `latest.pt`, `best_model.pt`) every N completed epochs (default: 50); `latest.pt` is also written at the final epoch
- `--resume`: Path to a checkpoint, or `latest` to load `<save_dir>/latest.pt`. `--num_epochs` is the total target epoch count, including epochs already completed before resuming

There is no `--context_length` training argument: the model has no context-history input, so training samples are just target sequences. Context length only exists in `inference.py`, where the last context frame anchors the first predicted point.

Checkpointing:
- `epoch_0050.pt`, `epoch_0100.pt`, ... are saved every `--save_every` completed epochs; nothing is written between save points
- `latest.pt` is refreshed at every save point and at the final epoch; it is the restart point after a crash or interruption
- `best_model.pt` is updated at save points when validation has improved since the previous save; it holds an EMA snapshot taken at the best epoch and remains the deployment checkpoint
- The checkpoint's `model_state_dict` contains the **EMA weights** (what is deployed at test time); `raw_model_state_dict` and `ema_state_dict` are stored for resuming
- Checkpoints record the completed epoch count, `global_step`, optimizer state, EMA settings, and Python/NumPy/PyTorch RNG state; resuming continues at the **next** epoch (a checkpoint with `next_epoch=50` resumes at epoch 51) rather than repeating the saved epoch
- Writes are atomic (temporary file + rename), so a failed save never corrupts the previous `latest.pt`
- `normalization_stats.npy` is written to `--save_dir` at startup and is required at inference time
- New checkpoints store `format_version=2`, `prediction_type='x0'`, `constraint_conditioning='clean_imputation_mask'`, and the complete model architecture/noise-schedule configuration. Inference and evaluation restore these settings instead of hardcoding them.
- **Retraining is required:** old 4-input noise-prediction checkpoints are rejected by inference, evaluation, and training resume. The new input projection has 8 features and a different prediction objective. Use a new run without `--resume`; the default output directory is now `./checkpoints_kimodo` to preserve existing weights.
- Resuming a new-format checkpoint requires matching model hyperparameters. The new L1 validation losses are not comparable to the old noise MSE values.

Monitoring:
- Per-epoch train/val loss and learning rate are logged to TensorBoard under `<save_dir>/logs/run_NNN` (auto-incremented per run)
- View with `tensorboard --logdir checkpoints_kimodo/logs` — each `run_NNN` appears as a separate run

How it works:
- Clips shorter than `--min_prediction_length` are skipped; clips longer than 700 frames are first split into near-equal chunks
- Sequences are z-score normalized over all extracted sequences, then split 80/20 into train+val vs. test, with 10% of the remainder going to validation (~72/8/20), using seed 42
- Each training sample gets a random sequence length in `[min, max]`, clamped to the frames available in the clip (set both equal for fixed-length)
- Shorter targets are **zero-padded** to `max_prediction_length`
- An `attention_mask` prevents the transformer from attending to padding
- The loss is masked so padded positions contribute nothing
- Every sequence is **canonicalized**: translated so its first point is at the origin, plus a random rotation applied to positions and directions during training
- The **first and last valid point and direction are always constrained**, along with randomly selected interior frames. A mask entry pins all four features, not just position.
- The same `generate_constraint_mask()` helper is used in training, validation, inference, and evaluation. It draws an interior count uniformly from zero to the maximum feasible count, then samples valid indices. All consecutive constrained indices differ by **at least 60 frames**, including endpoint-to-interior gaps. Padding is never constrained.
- Trajectories must have at least 61 valid frames. Lengths 61–120 permit endpoints only; length 121 first permits an interior constraint at index 60. Dataset sequences shorter than 61 frames are excluded.
- Training first noises the target normally, then replaces the constrained input features with their clean values and concatenates the feature mask inside the denoiser. The same imputation is used at every inference step. No timestep scaling is applied to constraint values.
- Explicit masks are supported but rejected if they violate the 60-frame spacing rule. Legacy noise-prediction checkpoints are incompatible with the new mask-conditioned clean-prediction model and must be retrained.

### 3. Inference

Generate predictions using a trained model:

```bash
python inference.py \
    --checkpoint ./checkpoints_kimodo/best_model.pt \
    --norm_stats ./checkpoints_kimodo/normalization_stats.npy \
    --data_dir ./data \
    --motion_data_dir ../motion_2/data \
    --context_length 30 \
    --prediction_length 181 \
    --max_prediction_length 600 \
    --num_samples 5 \
    --clip_name "PendulumHands_BR.bvh" \
    --visualize \
    --output_dir ./predictions
```

Key arguments:
- `--checkpoint`: Path to trained model checkpoint
- `--norm_stats`: Path to normalization statistics
- `--num_samples`: Number of diverse predictions to generate
- `--context_length`: Number of context frames to load (default: 30); only the last frame is used to pin the first predicted point
- `--clip_name`: Specific clip to use for context (optional)
- `--visualize`: Create visualization plots
- `--prediction_length`: Number of frames to predict (default: 120; at least 61 and no greater than the trained max)
- `--max_prediction_length`: Maximum length the model was trained with (default: `--prediction_length`). Shorter predictions are generated via internal padding, then trimmed.

`PathPredictor.predict()` accepts a raw `constraint_values` array `(prediction_length, 4)` and an optional binary `constraint_mask` `(prediction_length,)`. Without a mask it generates random interior entries plus both endpoints. With a mask it uses your selections, always pins frame 0 to `context[-1]`, and pins the last frame to `goal` (or the supplied last reference value). Explicit interior constraints require values; conflicting frame-0 values cannot override the context anchor. Every selected frame pins both position and direction.

If no interior values are provided, inference uses linearly interpolated endpoint positions and shortest-arc unit headings at the random indices. Providing a reference trajectory instead uses its actual values at those indices. `predict_batch()` supports the same randomized behavior with optional raw reference paths shaped `(B, max_prediction_length, 4)`. To inspect or visualize a mask before inference, call `inference.build_constraints()` and pass its returned mask and values to `predict()`.

The CLI samples interior constraints from the selected clip's actual future trajectory when `--clip_name` is given. With no clip it uses endpoint interpolation. This is constrained reconstruction when a reference future is supplied, not unconstrained forecasting.

`model.sample()` takes batched, normalized `constraint_mask` `(B, T)` and `constraint_values` `(B, T, 4)`. If the mask is omitted, supply a full reference path so every randomly selected frame has a meaningful value. Explicit masks must include both valid endpoints and obey the 60-frame minimum spacing. The first constrained point anchors canonicalization. `PathPredictor` restores the saved maximum sequence length and diffusion-step count automatically. An optional `max_seq_len` argument must match the checkpoint.

Random mask generation uses PyTorch's RNG; use `torch.manual_seed(seed)` to reproduce it. The test notebook also seeds its NumPy generator for reproducible synthetic waypoint values.

### 4. Evaluation

Evaluate model performance on test set:

```bash
python evaluate.py \
    --checkpoint ./checkpoints_kimodo/best_model.pt \
    --norm_stats ./checkpoints_kimodo/normalization_stats.npy \
    --data_dir ./data \
    --motion_data_dir ../motion_2/data \
    --min_prediction_length 120 \
    --max_prediction_length 600 \
    --batch_size 32 \
    --num_samples 5 \
    --create_visualizations \
    --num_visualizations 5 \
    --output_dir ./evaluation
```

Key arguments:
- `--num_samples`: Number of samples for stochastic evaluation
- `--num_eval_batches`: Number of batches to evaluate (optional)
- `--create_visualizations`: Generate visualization plots
- `--num_visualizations`: Number of visualization samples
- `--min_prediction_length`, `--max_prediction_length`: Must match the training configuration

## Evaluation Metrics

The model is evaluated using three metrics:

1. **Average Displacement Error (ADE)**: Mean Euclidean distance between predicted and ground truth positions over all time steps
2. **Final Displacement Error (FDE)**: Euclidean distance at the final time step
3. **Direction Error**: Mean angular error in forward direction (degrees)

## Model Configuration

Default hyperparameters (as defined in `train.py` and `diffusion_model.py`):
- Trajectory/output dimension: 4 (x, z, forward_x, forward_z); denoiser input dimension: 8 (imputed trajectory + feature mask)
- Prediction objective: clean trajectory (`x0`), masked L1 reconstruction loss over valid frames
- Sequence length: variable, 120-600 by default (zero-padded to `max_prediction_length`; minimum supported length is 61)
- Hidden dimension: 256; feedforward dimension: 1024 (4x hidden)
- Transformer layers: 6, post-norm (`norm_first=False`), GELU activation
- Attention heads: 8
- Dropout: 0.1 (transformer layers and positional encoding)
- Positional encoding: sinusoidal, non-learned; timestep embedding uses a separate sinusoidal table (sized to `num_timesteps`) plus a 2-layer SiLU MLP, broadcast to every position
- Diffusion timesteps: 1000
- Noise schedule: linear betas from 1e-4 to 0.02 (`beta_schedule='linear'`; a cosine schedule exists but is not used)
- Optimizer: Adam-atan2 (constant LR, no scheduler), betas (0.9, 0.999), eps 1e-8
- Learning rate: 2e-5
- Weight decay: 1e-6
- Gradient clipping: max norm 1.0
- EMA: decay 0.995, updated every 10 steps; used for validation and checkpoints

## File Structure

```
path_prediction/
├── data_preprocessing.py      # Data loading and preprocessing
├── diffusion_model.py         # Model architecture
├── train.py                   # Training script
├── inference.py               # Inference script
├── evaluate.py                # Evaluation script
├── data/
│   ├── root_feature_matrix.npy
│   └── normalization_stats.npy
├── checkpoints_kimodo/        # New-format model checkpoints
├── predictions/               # Generated predictions
└── evaluation/                # Evaluation results
```

## Example Workflow

1. Prepare data:
```bash
python data_preprocessing.py
```

2. Train model (resume an interrupted run with `--resume latest`):
```bash
python train.py --num_epochs 1000 --batch_size 16
python train.py --num_epochs 1000 --batch_size 16 --resume latest
```

3. Generate predictions (any length up to the trained max):
```bash
python inference.py \
    --checkpoint ./checkpoints_kimodo/best_model.pt \
    --norm_stats ./checkpoints_kimodo/normalization_stats.npy \
    --prediction_length 181 \
    --max_prediction_length 600 \
    --clip_name "WalkingStickLeft_BR.bvh" \
    --num_samples 10 \
    --visualize
```

4. Evaluate on test set:
```bash
python evaluate.py \
    --checkpoint ./checkpoints_kimodo/best_model.pt \
    --norm_stats ./checkpoints_kimodo/normalization_stats.npy \
    --create_visualizations
```

## Notes

- Multiple samples can be generated to capture the stochastic nature of human motion
- Normalization statistics are saved during preprocessing and used for both training and inference
- The model supports both CPU and GPU inference
- Targets are zero-padded to `max_prediction_length`; padding is excluded from attention (`src_key_padding_mask`) and from the loss (masked clean-trajectory L1 over valid positions only)
- For fixed-length behavior, set `--min_prediction_length` equal to `--max_prediction_length`

## Verification

- Synthetic constraint, padding, inference, and training-loss regression tests: `python -m pytest -q test_variable_length.py`
- Full smoke tests, including a small-model training epoch using the local dataset: `python -m pytest -q test_variable_length.py test_training.py quick_test.py`
- Real-checkpoint notebook: `jupyter nbconvert --to notebook --execute --inplace test_random_constraints.ipynb --ExecutePreprocessor.timeout=600`
- The training smoke test creates a new TensorBoard run under `test_checkpoints/logs`; it does not replace the production checkpoint. Checkpoint regression tests train a tiny model in temporary directories, verify new-format reload and legacy rejection, cover periodic/`latest.pt` saving, resume-from-next-epoch with RNG/EMA restoration, atomic-write failure handling, and execute the entire notebook against that test checkpoint without persisting its outputs.
- The notebook normally reloads `checkpoints_kimodo/best_model.pt` and does not train or save weights. Train a new model before running it normally. Old notebook outputs are cleared to avoid presenting legacy predictions as new-model results.

## Troubleshooting

**Out of memory**: Reduce `--batch_size` or `--hidden_dim`

**Slow training**: Reduce `--num_timesteps` or `--num_layers`

**Poor predictions**: 
- Increase training epochs
- Try different learning rates
- Check data quality and preprocessing
- Ensure proper normalization

**CUDA errors**: 
- Check GPU availability with `torch.cuda.is_available()`
- Verify PyTorch CUDA installation
- Use `--device cpu` to run on CPU

## References

This implementation is inspired by recent work on diffusion models for motion generation:
- DDPM: Denoising Diffusion Probabilistic Models
- MotionDiff: Diffusion models for human motion prediction
- SkeletonDiffusion: Nonisotropic Gaussian diffusion for motion prediction
- [Kimodo technical report, sections 4.2–4.3](https://arxiv.org/html/2603.15546v1): clean constraint imputation, explicit mask conditioning, clean-motion prediction, and the Adam-atan2 + EMA training recipe
