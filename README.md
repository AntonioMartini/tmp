# Path Prediction with Diffusion Models

This project implements an unconditional diffusion model for predicting character root trajectories (paths) from motion capture data. The model learns to generate diverse, plausible future paths, controlled via constraints on generated points.

## Overview

The diffusion model approach treats path prediction as a denoising problem:
- **Forward process**: Gradually add noise to ground truth trajectories
- **Reverse process**: Learn to denoise trajectories
- **Sampling**: Generate diverse predictions by iteratively denoising from random noise

The model is **unconditional**: there is no context input to the denoiser.
Generation is controlled only through constraints (first/last point), enforced
Kimodo-style by zeroing the diffusion noise at constrained positions during
training and injecting forward-noised constraint values during sampling. Each
sequence is canonicalized by translating its first point to the origin (plus a
random rotation during training); at inference, the first-point constraint
serves as the canonicalization anchor.

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
   - Unconditional denoising (no context input); control only via constraints enforced in the diffusion noise
   - Predicts noise added during diffusion process

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
    --min_prediction_length 20 \
    --max_prediction_length 600 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 2e-5 \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_heads 8 \
    --num_timesteps 1000 \
    --save_dir ./checkpoints
```

Key arguments:
- `--min_prediction_length`: Minimum prediction length per sample (default: 20)
- `--max_prediction_length`: Maximum prediction length; sets the padded tensor size (default: 600)
- `--num_timesteps`: Diffusion timesteps (default: 1000)
- `--hidden_dim`: Model hidden dimension (default: 256)
- `--num_layers`: Number of transformer layers (default: 6)
- `--save_dir`: Directory for checkpoints and normalization stats (default: `./checkpoints`, created if missing)
- `--val_every`: Run validation every N epochs (default: 1)
- `--resume`: Path to checkpoint to resume training

Checkpointing:
- Only `best_model.pt` is saved, overwritten whenever validation loss improves
- The checkpoint's `model_state_dict` contains the **EMA weights** (what is deployed at test time); `raw_model_state_dict` and `ema_state_dict` are stored for resuming
- `normalization_stats.npy` is written to `--save_dir` at startup and is required at inference time

Monitoring:
- Per-epoch train/val loss and learning rate are logged to TensorBoard under `<save_dir>/logs/run_NNN` (auto-incremented per run)
- View with `tensorboard --logdir checkpoints/logs` — each `run_NNN` appears as a separate run

How it works:
- Each training sample gets a random sequence length in `[min, max]`, clamped to the frames available in the clip (set both equal for fixed-length)
- Shorter targets are **zero-padded** to `max_prediction_length`
- An `attention_mask` prevents the transformer from attending to padding
- The loss is masked so padded positions contribute nothing
- Every sequence is **canonicalized**: translated so its first point is at the origin, plus a random rotation applied to positions and directions during training
- The **first and last valid point of each target are always constrained** (Kimodo-style: noise is zeroed at constrained positions during both training and sampling)
- At inference, any prediction length up to the trained max works without retraining
- Arbitrary points can be constrained at generation time via `constraint_mask` + `constraint_values` on `model.sample()` / `PathPredictor.predict()`: constrained positions are overwritten with the forward-noised constraint values at every denoising step and set exactly at the final step

### 3. Inference

Generate predictions using a trained model:

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --norm_stats ./checkpoints/normalization_stats.npy \
    --data_dir ./data \
    --motion_data_dir ../motion_2/data \
    --context_length 30 \
    --prediction_length 60 \
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
- `--prediction_length`: Number of frames to predict (any value up to the trained max)
- `--max_prediction_length`: Maximum length the model was trained with (default: `--prediction_length`). Shorter predictions are generated via internal padding, then trimmed.

Constraints (Python API only, not exposed on the CLI): `PathPredictor.predict()` and `model.sample()` accept `constraint_mask` `(prediction_length,)` and `constraint_values` `(prediction_length, 4)`. `predict()` takes them in the raw unnormalized frame and always pins index 0 to the last context frame; `sample()` expects the normalized frame and uses the constrained first point as the canonicalization anchor. **The first and last frame must always be constrained** (matching training): `predict()` accepts a `goal` for the last frame, and `sample()` raises if index 0 or the last valid index is not constrained. Constrained positions are pinned to the given values during generation.

### 4. Evaluation

Evaluate model performance on test set:

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --norm_stats ./checkpoints/normalization_stats.npy \
    --data_dir ./data \
    --motion_data_dir ../motion_2/data \
    --min_prediction_length 20 \
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

Default hyperparameters:
- Feature dimension: 4 (x, z, forward_x, forward_z)
- Sequence length: variable, 20-600 (zero-padded to 600)
- Hidden dimension: 256
- Transformer layers: 6
- Attention heads: 8
- Diffusion timesteps: 1000
- Noise schedule: Linear
- Optimizer: Adam-atan2 (constant LR, no scheduler)
- Learning rate: 2e-5
- Weight decay: 1e-6
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
├── checkpoints/               # Model checkpoints
├── predictions/               # Generated predictions
└── evaluation/                # Evaluation results
```

## Example Workflow

1. Prepare data:
```bash
python data_preprocessing.py
```

2. Train model:
```bash
python train.py --num_epochs 50 --batch_size 16
```

3. Generate predictions (any length up to the trained max):
```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --norm_stats ./data/normalization_stats.npy \
    --prediction_length 45 \
    --max_prediction_length 600 \
    --clip_name "WalkingStickLeft_BR.bvh" \
    --num_samples 10 \
    --visualize
```

4. Evaluate on test set:
```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --norm_stats ./data/normalization_stats.npy \
    --create_visualizations
```

## Notes

- Multiple samples can be generated to capture the stochastic nature of human motion
- Normalization statistics are saved during preprocessing and used for both training and inference
- The model supports both CPU and GPU inference
- Targets are zero-padded to `max_prediction_length`; padding is excluded from attention (`src_key_padding_mask`) and from the loss (masked MSE over valid positions only)
- For fixed-length behavior, set `--min_prediction_length` equal to `--max_prediction_length`

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
- Kimodo: kinematic motion diffusion (noise-based constraints, canonicalization, Adam-atan2 + EMA training recipe)
