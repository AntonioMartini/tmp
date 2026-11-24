"""
Diffusion Model for Path Prediction

This module implements a denoising diffusion probabilistic model (DDPM) for trajectory generation,
designed to predict future paths using a transformer encoder architecture with sinusoidal positional embeddings.
Based on the Kimodo transformer denoiser architecture adapted for path prediction.

Model Overview:
---------------
The PathDiffusionModel follows the DDPM framework where:
1. Forward Process: Gradually add Gaussian noise to ground truth trajectories over T timesteps
2. Reverse Process: Learn to denoise trajectories step-by-step to generate new samples
3. Training: Predict the clean trajectory with a masked L1 reconstruction loss
4. Inference: Generate samples by iteratively denoising from pure Gaussian noise

Key Components:
--------------
1. PositionalEncoding: Sinusoidal positional encoding for sequence positions
2. TimestepEmbedder: Encoder for diffusion steps using positional encoding + MLP
3. TransformerEncoderBlock: PyTorch TransformerEncoder layer for sequence modeling
4. PathDenoiser: Main denoising network using transformer encoder with timestep conditioning
5. GaussianDiffusion: Implements the forward and reverse diffusion processes
6. PathDiffusionModel: Complete model combining denoiser and diffusion process

Algorithms:
-----------
- DDPM (Denoising Diffusion Probabilistic Models): The core diffusion framework
- Linear/Cosine Beta Schedules: Noise scheduling strategies for the diffusion process
- Transformer Encoder Architecture: Self-attention mechanisms without causal masking
- Sinusoidal Positional Encoding: Standard positional encoding for transformer inputs
- Timestep Conditioning: Timestep embedding is added to each position (broadcast across sequence)
- Kimodo Constraints: Impute clean values and concatenate the feature mask before denoising

Architecture Details:
--------------------
- Input: Imputed trajectory plus feature mask (B, T, 8); 4 position/direction values + 4 mask features
- Output: Predicted clean trajectory x_0 (B, T, 4)
- Fixed Maximum Sequence Length: Configurable (default: 512)
- Hidden Dimension: Configurable (default: 256)
- Number of Layers: Configurable (default: 6)
- Attention Heads: Configurable (default: 8)
- Diffusion Timesteps: Configurable (default: 100)

The model uses a transformer encoder architecture with sinusoidal positional embeddings for position-aware processing.
Timestep embedding is broadcast and added to each position in the sequence (Kimodo-style).
Constraints are clean-imputed and explicitly mask-conditioned during training and sampling.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


MIN_CONSTRAINT_SPACING = 60
MODEL_FORMAT_VERSION = 3


def _valid_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2 or not all(attention_mask.shape):
        raise ValueError('attention_mask must be a nonempty (B, T) tensor')
    lengths = attention_mask.sum(dim=1).long()
    expected = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None] < lengths[:, None]
    if not torch.equal(attention_mask, expected.to(attention_mask.dtype)):
        raise ValueError('attention_mask must contain a valid prefix of ones followed by zeros')
    if (lengths <= MIN_CONSTRAINT_SPACING).any():
        raise ValueError('Trajectories need at least 61 valid frames for endpoints 60 frames apart')
    return lengths


def generate_constraint_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = _valid_lengths(attention_mask)
    mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    rows = torch.arange(len(lengths), device=attention_mask.device)
    last = lengths - 1
    mask[:, 0] = 1
    mask[rows, last] = 1
    capacity = last // MIN_CONSTRAINT_SPACING - 1
    counts = (torch.rand(len(lengths), device=mask.device) * (capacity + 1)).long()
    previous = torch.zeros_like(last)
    for slot in range(1, attention_mask.shape[1] // MIN_CONSTRAINT_SPACING):
        active = counts >= slot
        low = previous + MIN_CONSTRAINT_SPACING
        high = last - (counts - slot + 1) * MIN_CONSTRAINT_SPACING
        offset = (torch.rand(len(lengths), device=mask.device) * (high - low + 1).clamp(min=1)).long()
        index = torch.where(active, low + offset, 0)
        mask[rows, index] = 1
        previous = torch.where(active, index, previous)
    return mask


def validate_constraint_mask(constraint_mask: torch.Tensor, attention_mask: torch.Tensor):
    lengths = _valid_lengths(attention_mask)
    if constraint_mask.shape != attention_mask.shape:
        raise ValueError('constraint_mask must have the same (B, T) shape as attention_mask')
    if not ((constraint_mask == 0) | (constraint_mask == 1)).all():
        raise ValueError('constraint_mask must contain only 0 or 1')
    rows = torch.arange(len(lengths), device=constraint_mask.device)
    if not (constraint_mask[:, 0] == 1).all() or not (constraint_mask[rows, lengths - 1] == 1).all():
        raise ValueError('constraint_mask must constrain the first and last valid frame')
    if (constraint_mask.bool() & ~attention_mask.bool()).any():
        raise ValueError('Constraints cannot be placed in padding')
    indices = constraint_mask.nonzero()
    same_sequence = indices[1:, 0] == indices[:-1, 0]
    gaps = indices[1:, 1] - indices[:-1, 1]
    if (same_sequence & (gaps < MIN_CONSTRAINT_SPACING)).any():
        raise ValueError('Constrained frames must be at least 60 frames apart, including endpoints')


class PositionalEncoding(nn.Module):
    """Non-learned sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        dropout: Optional[float] = 0.0,
        max_len: Optional[int] = 5000,
    ):
        """
        Args:
            d_model (int): input dim
            dropout (Optional[float] = 0.0): dropout probability on output
            max_len (Optional[int] = 5000): maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len: int, d_model: int, device=None) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)

        # Note: have to replace torch.exp() and math.log() with torch.pow()
        # due to MKL exp() and ln() throws floating point exceptions on certain CPUs
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2, device=device).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input sequence.

        Args:
            x (torch.Tensor): [B, T, D] input motion sequence

        Returns:
            torch.Tensor: [B, T, D] input motion with PE added to it (and optionally dropout)
        """
        if x.shape[1] > self.pe.shape[1]:
            # Inference is not clamped to the training sequence length; extend
            # the deterministic sinusoidal table to cover longer sequences.
            self.pe = self._build_pe(x.shape[1], self.pe.shape[-1], x.device)
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)





class TimestepEmbedder(nn.Module):
    """Encoder for diffusion step."""

    def __init__(self, latent_dim: int, sequence_pos_encoder: PositionalEncoding):
        """
        Args:
            latent_dim (int): dim to encode to
            sequence_pos_encoder (PositionalEncoding): the PE to use on timesteps
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps by adding PE then going through linear layers.

        Args:
            timesteps (torch.Tensor): [B]

        Returns:
            torch.Tensor: [B, 1, D]
        """
        return self.time_embed(self.sequence_pos_encoder.pe.transpose(0, 1)[timesteps])


class PathDenoiser(nn.Module):
    """
    Denoising network for path prediction using transformer encoder architecture.
    
    This denoiser processes noisy trajectories with sinusoidal positional embeddings.
    The model has no context encoder. Clean constraints overwrite the noisy target
    and their feature-wise binary mask is concatenated before input projection.
    The denoiser predicts clean trajectories, following Kimodo-style conditioning.
    The timestep embedding is prepended as a prefix token that pose tokens read
    through self-attention (Kimodo-style).
    
    Based on the Kimodo transformer denoiser architecture adapted for path prediction.
    
    Input format: position + direction (x, y, dx, dy) where dx, dy represent direction.
    """
    
    def __init__(
        self,
        feature_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_size: Optional[int] = None,
        dropout: float = 0.0,
        pe_dropout: float = 0.0,
        max_seq_len: int = 512,
        num_timesteps: int = 100,
        norm_first: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_timesteps = num_timesteps
        # Kimodo uses ff_size = 2x latent_dim (2048 for latent 1024)
        ff_size = hidden_dim * 2 if ff_size is None else ff_size
        
        # Positional encoding (+1 row for the timestep prefix token)
        self.sequence_pos_encoder = PositionalEncoding(hidden_dim, pe_dropout, max_seq_len + 1)
        
        # Timestep embedder (separate PE table since timesteps can exceed max_seq_len)
        self.timestep_pos_encoder = PositionalEncoding(hidden_dim, 0.0, num_timesteps)
        self.embed_timestep = TimestepEmbedder(hidden_dim, self.timestep_pos_encoder)
        
        # Input projection for imputed position+direction and the feature mask
        self.input_linear = nn.Linear(feature_dim * 2, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, feature_dim)
        
        # Transformer encoder with multiple layers
        trans_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            trans_enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
    
    def forward(
        self,
        noisy_path: torch.Tensor,
        timestep: torch.Tensor,
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noisy_path: (B, T_pred, 4) noisy trajectory to denoise [x, y, dx, dy]
            timestep: (B,) diffusion timestep
            constraint_mask: Optional binary (B, T_pred) frame mask, expanded to all
                           features and concatenated to the clean-imputed motion.
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
                           (prevents attention to padded positions)
            constraint_values: Clean normalized (B, T_pred, 4) values. Required when
                           any frame is constrained; unselected values are ignored.
            
        Returns:
            (B, T_pred, 4) predicted clean trajectory
        """
        B, T_pred, _ = noisy_path.shape
        if constraint_mask is None:
            if constraint_values is not None:
                raise ValueError('constraint_mask is required when supplying constraint_values')
            constraint_mask = noisy_path.new_zeros(B, T_pred)
        if constraint_mask.shape != (B, T_pred):
            raise ValueError('constraint_mask must have shape (B, T_pred)')
        mask = constraint_mask.to(dtype=noisy_path.dtype).unsqueeze(-1).expand_as(noisy_path)
        if constraint_values is None:
            if constraint_mask.any():
                raise ValueError('Clean constraint_values are required at constrained frames')
            constraint_values = torch.zeros_like(noisy_path)
        if constraint_values.shape != noisy_path.shape:
            raise ValueError('constraint_values must have the same shape as noisy_path')
        # Clean imputation: the denoiser sees the clean constraint values at
        # constrained frames and the noisy path x_t everywhere else
        imputed = torch.where(mask.bool(), constraint_values, noisy_path)
        if attention_mask is not None:
            # Force padded positions to exact zeros in both the imputed path and
            # the mask channel, so the input projection sees an all-zero
            # 8-feature vector there (padding is also excluded from attention)
            valid = attention_mask.bool().unsqueeze(-1)
            imputed = imputed.masked_fill(~valid, 0)
            mask = mask.masked_fill(~valid, 0)
        
        # Project imputed path concatenated with its feature-wise constraint mask
        noisy_proj = self.input_linear(torch.cat([imputed, mask], dim=-1))  # (B, T_pred, hidden_dim)
        
        # Encode timestep as a prefix token (Kimodo-style): pose tokens read it
        # through self-attention
        emb_time = self.embed_timestep(timestep)  # (B, 1, hidden_dim)
        xseq = torch.cat((emb_time, noisy_proj), dim=1)  # (B, 1 + T_pred, hidden_dim)

        # Add positional encoding
        xseq = self.sequence_pos_encoder(xseq)

        # Apply attention mask if provided (for padding)
        if attention_mask is not None:
            # Convert attention mask to key padding mask format for transformer
            # attention_mask: (B, T_pred) where 1 = valid, 0 = padding
            # key_padding_mask: (B, 1 + T_pred) where True = padding (ignore), False = valid
            # The timestep prefix token is always valid
            key_padding_mask = (attention_mask == 0).bool()
            time_mask = torch.zeros(B, 1, dtype=torch.bool, device=xseq.device)
            key_padding_mask = torch.cat((time_mask, key_padding_mask), dim=1)
        else:
            key_padding_mask = None

        # Input to the transformer
        if isinstance(self.seqTransEncoder, nn.TransformerEncoder):
            assert not self.seqTransEncoder.use_nested_tensor, "Flash attention should be disabled due to bug!"

        output = self.seqTransEncoder(xseq, src_key_padding_mask=key_padding_mask)
        output = output[:, 1:]  # (B, T_pred, hidden_dim): slice off the timestep token

        # Output projection
        clean_pred = self.output_linear(output)
        
        return clean_pred


class GaussianDiffusion:
    """
    Gaussian diffusion process for path generation.
    Implements the forward and reverse diffusion processes.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = 'linear',
        device: str = 'cpu'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Create beta schedule
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        elif schedule == 'cosine':
            self.betas = self.cosine_beta_schedule(num_timesteps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Precompute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def cosine_beta_schedule(num_timesteps: int, s: float = 0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to the data according to the diffusion schedule.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the posterior distribution
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_t
    
    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        enforce_constraints: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the reverse diffusion process
        p(x_{t-1} | x_t) using the model to predict x_0.
        Clean constraints and their mask are supplied to the denoiser at every step.
        """
        # Predict clean motion directly; forward() would add more noise to x_t
        pred_x_start = model.denoiser(
            x_t, t, constraint_mask=constraint_mask, attention_mask=attention_mask,
            constraint_values=constraint_values
        )
        
        # Use the predicted x_0 directly, preserving constraints even when clipping
        if clip_denoised:
            # Very wide guard against early-step divergence only: canonicalized
            # positions legitimately reach ~4 sigma, so a tighter bound flattens
            # real tails. Use clip_denoised=False to disable entirely.
            pred_x_start = torch.clamp(pred_x_start, -100.0, 100.0)
        if enforce_constraints and constraint_mask is not None and constraint_values is not None:
            pred_x_start = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, pred_x_start)
        
        # Compute posterior mean and variance
        posterior_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            pred_x_start, x_t, t
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True,
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        enforce_constraints: bool = True
    ) -> torch.Tensor:
        """
        Sample from the reverse diffusion process p(x_{t-1} | x_t).
        """
        posterior_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            model, x_t, t, clip_denoised, constraint_mask, attention_mask, constraint_values, enforce_constraints
        )

        noise = torch.randn_like(x_t)

        # Suppress sampling noise at frames that are pinned to clean constraints
        if constraint_mask is not None:
            noise = noise * (1 - constraint_mask.unsqueeze(-1).to(noise.dtype))

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        sample = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

        # Overwrite constrained positions with the clean constraint values
        # (Kimodo-style: constraints are directly overwritten in the noisy input each
        # step, so the denoiser sees the clean constraint signal at the next step)
        if enforce_constraints and constraint_mask is not None and constraint_values is not None:
            sample = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, sample)

        return sample
    
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        sampling_steps: int,
        clip_denoised: bool = True,
        device: str = 'cuda',
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        enforce_final_constraints: bool = True
    ) -> torch.Tensor:
        if not 1 <= sampling_steps < self.num_timesteps:
            raise ValueError('sampling_steps must be between 1 and num_timesteps - 1 for DDIM sampling')
        B = shape[0]
        img = torch.randn(shape, device=device)
        if constraint_mask is not None and constraint_values is not None:
            img = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, img)
        timesteps = torch.linspace(0, self.num_timesteps - 1, sampling_steps, device=device).round().long().flip(0)

        for index, i in enumerate(timesteps.tolist()):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_x_start = model.denoiser(
                img, t, constraint_mask=constraint_mask, attention_mask=attention_mask,
                constraint_values=constraint_values
            )
            if clip_denoised:
                pred_x_start = torch.clamp(pred_x_start, -100.0, 100.0)
            enforce_constraints = enforce_final_constraints or i > 0
            if enforce_constraints and constraint_mask is not None and constraint_values is not None:
                pred_x_start = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, pred_x_start)

            alpha_t = self._extract(self.alphas_cumprod, t, img.shape)
            next_i = timesteps[index + 1].item() if index + 1 < len(timesteps) else -1
            alpha_next = (torch.ones_like(alpha_t) if next_i < 0 else
                          self._extract(self.alphas_cumprod, torch.full((B,), next_i, device=device, dtype=torch.long), img.shape))
            pred_noise = (img - torch.sqrt(alpha_t) * pred_x_start) / torch.sqrt(1.0 - alpha_t)
            img = torch.sqrt(alpha_next) * pred_x_start + torch.sqrt(1.0 - alpha_next) * pred_noise
            if enforce_constraints and constraint_mask is not None and constraint_values is not None:
                img = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, img)

        return img

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        clip_denoised: bool = True,
        device: str = 'cuda',
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        enforce_final_constraints: bool = True,
        sampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples by iteratively denoising.
        """
        if sampling_steps is not None and sampling_steps != self.num_timesteps:
            return self.ddim_sample_loop(
                model, shape, sampling_steps, clip_denoised, device, constraint_mask,
                attention_mask, constraint_values, enforce_final_constraints
            )
        B = shape[0]
        img = torch.randn(shape, device=device)

        # Initialize constrained positions with the clean constraint values
        if constraint_mask is not None and constraint_values is not None:
            img = torch.where(constraint_mask.unsqueeze(-1).bool(), constraint_values, img)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                model, img, t, clip_denoised, constraint_mask,
                attention_mask, constraint_values, enforce_final_constraints or i > 0
            )

        return img
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape to x_shape."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class PathDiffusionModel(nn.Module):
    """
    Complete diffusion model for path prediction using transformer encoder architecture.
    Combines the denoiser with the diffusion process.
    
    Clean constraint imputation and concatenated feature masks condition every denoising step (Kimodo-style).
    The timestep embedding is a prefix token read through self-attention (Kimodo-style).
    """
    
    def __init__(
        self,
        feature_dim: int = 4,
        num_timesteps: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_size: Optional[int] = None,
        dropout: float = 0.0,
        pe_dropout: float = 0.0,
        beta_schedule: str = 'linear',
        max_seq_len: int = 512,
        norm_first: bool = False,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        # Kimodo uses ff_size = 2x latent_dim (2048 for latent 1024)
        ff_size = hidden_dim * 2 if ff_size is None else ff_size
        self.config = dict(
            feature_dim=feature_dim, num_timesteps=num_timesteps, hidden_dim=hidden_dim,
            num_layers=num_layers, num_heads=num_heads, ff_size=ff_size,
            dropout=dropout, pe_dropout=pe_dropout,
            beta_schedule=beta_schedule, max_seq_len=max_seq_len, norm_first=norm_first
        )
        self.denoiser = PathDenoiser(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
            pe_dropout=pe_dropout,
            max_seq_len=max_seq_len,
            num_timesteps=num_timesteps,
            norm_first=norm_first
        )
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            schedule=beta_schedule,
            device=device
        )
    
    @staticmethod
    def checkpoint_config(checkpoint):
        if (checkpoint.get('format_version') != MODEL_FORMAT_VERSION
                or checkpoint.get('prediction_type') != 'x0'
                or checkpoint.get('constraint_conditioning') != 'clean_imputation_mask'):
            raise ValueError(
                'Incompatible checkpoint: Kimodo conditioning requires clean x0 prediction, '
                'an 8-feature imputed-path/mask input, and timestep prefix-token conditioning. '
                'Older checkpoints must be retrained; start a new run with '
                '--save_dir ./checkpoints_kimodo, without --resume.'
            )
        config = checkpoint.get('model_config')
        required = {'feature_dim', 'num_timesteps', 'hidden_dim', 'num_layers', 'num_heads',
                    'ff_size', 'dropout', 'pe_dropout', 'beta_schedule', 'max_seq_len', 'norm_first'}
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError('Checkpoint is missing a complete model_config')
        return dict(config)

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        model = cls(**cls.checkpoint_config(checkpoint), device=device).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.eval()

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.device = device
        self.diffusion.device = device
        self.diffusion.betas = self.diffusion.betas.to(device)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.diffusion.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)
        self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
        self.diffusion.posterior_log_variance_clipped = self.diffusion.posterior_log_variance_clipped.to(device)
        self.diffusion.posterior_mean_coef1 = self.diffusion.posterior_mean_coef1.to(device)
        self.diffusion.posterior_mean_coef2 = self.diffusion.posterior_mean_coef2.to(device)
        return self
    
    def forward(
        self,
        target: torch.Tensor,
        timestep: torch.Tensor,
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training. Noise the target, then condition the denoiser
        on clean-imputed constraints and their feature mask to predict x_0.
        Args:
            target: (B, T_pred, 4) ground truth trajectory [x, y, dx, dy]
            timestep: (B,) diffusion timestep
            constraint_mask: Optional mask (B, T_pred) for constrained positions
                           (clean values and mask are passed into the denoiser)
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
        Returns:
            (B, T_pred, 4) predicted clean trajectory
        """
        if attention_mask is None:
            attention_mask = torch.ones(target.shape[:2], device=target.device)
        if constraint_mask is None:
            constraint_mask = generate_constraint_mask(attention_mask)
        validate_constraint_mask(constraint_mask, attention_mask)
        constraint_mask = constraint_mask.to(dtype=target.dtype)

        # Add noise to target
        noise = torch.randn_like(target)

        # Impute clean constraints inside the denoiser after the forward diffusion
        noisy_target = self.diffusion.q_sample(target, timestep, noise)

        # Predict the clean trajectory with explicit constraint-mask conditioning
        clean_pred = self.denoiser(
            noisy_target, timestep, constraint_mask=constraint_mask,
            attention_mask=attention_mask, constraint_values=target
        )

        return clean_pred
    
    @torch.no_grad()
    def sample(
        self,
        prediction_length: int,
        num_samples: int = 1,
        clip_denoised: bool = True,
        constraint_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda',
        enforce_final_constraints: bool = True,
        sampling_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate paths without text or a context encoder; control is supplied via
        constraint_mask/constraint_values.

        The model is trained with the first and last valid frame always
        constrained, so both must be constrained here as well.
        Args:
            prediction_length: Number of frames to predict
            num_samples: Number of samples to generate per batch element
            clip_denoised: Whether to clip denoised predictions
            constraint_mask: Optional binary (B, T_pred) mask. When absent,
                           randomly select interior frames plus both endpoints.
                           Explicit masks must include index 0 and the last valid
                           index, with at least 60 frames between selected indices.
            constraint_values: (B, T_pred, 4) normalized positions and directions.
                           When the mask is absent, supply the full reference path
                           so randomly selected frames have meaningful values.
                           The first point anchors trajectory canonicalization.
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
            device: Device to run on
        Returns:
            (B, num_samples, T_pred, 4) generated trajectories
        """
        self.eval()
        feature_dim = self.feature_dim

        if constraint_values is None:
            raise ValueError('constraint_values are required for endpoint and interior constraints')
        if constraint_values.ndim != 3 or constraint_values.shape[1:] != (prediction_length, feature_dim):
            raise ValueError('constraint_values must have shape (B, prediction_length, feature_dim)')
        if not torch.isfinite(constraint_values).all():
            raise ValueError('constraint_values must be finite')
        B = constraint_values.shape[0]
        if num_samples < 1:
            raise ValueError('num_samples must be positive')

        if attention_mask is None:
            attention_mask = torch.ones(B, prediction_length, device=constraint_values.device)
        if attention_mask.shape != (B, prediction_length):
            raise ValueError('attention_mask must have shape (B, prediction_length)')
        if constraint_mask is None:
            constraint_mask = generate_constraint_mask(attention_mask)

        # Verify the first and last valid frame are constrained
        validate_constraint_mask(constraint_mask, attention_mask)
        constraint_mask = constraint_mask.to(dtype=constraint_values.dtype)

        # Canonicalize: translate so the constrained first point sits at the
        # origin (matches training, where the first point is always constrained).
        origin = constraint_values[:, :1, :2]
        constraint_values = torch.cat(
            [constraint_values[..., :2] - origin, constraint_values[..., 2:]], dim=-1
        )

        # Repeat constraint mask / values if provided
        if constraint_mask is not None:
            constraint_mask_repeated = constraint_mask.unsqueeze(1).repeat(1, num_samples, 1)
            constraint_mask_repeated = constraint_mask_repeated.reshape(B * num_samples, prediction_length)
            constraint_values_repeated = constraint_values.unsqueeze(1).repeat(1, num_samples, 1, 1)
            constraint_values_repeated = constraint_values_repeated.reshape(
                B * num_samples, prediction_length, feature_dim
            )
        else:
            constraint_mask_repeated = None
            constraint_values_repeated = None

        # Repeat attention mask if provided
        if attention_mask is not None:
            attention_mask_repeated = attention_mask.unsqueeze(1).repeat(1, num_samples, 1)
            attention_mask_repeated = attention_mask_repeated.reshape(B * num_samples, prediction_length)
        else:
            attention_mask_repeated = None

        # Sample from diffusion process
        samples = self.diffusion.p_sample_loop(
            self,
            shape=(B * num_samples, prediction_length, feature_dim),
            clip_denoised=clip_denoised,
            device=device,
            constraint_mask=constraint_mask_repeated,
            attention_mask=attention_mask_repeated,
            constraint_values=constraint_values_repeated,
            enforce_final_constraints=enforce_final_constraints,
            sampling_steps=sampling_steps
        )
        
        # Reshape to separate samples
        samples = samples.reshape(B, num_samples, prediction_length, feature_dim)

        # Translate samples back to the original frame
        samples[..., :2] = samples[..., :2] + origin.unsqueeze(1)

        return samples


if __name__ == '__main__':
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model with transformer encoder architecture
    model = PathDiffusionModel(
        feature_dim=4,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=512,
        device=device
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass with position+direction format
    batch_size = 4
    prediction_length = 181

    # Input format: [x, y, dx, dy] (position + direction)
    target = torch.randn(batch_size, prediction_length, 4).to(device)
    timestep = torch.randint(0, 100, (batch_size,)).to(device)

    # Test with automatically sampled constraints
    clean_pred = model(target, timestep)
    print(f"Clean prediction shape: {clean_pred.shape}")

    # Test with constraint mask (clean imputation and mask concatenation)
    constraint_mask = generate_constraint_mask(torch.ones(batch_size, prediction_length, device=device))
    constraint_mask[:, 0] = 1  # Constrain first position
    clean_pred_constrained = model(target, timestep, constraint_mask)
    print(f"Constrained clean prediction shape: {clean_pred_constrained.shape}")

    # Test sampling with constraints (first/last always required)
    constraint_mask_sample = generate_constraint_mask(torch.ones(batch_size, prediction_length, device=device))
    constraint_mask_sample[:, 0] = 1
    constraint_mask_sample[:, -1] = 1  # Constrain first and last positions
    constraint_values_sample = torch.randn(batch_size, prediction_length, 4).to(device)
    samples = model.sample(
        prediction_length, num_samples=2,
        constraint_mask=constraint_mask_sample,
        constraint_values=constraint_values_sample
    )
    print(f"Samples shape: {samples.shape}")

    # First/last must always be constrained
    try:
        model.sample(prediction_length, num_samples=1)  # Should raise error
    except ValueError as e:
        print(f"Missing constraints error: {e}")

    # Sequences longer than the training max_seq_len are allowed: the sinusoidal
    # positional encoding extends to arbitrary lengths.
    long_samples = model.sample(
        600, num_samples=1,
        constraint_mask=constraint_mask_sample[:1, :1].repeat(1, 600),
        constraint_values=constraint_values_sample[:1, :1].repeat(1, 600, 1)
    )
    print(f"Extended-length samples shape: {long_samples.shape}")

    print("Model test passed!")
