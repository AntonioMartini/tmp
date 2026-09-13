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
3. Training: Train a neural network to predict the noise added at each timestep
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
- Noise-based Constraints: Constraints are enforced by zeroing noise at constrained positions

Architecture Details:
--------------------
- Input: Noisy trajectory (B, T, 4) where 4 = [x, y, dx, dy] (position + direction)
- Output: Predicted noise (B, T, 4)
- Fixed Maximum Sequence Length: Configurable (default: 512)
- Hidden Dimension: Configurable (default: 256)
- Number of Layers: Configurable (default: 6)
- Attention Heads: Configurable (default: 8)
- Diffusion Timesteps: Configurable (default: 100)

The model uses a transformer encoder architecture with sinusoidal positional embeddings for position-aware processing.
Timestep embedding is broadcast and added to each position in the sequence (Kimodo-style).
Constraints are enforced by zeroing noise at constrained positions during the diffusion process.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    """Non-learned sinusoidal positional encoding."""

    def __init__(
        self,
        d_model: int,
        dropout: Optional[float] = 0.1,
        max_len: Optional[int] = 5000,
    ):
        """
        Args:
            d_model (int): input dim
            dropout (Optional[float] = 0.1): dropout probability on output
            max_len (Optional[int] = 5000): maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Note: have to replace torch.exp() and math.log() with torch.pow()
        # due to MKL exp() and ln() throws floating point exceptions on certain CPUs
        div_term = torch.pow(10000.0, -torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input sequence.

        Args:
            x (torch.Tensor): [B, T, D] input motion sequence

        Returns:
            torch.Tensor: [B, T, D] input motion with PE added to it (and optionally dropout)
        """
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
    The model is unconditional: it sees only the noisy target, timestep and positional
    embeddings. Conditioning happens only through constraints, which are directly
    enforced in the noise during the diffusion process (Kimodo-style).
    Timestep embedding is added to each position (broadcast across sequence).
    
    Based on the Kimodo transformer denoiser architecture adapted for path prediction.
    
    Input format: position + direction (x, y, dx, dy) where dx, dy represent direction.
    """
    
    def __init__(
        self,
        feature_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        pe_dropout: float = 0.1,
        max_seq_len: int = 512,
        num_timesteps: int = 100,
        norm_first: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_timesteps = num_timesteps
        
        # Positional encoding
        self.sequence_pos_encoder = PositionalEncoding(hidden_dim, pe_dropout, max_seq_len)
        
        # Timestep embedder (separate PE table since timesteps can exceed max_seq_len)
        self.timestep_pos_encoder = PositionalEncoding(hidden_dim, 0.0, num_timesteps)
        self.embed_timestep = TimestepEmbedder(hidden_dim, self.timestep_pos_encoder)
        
        # Input projection for position+direction
        self.input_linear = nn.Linear(feature_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, feature_dim)
        
        # Transformer encoder with multiple layers
        trans_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noisy_path: (B, T_pred, 4) noisy trajectory to denoise [x, y, dx, dy]
            timestep: (B,) diffusion timestep
            constraint_mask: Optional mask (B, T_pred) where 1 indicates constrained positions
                           (constraints are enforced in the noise during diffusion, not here)
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
                           (prevents attention to padded positions)
            
        Returns:
            (B, T_pred, 4) predicted noise
        """
        B, T_pred, _ = noisy_path.shape
        
        # Project noisy path
        noisy_proj = self.input_linear(noisy_path)  # (B, T_pred, hidden_dim)
        
        # Encode timestep and broadcast across sequence (Kimodo-style)
        emb_time = self.embed_timestep(timestep)  # (B, 1, hidden_dim)
        emb_time = emb_time.expand(-1, T_pred, -1)  # (B, T_pred, hidden_dim)
        
        # Add timestep embedding to noisy path (broadcast at each position)
        xseq = noisy_proj + emb_time  # (B, T_pred, hidden_dim)
        
        # Add positional encoding
        xseq = self.sequence_pos_encoder(xseq)
        
        # Apply attention mask if provided (for padding)
        if attention_mask is not None:
            # Convert attention mask to key padding mask format for transformer
            # attention_mask: (B, T_pred) where 1 = valid, 0 = padding
            # key_padding_mask: (B, T_pred) where True = padding (ignore), False = valid
            key_padding_mask = (attention_mask == 0).bool()
        else:
            key_padding_mask = None
        
        # Input to the transformer
        if isinstance(self.seqTransEncoder, nn.TransformerEncoder):
            assert not self.seqTransEncoder.use_nested_tensor, "Flash attention should be disabled due to bug!"

        output = self.seqTransEncoder(xseq, src_key_padding_mask=key_padding_mask)
        
        # Output projection
        noise_pred = self.output_linear(output)
        
        return noise_pred


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
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and variance of the reverse diffusion process
        p(x_{t-1} | x_t) using the model to predict x_0.
        Note: constraint_mask is only used in p_sample, not passed to model.
        """
        # Predict noise (call the denoiser directly; forward() would add more noise to x_t)
        noise_pred = model.denoiser(x_t, t, attention_mask=attention_mask)
        
        # Predict x_0 from noise
        sqrt_recip_alphas_cumprod_t = self._extract(
            1.0 / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            torch.sqrt(1.0 / self.alphas_cumprod - 1), t, x_t.shape
        )
        
        pred_x_start = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise_pred
        
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -3.0, 3.0)
        
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
        constraint_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from the reverse diffusion process p(x_{t-1} | x_t).
        """
        posterior_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            model, x_t, t, clip_denoised, constraint_mask, attention_mask
        )

        noise = torch.randn_like(x_t)

        # Enforce constraints in the noise (Kimodo-style: zero noise at constrained positions)
        if constraint_mask is not None:
            noise = noise * (1 - constraint_mask.unsqueeze(-1))

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        sample = posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

        # Overwrite constrained positions with the forward-noised constraint values
        # (Kimodo-style: constraints are directly overwritten in the noisy input each
        # step, so the denoiser sees the clean constraint signal at the next step)
        if constraint_mask is not None and constraint_values is not None:
            t_prev = (t - 1).clamp(min=0)
            mask = constraint_mask.unsqueeze(-1)
            constrained = self.q_sample(constraint_values, t_prev)
            sample = sample * (1 - mask) + constrained * mask

        return sample
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,
        clip_denoised: bool = True,
        device: str = 'cuda',
        constraint_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples by iteratively denoising.
        """
        B = shape[0]
        img = torch.randn(shape, device=device)

        # Initialize constrained positions with the noised constraint values
        if constraint_mask is not None and constraint_values is not None:
            t_last = torch.full((B,), self.num_timesteps - 1, device=device, dtype=torch.long)
            mask = constraint_mask.unsqueeze(-1)
            img = img * (1 - mask) + self.q_sample(constraint_values, t_last) * mask

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                model, img, t, clip_denoised, constraint_mask,
                attention_mask, constraint_values
            )

        # Set constrained positions to the exact constraint values at t = 0
        if constraint_mask is not None and constraint_values is not None:
            mask = constraint_mask.unsqueeze(-1)
            img = img * (1 - mask) + constraint_values * mask

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
    
    Constraints are enforced by zeroing noise at constrained positions during the diffusion process (Kimodo-style).
    Timestep embedding is broadcast and added to each position in the sequence.
    """
    
    def __init__(
        self,
        feature_dim: int = 4,
        num_timesteps: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        pe_dropout: float = 0.1,
        beta_schedule: str = 'linear',
        max_seq_len: int = 512,
        norm_first: bool = False,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len
        self.denoiser = PathDenoiser(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
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
        Forward pass for training. The model is unconditional; conditioning
        happens only through constraints enforced in the diffusion noise.
        Args:
            target: (B, T_pred, 4) ground truth trajectory [x, y, dx, dy]
            timestep: (B,) diffusion timestep
            constraint_mask: Optional mask (B, T_pred) for constrained positions
                           (constraints are enforced in the noise during diffusion)
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
        Returns:
            (B, T_pred, 4) predicted noise
        """
        # Add noise to target
        noise = torch.randn_like(target)

        # Enforce constraints in the noise (Kimodo-style: zero noise at constrained positions)
        if constraint_mask is not None:
            noise = noise * (1 - constraint_mask.unsqueeze(-1))

        noisy_target = self.diffusion.q_sample(target, timestep, noise)

        # Predict noise using the denoiser (denoiser no longer handles constraints)
        noise_pred = self.denoiser(noisy_target, timestep, attention_mask=attention_mask)

        return noise_pred
    
    @torch.no_grad()
    def sample(
        self,
        prediction_length: int,
        num_samples: int = 1,
        clip_denoised: bool = True,
        constraint_mask: Optional[torch.Tensor] = None,
        constraint_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate path predictions unconditionally; control happens only through
        constraint_mask/constraint_values.

        The model is trained with the first and last valid frame always
        constrained, so both must be constrained here as well.
        Args:
            prediction_length: Number of frames to predict
            num_samples: Number of samples to generate per batch element
            clip_denoised: Whether to clip denoised predictions
            constraint_mask: (B, T_pred) mask with 1 at constrained positions.
                           Index 0 and the last valid index (per attention_mask,
                           or T_pred-1) must always be constrained.
            constraint_values: (B, T_pred, 4) values at constrained positions, in
                           normalized coordinates. The constrained first point
                           anchors the generated trajectory (canonicalization).
            attention_mask: Optional mask (B, T_pred) where 1 indicates valid positions, 0 indicates padding
            device: Device to run on
        Returns:
            (B, num_samples, T_pred, 4) generated trajectories
        """
        self.eval()
        feature_dim = self.feature_dim

        if constraint_mask is None or constraint_values is None:
            raise ValueError(
                "constraint_mask and constraint_values are required: the model is "
                "trained with the first and last frame always constrained"
            )
        B = constraint_mask.shape[0]

        if prediction_length > self.max_seq_len:
            raise ValueError(f"Prediction length {prediction_length} exceeds max_seq_len {self.max_seq_len}")

        # Verify the first and last valid frame are constrained
        if not (constraint_mask[:, 0] == 1).all():
            raise ValueError("constraint_mask must constrain index 0 (first frame)")
        if attention_mask is not None:
            last_idx = (attention_mask.sum(dim=1).long() - 1).clamp(min=0)
        else:
            last_idx = torch.full((B,), prediction_length - 1,
                                  dtype=torch.long, device=constraint_mask.device)
        if not (constraint_mask[torch.arange(B, device=constraint_mask.device), last_idx] == 1).all():
            raise ValueError("constraint_mask must constrain the last valid frame")

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
            constraint_values=constraint_values_repeated
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
    prediction_length = 60

    # Input format: [x, y, dx, dy] (position + direction)
    target = torch.randn(batch_size, prediction_length, 4).to(device)
    timestep = torch.randint(0, 100, (batch_size,)).to(device)

    # Test without constraints
    noise_pred = model(target, timestep)
    print(f"Noise prediction shape: {noise_pred.shape}")

    # Test with constraint mask (constraints enforced in noise during diffusion)
    constraint_mask = torch.zeros(batch_size, prediction_length).to(device)
    constraint_mask[:, :10] = 1  # Constrain first 10 positions
    noise_pred_constrained = model(target, timestep, constraint_mask)
    print(f"Constrained noise prediction shape: {noise_pred_constrained.shape}")

    # Test sampling with constraints (first/last always required)
    constraint_mask_sample = torch.zeros(batch_size, prediction_length).to(device)
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

    # Test sequence length constraint
    try:
        model.sample(600, num_samples=1,
                     constraint_mask=constraint_mask_sample[:1, :1].repeat(1, 600),
                     constraint_values=constraint_values_sample[:1, :1].repeat(1, 600, 1))  # 600 > 512
    except ValueError as e:
        print(f"Sequence length constraint working: {e}")
    
    print("Model test passed!")
