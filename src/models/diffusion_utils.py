"""
Diffusion Utilities for DDPM (Denoising Diffusion Probabilistic Models)

Implements the core diffusion process:
- Forward process: q(x_t | x_0) - gradually add noise
- Reverse process: p(x_{t-1} | x_t) - gradually denoise

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class DiffusionSchedule:
    """
    Manages the noise schedule for the diffusion process
    """

    def __init__(self,
                 timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear',
                 device: str = 'cpu'):
        """
        Initialize diffusion schedule

        Args:
            timesteps: Number of diffusion steps (T)
            beta_start: Starting noise level
            beta_end: Ending noise level
            schedule_type: 'linear' or 'cosine'
            device: Device to store tensors
        """
        self.timesteps = timesteps
        self.device = device

        # Generate beta schedule
        if schedule_type == 'linear':
            self.betas = self._linear_beta_schedule(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute useful quantities
        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device),
                                               self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Clipping for numerical stability
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _linear_beta_schedule(self, beta_start: float, beta_end: float, timesteps: int) -> torch.Tensor:
        """Linear beta schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_start: Clean images [B, C, H, W]
            t: Timesteps [B]
            noise: Optional pre-sampled noise

        Returns:
            Noisy images x_t [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0)

        Returns:
            posterior_mean, posterior_variance
        """
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)

        return posterior_mean, posterior_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def p_mean_variance(self,
                       model_output: torch.Tensor,
                       x_t: torch.Tensor,
                       t: torch.Tensor,
                       clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for reverse process p(x_{t-1} | x_t)

        Args:
            model_output: Predicted noise from the model
            x_t: Current noisy image
            t: Current timestep
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            mean, variance
        """
        # Predict x_0
        x_recon = self.predict_start_from_noise(x_t, t, model_output)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

        # Compute posterior mean and variance
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_recon, x_t, t)

        return model_mean, posterior_variance

    @torch.no_grad()
    def p_sample(self,
                 model: nn.Module,
                 x_t: torch.Tensor,
                 t: torch.Tensor,
                 condition: Optional[torch.Tensor] = None,
                 clip_denoised: bool = True) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)

        Args:
            model: Denoising model
            x_t: Current noisy image [B, C, H, W]
            t: Current timestep [B]
            condition: Optional conditioning (V_Meta)
            clip_denoised: Whether to clip predicted x_0

        Returns:
            x_{t-1}
        """
        # Get model prediction
        if condition is not None:
            model_output = model(x_t, t, condition)
        else:
            model_output = model(x_t, t)

        # Get mean and variance
        model_mean, model_variance = self.p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised
        )

        # Sample from N(model_mean, model_variance)
        noise = torch.randn_like(x_t)

        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))

        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self,
                     model: nn.Module,
                     shape: Tuple[int, ...],
                     condition: Optional[torch.Tensor] = None,
                     clip_denoised: bool = True,
                     progress: bool = True) -> torch.Tensor:
        """
        Generate samples by running the full reverse diffusion process

        Args:
            model: Denoising model
            shape: Shape of images to generate [B, C, H, W]
            condition: Optional conditioning (V_Meta)
            clip_denoised: Whether to clip predicted x_0
            progress: Whether to show progress bar

        Returns:
            Generated images x_0
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Start from pure noise
        img = torch.randn(shape, device=device)

        # Reverse diffusion
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps)
        else:
            timesteps = reversed(range(0, self.timesteps))

        for i in timesteps:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, condition=condition, clip_denoised=clip_denoised)

        return img

    @torch.no_grad()
    def ddim_sample(self,
                   model: nn.Module,
                   shape: Tuple[int, ...],
                   condition: Optional[torch.Tensor] = None,
                   ddim_timesteps: int = 50,
                   eta: float = 0.0,
                   clip_denoised: bool = True,
                   progress: bool = True) -> torch.Tensor:
        """
        DDIM sampling (faster than DDPM)

        Args:
            model: Denoising model
            shape: Shape of images to generate
            condition: Optional conditioning
            ddim_timesteps: Number of sampling steps (< T for speedup)
            eta: DDIM parameter (0 = deterministic, 1 = DDPM)
            clip_denoised: Whether to clip
            progress: Show progress bar

        Returns:
            Generated images
        """
        device = next(model.parameters()).device
        batch_size = shape[0]

        # Create DDIM timestep sequence
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))

        # Start from pure noise
        img = torch.randn(shape, device=device)

        # Reverse diffusion with DDIM
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(reversed(ddim_timestep_seq), desc='DDIM Sampling', total=len(ddim_timestep_seq))
        else:
            timesteps = reversed(ddim_timestep_seq)

        for i in timesteps:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Predict noise
            if condition is not None:
                predicted_noise = model(img, t, condition)
            else:
                predicted_noise = model(img, t)

            # Predict x_0
            x_0_pred = self.predict_start_from_noise(img, t, predicted_noise)

            if clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Get alpha values
            alpha_t = self._extract(self.alphas_cumprod, t, img.shape)

            if i > 0:
                t_prev = torch.full((batch_size,), ddim_timestep_seq[ddim_timestep_seq < i][-1],
                                   device=device, dtype=torch.long)
                alpha_t_prev = self._extract(self.alphas_cumprod, t_prev, img.shape)
            else:
                alpha_t_prev = torch.ones_like(alpha_t)

            # Compute variance
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

            # Compute mean
            mean = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * predicted_noise

            # Add noise
            noise = torch.randn_like(img)
            img = mean + sigma_t * noise if i > 0 else mean

        return img

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from a at timesteps t and reshape to broadcast with x_shape

        Args:
            a: Source tensor [T]
            t: Timesteps [B]
            x_shape: Target shape [B, C, H, W]

        Returns:
            Extracted values reshaped to [B, 1, 1, 1]
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def test_diffusion_schedule():
    """Test the diffusion schedule"""
    print("=" * 60)
    print("Testing Diffusion Schedule")
    print("=" * 60)

    # Create schedule
    schedule = DiffusionSchedule(timesteps=1000, device='cpu')

    print(f"\nTimesteps: {schedule.timesteps}")
    print(f"Beta range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"Alpha range: [{schedule.alphas[0]:.6f}, {schedule.alphas[-1]:.6f}]")
    print(f"Alpha_cumprod range: [{schedule.alphas_cumprod[0]:.6f}, {schedule.alphas_cumprod[-1]:.6f}]")

    # Test forward process
    print("\n" + "-" * 60)
    print("Testing forward process (adding noise)")
    print("-" * 60)

    x_0 = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    t = torch.tensor([0, 250, 500, 999])  # Different timesteps

    x_t = schedule.q_sample(x_0, t)

    print(f"x_0 shape: {x_0.shape}")
    print(f"x_t shape: {x_t.shape}")
    print(f"x_0 mean/std: {x_0.mean():.4f} / {x_0.std():.4f}")
    print(f"x_t mean/std: {x_t.mean():.4f} / {x_t.std():.4f}")

    # Test noise prediction
    print("\n" + "-" * 60)
    print("Testing noise prediction")
    print("-" * 60)

    noise = torch.randn_like(x_0)
    x_t = schedule.q_sample(x_0, t, noise)

    # Predict x_0 from x_t and noise
    x_0_pred = schedule.predict_start_from_noise(x_t, t, noise)

    print(f"x_0 reconstruction error: {(x_0 - x_0_pred).abs().mean():.6f}")

    print("\n" + "=" * 60)
    print("Diffusion schedule test passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_diffusion_schedule()
