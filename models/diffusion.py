"""
Denoising Diffusion Probabilistic Model (DDPM) Implementation
For Medical Image Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class GaussianDiffusion:
    """
    DDPM Gaussian Diffusion Process
    Implements forward noising and reverse denoising
    """
    
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='linear',
        device='cuda'
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Create beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'quadratic':
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        elif beta_schedule == 'cosine':
            betas = self.cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        # Precompute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(device)
        
        # Calculations for forward diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        ).to(device)
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        ).to(device)
        
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule from Improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_0 to get x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (
            x_t - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t
    
    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """Predict mean and variance for reverse process"""
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Predict x_0
        x_recon = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        # Get posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_recon, x_t, t
        )
        
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x_t, t, clip_denoised=True):
        """Sample x_{t-1} from x_t"""
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        """Generate samples from noise"""
        device = self.device
        batch_size = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        for t in tqdm(
            reversed(range(0, self.timesteps)),
            desc='Sampling',
            total=self.timesteps
        ):
            img = self.p_sample(
                img,
                torch.full((batch_size,), t, device=device, dtype=torch.long)
            )
            imgs.append(img)
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=1)
        else:
            return img
    
    @torch.no_grad()
    def ddim_sample(self, shape, ddim_timesteps=50, eta=0.0):
        """
        DDIM sampling for faster inference
        eta=0 gives deterministic sampling
        """
        device = self.device
        batch_size = shape[0]
        
        # Create subsequence of timesteps
        step = self.timesteps // ddim_timesteps
        timesteps = torch.arange(0, self.timesteps, step, device=device).long()
        timesteps = torch.cat([timesteps, torch.tensor([self.timesteps - 1], device=device)])
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(
            reversed(range(len(timesteps))),
            desc='DDIM Sampling',
            total=len(timesteps)
        ):
            t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(img, t)
            
            # Get alpha values
            alpha = self.extract(self.alphas_cumprod, t, img.shape)
            
            if i > 0:
                alpha_prev = self.extract(
                    self.alphas_cumprod,
                    torch.full((batch_size,), timesteps[i - 1], device=device, dtype=torch.long),
                    img.shape
                )
            else:
                alpha_prev = torch.ones_like(alpha)
            
            # Predict x_0
            pred_x0 = (img - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            
            noise = torch.randn_like(img) if i > 0 else 0
            
            # DDIM update
            img = (
                torch.sqrt(alpha_prev) * pred_x0 +
                torch.sqrt(1 - alpha_prev - sigma ** 2) * predicted_noise +
                sigma * noise
            )
        
        return img
    
    def training_losses(self, x_start, t, noise=None):
        """
        Compute training loss
        L_simple = E[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


class DiffusionTrainer:
    """Trainer for diffusion model"""
    
    def __init__(
        self,
        diffusion,
        optimizer,
        device='cuda',
        grad_clip=1.0
    ):
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        
    def train_step(self, batch):
        """Single training step"""
        self.diffusion.model.train()
        
        # Move to device
        x = batch.to(self.device)
        
        # Sample random timesteps
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.diffusion.timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        # Compute loss
        loss = self.diffusion.training_losses(x, t)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.diffusion.model.parameters(),
                self.grad_clip
            )
        
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate_step(self, batch):
        """Single validation step"""
        self.diffusion.model.eval()
        
        x = batch.to(self.device)
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.diffusion.timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        loss = self.diffusion.training_losses(x, t)
        
        return loss.item()


def test_diffusion():
    """Test diffusion model"""
    from unet3d import UNet3D
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4),
    ).to(device)
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=1000,
        device=device
    )
    
    # Test forward diffusion
    x = torch.randn(1, 1, 32, 32, 32).to(device)
    t = torch.tensor([500]).to(device)
    
    x_noisy = diffusion.q_sample(x, t)
    print(f"Original shape: {x.shape}")
    print(f"Noisy shape: {x_noisy.shape}")
    
    # Test training loss
    loss = diffusion.training_losses(x, t)
    print(f"Training loss: {loss.item():.4f}")
    
    return diffusion


if __name__ == "__main__":
    test_diffusion()
