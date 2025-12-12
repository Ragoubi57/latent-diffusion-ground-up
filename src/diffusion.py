"""
Diffusion process implementation based on the DDPM paper:
"Denoising Diffusion Probabilistic Models" by Ho et al.

This module implements the forward and reverse diffusion processes.
"""

import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """
    Implements the Gaussian diffusion process from the DDPM paper.
    
    The forward process gradually adds noise to the data:
    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
    
    The reverse process learns to denoise:
    p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), Sigma_theta(x_t, t))
    """
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
        """
        Args:
            timesteps: Number of diffusion steps
            beta_start: Starting value of beta
            beta_end: Ending value of beta
            schedule: Schedule type ('linear' or 'cosine')
        """
        self.timesteps = timesteps
        
        # Define beta schedule
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute useful values for diffusion
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Calculations for reverse process mean
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Sample from the diffusion process at timestep t.
        
        Args:
            x_start: Initial data (x_0)
            t: Timestep
            noise: Optional pre-generated noise
            
        Returns:
            Noised data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, t_index):
        """
        Reverse diffusion process: p_theta(x_{t-1} | x_t)
        Sample from the reverse process at timestep t.
        
        Args:
            model: Neural network that predicts noise
            x_t: Data at timestep t
            t: Current timestep
            t_index: Index of current timestep (for noise addition)
            
        Returns:
            Predicted x_{t-1}
        """
        # Predict noise using the model
        predicted_noise = model(x_t, t)
        
        # Calculate mean of reverse process
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        # Equation 11 in the DDPM paper
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def p_sample_loop(self, model, shape, device):
        """
        Complete sampling loop: Generate samples from noise.
        
        Args:
            model: Neural network that predicts noise
            shape: Shape of the samples to generate
            device: Device to run on
            
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
        
        return img
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from array at given timesteps and reshape for broadcasting.
        
        Args:
            arr: Array to extract from
            timesteps: Timesteps to extract
            broadcast_shape: Shape to broadcast to
            
        Returns:
            Extracted and reshaped values
        """
        batch_size = timesteps.shape[0]
        out = arr.to(timesteps.device)[timesteps]
        return out.reshape(batch_size, *((1,) * (len(broadcast_shape) - 1)))
    
    def training_losses(self, model, x_start, t):
        """
        Calculate training loss (simplified from DDPM paper).
        
        The loss is the MSE between predicted noise and actual noise:
        L = E[||epsilon - epsilon_theta(x_t, t)||^2]
        
        Args:
            model: Neural network that predicts noise
            x_start: Original data
            t: Timesteps
            
        Returns:
            Loss value
        """
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Get noisy image at timestep t
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Predict the noise
        predicted_noise = model(x_t, t)
        
        # Calculate MSE loss (simplified objective from DDPM)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
