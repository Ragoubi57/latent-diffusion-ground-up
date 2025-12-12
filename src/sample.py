"""
Sampling script for the diffusion model.
Generates images from pure noise using the trained model.
"""

import torch
import torchvision
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from unet import UNet
from diffusion import GaussianDiffusion


def denormalize(images):
    """
    Denormalize images from [-1, 1] to [0, 1].
    """
    return (images + 1) / 2


def sample_images(
    model,
    diffusion,
    num_samples,
    image_size,
    device,
    save_path='samples.png'
):
    """
    Generate samples from the diffusion model.
    
    Args:
        model: Trained U-Net model
        diffusion: GaussianDiffusion instance
        num_samples: Number of samples to generate
        image_size: Size of the images
        device: Device to run on
        save_path: Path to save the generated images
    """
    model.eval()
    
    with torch.no_grad():
        # Generate samples
        shape = (num_samples, 3, image_size, image_size)
        samples = diffusion.p_sample_loop(model, shape, device)
        
        # Denormalize and save
        samples = denormalize(samples)
        samples = torch.clamp(samples, 0, 1)
        
        # Create grid
        grid = make_grid(samples, nrow=int(num_samples ** 0.5), padding=2)
        save_image(grid, save_path)
        print(f'Samples saved to {save_path}')
        
        return samples


def visualize_diffusion_process(
    model,
    diffusion,
    device,
    image_size=32,
    save_path='diffusion_process.png'
):
    """
    Visualize the reverse diffusion process step by step.
    
    Args:
        model: Trained U-Net model
        diffusion: GaussianDiffusion instance
        device: Device to run on
        image_size: Size of the images
        save_path: Path to save the visualization
    """
    model.eval()
    
    with torch.no_grad():
        # Start from pure noise
        shape = (1, 3, image_size, image_size)
        img = torch.randn(shape, device=device)
        
        # Store intermediate steps
        steps_to_save = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
        intermediate_images = []
        
        for i in tqdm(reversed(range(0, diffusion.timesteps)), desc='Sampling'):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = diffusion.p_sample(model, img, t, i)
            
            if i in steps_to_save:
                intermediate_images.append(img.cpu().clone())
        
        # Denormalize and create visualization
        intermediate_images = [denormalize(img) for img in intermediate_images]
        intermediate_images = [torch.clamp(img, 0, 1) for img in intermediate_images]
        
        # Create grid
        grid = make_grid(torch.cat(intermediate_images, dim=0), nrow=len(steps_to_save), padding=2)
        save_image(grid, save_path)
        print(f'Diffusion process visualization saved to {save_path}')


def main():
    """
    Main sampling function.
    """
    # Configuration
    num_samples = 16
    image_size = 32
    timesteps = 1000
    model_path = 'diffusion_model_final.pt'  # Path to trained model
    output_dir = 'samples'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model and diffusion
    model = UNet(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_levels=(1, 2),
        channel_mult=(1, 2, 4, 8)
    ).to(device)
    
    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule='linear'
    )
    
    # Load trained model
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f'Warning: Model file {model_path} not found. Using untrained model.')
    
    # Generate samples
    print(f'Generating {num_samples} samples...')
    samples = sample_images(
        model,
        diffusion,
        num_samples,
        image_size,
        device,
        save_path=os.path.join(output_dir, 'samples.png')
    )
    
    # Visualize diffusion process
    print('Visualizing diffusion process...')
    visualize_diffusion_process(
        model,
        diffusion,
        device,
        image_size,
        save_path=os.path.join(output_dir, 'diffusion_process.png')
    )
    
    print('Done!')


if __name__ == '__main__':
    main()
