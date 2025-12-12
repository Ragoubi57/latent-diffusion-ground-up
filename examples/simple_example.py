"""
Simple example demonstrating the diffusion model.
This shows the core concepts of the DDPM paper in a minimal example.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from diffusion import GaussianDiffusion
from unet import UNet


def visualize_forward_process():
    """
    Visualize the forward diffusion process (adding noise).
    """
    print("=== Forward Diffusion Process ===")
    
    # Create a simple image (3x32x32)
    image = torch.randn(1, 3, 32, 32)
    image = (image + 1) / 2  # Normalize to [0, 1] for visualization
    
    # Initialize diffusion
    diffusion = GaussianDiffusion(timesteps=1000, schedule='linear')
    
    # Show how noise is gradually added
    timesteps = [0, 100, 200, 400, 600, 800, 999]
    noisy_images = []
    
    for t in timesteps:
        t_tensor = torch.tensor([t])
        noisy = diffusion.q_sample(image * 2 - 1, t_tensor)  # Back to [-1, 1]
        noisy = (noisy + 1) / 2  # Back to [0, 1]
        noisy = torch.clamp(noisy, 0, 1)
        noisy_images.append(noisy)
    
    # Create visualization
    grid = make_grid(torch.cat(noisy_images, dim=0), nrow=len(timesteps))
    
    plt.figure(figsize=(15, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title('Forward Diffusion Process: Adding Noise Over Time')
    plt.axis('off')
    
    for i, t in enumerate(timesteps):
        plt.text(i * 35 + 16, -5, f't={t}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'forward_diffusion.png'")


def demonstrate_model():
    """
    Demonstrate the U-Net model architecture.
    """
    print("\n=== U-Net Model Architecture ===")
    
    # Initialize model
    model = UNet(
        in_channels=3,
        model_channels=64,  # Smaller for this example
        out_channels=3,
        num_res_blocks=2,
        attention_levels=(1,),
        channel_mult=(1, 2, 4)
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    image_size = 32
    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, t)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Model forward pass successful!")


def demonstrate_loss_calculation():
    """
    Demonstrate the loss calculation from the DDPM paper.
    """
    print("\n=== Loss Calculation ===")
    
    # Initialize model and diffusion
    model = UNet(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=1,
        attention_levels=(),
        channel_mult=(1, 2)
    )
    
    diffusion = GaussianDiffusion(timesteps=1000, schedule='linear')
    
    # Create sample data
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Calculate loss
    loss = diffusion.training_losses(model, x, t)
    
    print(f"Batch size: {batch_size}")
    print(f"Loss value: {loss.item():.4f}")
    print("✓ Loss calculation successful!")


def main():
    """
    Run all demonstrations.
    """
    print("=" * 50)
    print("DDPM Implementation Examples")
    print("Based on: 'Denoising Diffusion Probabilistic Models'")
    print("by Ho et al.")
    print("=" * 50)
    
    # Run demonstrations
    visualize_forward_process()
    demonstrate_model()
    demonstrate_loss_calculation()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
