"""
Training script for the diffusion model.
Implements the training procedure from the DDPM paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from unet import UNet
from diffusion import GaussianDiffusion


def train(
    model,
    diffusion,
    dataloader,
    optimizer,
    device,
    num_epochs=10,
    save_dir='checkpoints'
):
    """
    Train the diffusion model.
    
    Args:
        model: U-Net model
        diffusion: GaussianDiffusion instance
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of training epochs
        save_dir: Directory to save checkpoints
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Calculate loss
            loss = diffusion.training_losses(model, images, t)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')


def main():
    """
    Main training function.
    """
    # Hyperparameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 2e-4
    image_size = 32
    timesteps = 1000
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data preparation
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    # Using CIFAR-10 as example dataset
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
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
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Training on {len(dataset)} images')
    
    # Train
    train(model, diffusion, dataloader, optimizer, device, num_epochs)
    
    # Save final model
    torch.save(model.state_dict(), 'diffusion_model_final.pt')
    print('Training complete! Model saved.')


if __name__ == '__main__':
    main()
