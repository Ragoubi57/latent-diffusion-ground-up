# Latent Diffusion from Ground Up

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) built from scratch, following the seminal paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) by Jonathan Ho, Ajay Jain, and Pieter Abbeel.

## Overview

This project implements a complete diffusion model pipeline including:
- **Forward diffusion process**: Gradually adding Gaussian noise to data
- **Reverse diffusion process**: Learning to denoise and generate samples
- **U-Net architecture**: Deep neural network with attention mechanisms
- **Training and sampling**: Complete training loop and image generation

## Key Features

- ✨ **From Scratch Implementation**: Built entirely with PyTorch without relying on external diffusion libraries
- 📚 **Well-Documented**: Clear code with extensive comments explaining the DDPM paper concepts
- 🎯 **DDPM Paper Faithful**: Implements the exact formulations from the original paper
- 🔧 **Modular Design**: Easy to understand, modify, and extend
- 🚀 **Ready to Train**: Includes training scripts and example usage

## Installation

```bash
# Clone the repository
git clone https://github.com/Ragoubi57/latent-diffusion-ground-up.git
cd latent-diffusion-ground-up

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
latent-diffusion-ground-up/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── diffusion.py          # Forward and reverse diffusion processes
│   ├── unet.py               # U-Net architecture with attention
│   ├── train.py              # Training script
│   └── sample.py             # Sampling and inference script
├── examples/
│   └── simple_example.py     # Simple demonstrations
├── requirements.txt          # Project dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Core Components

### 1. Diffusion Process (`src/diffusion.py`)

Implements the Gaussian diffusion process:

- **Forward process**: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)`
- **Reverse process**: `p_θ(x_{t-1} | x_t)` learned by the neural network
- **Training objective**: Simplified loss `L = ||ε - ε_θ(x_t, t)||²`

Key classes:
- `GaussianDiffusion`: Main diffusion process handler with both linear and cosine schedules

### 2. U-Net Architecture (`src/unet.py`)

A U-Net with the following components:

- **Time embeddings**: Sinusoidal position embeddings for timestep conditioning
- **Residual blocks**: With group normalization and time injection
- **Attention blocks**: Multi-head self-attention for spatial features
- **Skip connections**: Connecting encoder and decoder paths

### 3. Training (`src/train.py`)

Training script that:
- Loads and preprocesses data (CIFAR-10 by default)
- Implements the training loop from the DDPM paper
- Saves checkpoints periodically
- Tracks loss and progress

### 4. Sampling (`src/sample.py`)

Sampling script that:
- Generates images from pure noise
- Visualizes the reverse diffusion process
- Saves generated samples

## Usage

### Quick Start

Run a simple example to understand the concepts:

```bash
python examples/simple_example.py
```

This will:
- Visualize the forward diffusion process
- Show the U-Net architecture
- Demonstrate loss calculation

### Training

Train the model on CIFAR-10 (or your own dataset):

```bash
cd src
python train.py
```

Configuration options in `train.py`:
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: Learning rate (default: 2e-4)
- `image_size`: Size of images (default: 32)
- `timesteps`: Number of diffusion steps (default: 1000)

### Generating Samples

After training, generate samples:

```bash
cd src
python sample.py
```

This will:
- Load the trained model
- Generate sample images
- Visualize the step-by-step denoising process

### Using as a Library

```python
from src.diffusion import GaussianDiffusion
from src.unet import UNet
import torch

# Initialize model and diffusion
model = UNet(in_channels=3, model_channels=128, out_channels=3)
diffusion = GaussianDiffusion(timesteps=1000, schedule='linear')

# Training: calculate loss
x = torch.randn(4, 3, 32, 32)  # Batch of images
t = torch.randint(0, 1000, (4,))  # Random timesteps
loss = diffusion.training_losses(model, x, t)

# Sampling: generate images
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
samples = diffusion.p_sample_loop(model, shape=(16, 3, 32, 32), device=device)
```

## Theory: DDPM Paper Summary

### Forward Diffusion

The forward process gradually adds Gaussian noise to the data over T timesteps:

```
x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε,  where ε ~ N(0, I)
```

### Reverse Diffusion

The model learns to reverse this process:

```
x_{t-1} = 1/√(α_t) (x_t - (β_t/√(1-ᾱ_t)) ε_θ(x_t, t)) + σ_t z
```

where `ε_θ` is the neural network that predicts the noise.

### Training Objective

The simplified training objective:

```
L = E[||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
```

## Implementation Details

- **Beta schedule**: Linear schedule from 0.0001 to 0.02, or cosine schedule
- **Timesteps**: 1000 diffusion steps (T=1000)
- **Architecture**: U-Net with residual blocks and attention
- **Normalization**: Group normalization (8 groups)
- **Optimizer**: Adam with learning rate 2e-4
- **Data**: Normalized to [-1, 1] range

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- numpy
- Pillow
- tqdm
- matplotlib

## References

1. **Original Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
   - Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)

2. **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
   - Alex Nichol, Prafulla Dhariwal (2021)

## License

MIT License - Feel free to use this code for learning and research purposes.

## Acknowledgments

This implementation is built for educational purposes, following the groundbreaking work by Ho et al. on Denoising Diffusion Probabilistic Models.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

## Author

Built with ❤️ to understand diffusion models from the ground up.