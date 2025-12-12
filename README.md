# Latent Diffusion Models: A PyTorch Implementation

This repository presents a faithful, ground-up implementation of Stable Diffusion, designed to bridge the gap between theoretical papers and production-grade code. It focuses on the mathematical rigor of Denoising Diffusion Probabilistic Models (DDPM) and their application in latent space.

## Theoretical Foundations

The implementation is grounded in the seminal work of Ho et al. in *Denoising Diffusion Probabilistic Models* (2020) and Rombach et al. in *High-Resolution Image Synthesis with Latent Diffusion Models* (2022).

### The Diffusion Process
We model the data distribution $p(x)$ by reversing a gradual noising process. The forward process $q$ is a fixed Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_t$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

### Reverse Process & U-Net
The reverse process $p_\theta$ is learned. We train a U-Net with Cross-Attention mechanisms to approximate the noise $\epsilon_\theta(x_t, t)$ added at each step. This allows us to sample from the data distribution by iteratively denoising a random Gaussian sample.

### Latent Space
Unlike standard DDPMs that operate in pixel space, this implementation operates in the latent space of a pre-trained Variational Autoencoder (VAE). This reduces computational complexity while preserving perceptual quality.

## Architecture

The codebase is modularized to reflect the distinct components of the architecture:

- **`core/diffusion.py`**: Implements the diffusion loop and the U-Net backbone.
- **`core/attention.py`**: Self-Attention and Cross-Attention layers (implementing $Q, K, V$ mechanisms).
- **`core/encoder.py` & `core/decoder.py`**: The VAE components for translating between pixel and latent space.
- **`core/clip.py`**: The text encoder for conditioning the generation process.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Weights Setup

To run the model, you need to download the pre-trained weights and tokenizer files. This implementation is compatible with Stable Diffusion v1.5.

1. Create a `data` folder in the root directory.
2. **Tokenizer Files**:
   - Download `vocab.json` and `merges.txt` from [HuggingFace Tokenizer Files](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer).
   - Save them in `data/`.
3. **Model Checkpoint**:
   - Download `v1-5-pruned-emaonly.ckpt` from [HuggingFace Model Files](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main).
   - Save it in `data/`.

## Usage

### Image Generation
The main entry point for generation is the `inference.ipynb` notebook. It demonstrates the full pipeline:
1.  Tokenizing the prompt.
2.  Encoding the prompt via CLIP.
3.  Iterative denoising in latent space.
4.  Decoding the final latent to pixel space.

```bash
jupyter notebook inference.ipynb
```

### Visualizing the Process
To understand the forward process (adding noise), refer to `noise_process.ipynb`.

```bash
jupyter notebook noise_process.ipynb
```

## References

1.  Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239.
2.  Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR.
3.  Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.

