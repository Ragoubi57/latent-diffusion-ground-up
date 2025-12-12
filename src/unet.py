"""
U-Net architecture for the diffusion model.
Based on the architecture described in the DDPM paper.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time steps.
    Similar to the positional encoding in Transformers.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding and group normalization.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for spatial features.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Normalize and compute Q, K, V
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        
        # Attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)
        out = self.proj(out)
        
        return x + out


class DownBlock(nn.Module):
    """
    Downsampling block with residual connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, 
                 downsample=True, attention=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels, 
                out_channels, 
                time_emb_dim
            ) for i in range(num_layers)
        ])
        
        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if attention else nn.Identity()
            for _ in range(num_layers)
        ])
        
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                       stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x, t):
        for layer, attn in zip(self.layers, self.attention):
            x = layer(x, t)
            x = attn(x)
        
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with residual connections and skip connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2, 
                 upsample=True, attention=False):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim
            ) for i in range(num_layers)
        ])
        
        self.attention = nn.ModuleList([
            AttentionBlock(out_channels) if attention else nn.Identity()
            for _ in range(num_layers)
        ])
        
        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 
                                              kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()

    def forward(self, x, t):
        for layer, attn in zip(self.layers, self.attention):
            x = layer(x, t)
            x = attn(x)
        
        x = self.upsample(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        model_channels: Base channel count
        out_channels: Number of output channels (same as input)
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels to apply attention
        channel_mult: Channel multiplier for each level
        dropout: Dropout probability
    """
    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_levels=(1, 2),
        channel_mult=(1, 2, 4, 8),
        dropout=0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList([])
        channels = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        out_ch,
                        time_emb_dim,
                        num_layers=1,
                        downsample=False,
                        attention=(level in attention_levels)
                    )
                )
                now_channels = out_ch
                channels.append(now_channels)
            
            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    DownBlock(
                        now_channels,
                        now_channels,
                        time_emb_dim,
                        num_layers=1,
                        downsample=True,
                        attention=False
                    )
                )
                channels.append(now_channels)
        
        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout),
            AttentionBlock(now_channels),
            ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        ])
        
        # Upsampling path
        self.up_blocks = nn.ModuleList([])
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                self.up_blocks.append(
                    UpBlock(
                        now_channels + channels.pop(),
                        out_ch,
                        time_emb_dim,
                        num_layers=1,
                        upsample=False,
                        attention=(level in attention_levels)
                    )
                )
                now_channels = out_ch
            
            # Upsample (except at the first level)
            if level != 0:
                self.up_blocks.append(
                    UpBlock(
                        now_channels,
                        now_channels,
                        time_emb_dim,
                        num_layers=1,
                        upsample=True,
                        attention=False
                    )
                )
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, timesteps):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            timesteps: Timestep tensor (B,)
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t = self.time_mlp(timesteps)
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Downsampling with skip connections
        skips = [x]
        for block in self.down_blocks:
            x = block(x, t)
            skips.append(x)
        
        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t)
            else:
                x = layer(x)
        
        # Upsampling with skip connections
        for block in self.up_blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, t)
        
        # Output
        x = self.conv_out(x)
        
        return x
