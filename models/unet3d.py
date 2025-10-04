"""
3D U-Net Architecture for Medical Image Diffusion
Implements a 3D U-Net with attention mechanisms for DDPM-based enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock3D(nn.Module):
    """3D Residual Block with GroupNorm and SiLU activation"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
    def forward(self, x, t_emb):
        # First convolution
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None, None]
        
        # Second convolution
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class Attention3D(nn.Module):
    """3D Self-Attention mechanism"""
    
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, d * h * w)
        
        # Attention
        q = q.permute(0, 1, 3, 2)  # b, heads, dhw, c
        attention = torch.softmax(q @ k / math.sqrt(c // self.num_heads), dim=-1)
        out = attention @ v.permute(0, 1, 3, 2)  # b, heads, dhw, c
        
        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(b, c, d, h, w)
        out = self.proj(out)
        
        return out + x


class Downsample3D(nn.Module):
    """3D Downsampling layer"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D Upsampling layer"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for Medical Image Enhancement with Diffusion
    
    Architecture:
    - Encoder: 4 levels with ResBlocks and Attention
    - Bottleneck: ResBlocks with Attention
    - Decoder: 4 levels with ResBlocks and Skip Connections
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        dropout=0.1,
        attention_resolutions=(2, 4)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        channels = [base_channels * m for m in channel_mults]
        in_ch = base_channels
        
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
                
                # Add attention at specified resolutions
                if level in attention_resolutions:
                    blocks.append(Attention3D(out_ch))
                    
            self.encoder_blocks.append(blocks)
            
            # Downsample (except last level)
            if level < len(channels) - 1:
                self.downsample_blocks.append(Downsample3D(out_ch))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock3D(channels[-1], channels[-1], time_emb_dim, dropout),
            Attention3D(channels[-1]),
            ResBlock3D(channels[-1], channels[-1], time_emb_dim, dropout)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        
        for level, out_ch in enumerate(reversed_channels):
            blocks = nn.ModuleList()
            
            # Account for skip connections (concatenation)
            in_ch = out_ch if level == 0 else reversed_channels[level - 1]
            skip_ch = out_ch
            
            for i in range(num_res_blocks + 1):
                # First block receives skip connection
                res_in_ch = in_ch + skip_ch if i == 0 else out_ch
                blocks.append(ResBlock3D(res_in_ch, out_ch, time_emb_dim, dropout))
                
                # Add attention at specified resolutions
                if (len(channels) - 1 - level) in attention_resolutions:
                    blocks.append(Attention3D(out_ch))
                    
            self.decoder_blocks.append(blocks)
            
            # Upsample (except last level)
            if level < len(reversed_channels) - 1:
                self.upsample_blocks.append(Upsample3D(out_ch))
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t):
        """
        Args:
            x: Input tensor [B, C, D, H, W]
            t: Timestep tensor [B]
            
        Returns:
            Output tensor [B, C, D, H, W]
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Encoder with skip connections
        skip_connections = []
        
        for blocks, downsample in zip(self.encoder_blocks[:-1], self.downsample_blocks):
            for block in blocks:
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                else:  # Attention
                    h = block(h)
            skip_connections.append(h)
            h = downsample(h)
        
        # Last encoder level (no downsampling)
        for block in self.encoder_blocks[-1]:
            if isinstance(block, ResBlock3D):
                h = block(h, t_emb)
            else:
                h = block(h)
        skip_connections.append(h)
        
        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResBlock3D):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Decoder with skip connections
        for level, (blocks, upsample) in enumerate(
            zip(self.decoder_blocks[:-1], self.upsample_blocks)
        ):
            skip = skip_connections.pop()
            
            for i, block in enumerate(blocks):
                if i == 0:
                    # Concatenate skip connection
                    h = torch.cat([h, skip], dim=1)
                    
                if isinstance(block, ResBlock3D):
                    h = block(h, t_emb)
                else:
                    h = block(h)
                    
            h = upsample(h)
        
        # Last decoder level
        skip = skip_connections.pop()
        for i, block in enumerate(self.decoder_blocks[-1]):
            if i == 0:
                h = torch.cat([h, skip], dim=1)
                
            if isinstance(block, ResBlock3D):
                h = block(h, t_emb)
            else:
                h = block(h)
        
        # Final convolution
        output = self.final_conv(h)
        
        return output


def test_model():
    """Test the model with a sample input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(2, 4)
    ).to(device)
    
    # Test input
    batch_size = 1
    x = torch.randn(batch_size, 1, 64, 64, 64).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_model()
