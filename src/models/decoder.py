"""
Conditional DDPM Decoder (Module 7)
Generates design images from semantic metadata using denoising diffusion

Architecture:
    (Noisy Image x_t, Timestep t, V_Meta) → U-Net → Predicted Noise ε

Input:  x_t [B, 3, 256, 256], t [B], v_meta [B, vmeta_dim]
Output: ε_predicted [B, 3, 256, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embeddings for timesteps (similar to Transformer position encoding)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [B] timestep indices

        Returns:
            embeddings: [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Projects timestep embeddings to model dimension
    """

    def __init__(self, time_dim: int, emb_dim: int):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timesteps

        Returns:
            time_emb: [B, emb_dim]
        """
        t_emb = self.time_embedding(t)
        return self.mlp(t_emb)


class MetaEmbedding(nn.Module):
    """
    Embeds V_Meta (goal, format, tone) into conditioning vector
    """

    def __init__(self,
                 n_goal_classes: int = 10,
                 n_format_classes: int = 4,
                 emb_dim: int = 256):
        super().__init__()

        # Embeddings for categorical variables
        self.goal_embedding = nn.Embedding(n_goal_classes, emb_dim // 4)
        self.format_embedding = nn.Embedding(n_format_classes, emb_dim // 4)

        # MLP for tone (continuous)
        self.tone_mlp = nn.Sequential(
            nn.Linear(1, emb_dim // 4),
            nn.SiLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )

        # Projection to full embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(3 * emb_dim // 4, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, v_meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_meta: [B, 3] containing [v_goal_id, v_format_id, v_tone]

        Returns:
            meta_emb: [B, emb_dim]
        """
        # Extract components
        v_goal = v_meta[:, 0].long()      # [B]
        v_format = v_meta[:, 1].long()    # [B]
        v_tone = v_meta[:, 2:3]           # [B, 1]

        # Embed each component
        goal_emb = self.goal_embedding(v_goal)          # [B, emb_dim//4]
        format_emb = self.format_embedding(v_format)    # [B, emb_dim//4]
        tone_emb = self.tone_mlp(v_tone)                # [B, emb_dim//4]

        # Concatenate and project
        combined = torch.cat([goal_emb, format_emb, tone_emb], dim=1)  # [B, 3*emb_dim//4]
        return self.projection(combined)  # [B, emb_dim]


class ResidualBlock(nn.Module):
    """
    Residual block with time and metadata conditioning
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            time_emb: [B, time_emb_dim]

        Returns:
            out: [B, out_channels, H, W]
        """
        residual = self.residual_conv(x)

        # First conv
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding
        time_emb_proj = self.time_mlp(time_emb)[:, :, None, None]  # [B, C, 1, 1]
        h = h + time_emb_proj

        h = F.silu(h)
        h = self.dropout(h)

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)

        return F.silu(h + residual)


class AttentionBlock(nn.Module):
    """
    Self-attention block for U-Net
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]

        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)  # [B, 3C, H, W]

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)  # [B, 3, heads, C//heads, HW]
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, HW, C//heads]

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, C//heads]

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)

        return out + residual


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for DDPM with time and metadata conditioning
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 base_channels: int = 64,
                 channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
                 n_goal_classes: int = 10,
                 n_format_classes: int = 4,
                 time_emb_dim: int = 256,
                 meta_emb_dim: int = 256,
                 num_res_blocks: int = 2,
                 attention_levels: Tuple[int, ...] = (1, 2),
                 dropout: float = 0.1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Embeddings
        self.time_embedding = TimeEmbedding(time_emb_dim, time_emb_dim)
        self.meta_embedding = MetaEmbedding(n_goal_classes, n_format_classes, meta_emb_dim)

        # Combine time and meta embeddings
        self.cond_projection = nn.Linear(time_emb_dim + meta_emb_dim, time_emb_dim)

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels_block = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(now_channels, out_channels_block, time_emb_dim, dropout)
                )
                now_channels = out_channels_block
                channels.append(now_channels)

                # Add attention
                if i in attention_levels:
                    self.down_blocks.append(AttentionBlock(now_channels))
                    channels.append(now_channels)

            # Downsample (except last level)
            if i != len(channel_multipliers) - 1:
                self.down_blocks.append(nn.Conv2d(now_channels, now_channels, 3, stride=2, padding=1))
                channels.append(now_channels)

        # Middle blocks
        self.mid_block1 = ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(now_channels)
        self.mid_block2 = ResidualBlock(now_channels, now_channels, time_emb_dim, dropout)

        # Upsample blocks
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_multipliers)):
            out_channels_block = base_channels * mult

            for j in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResidualBlock(
                        now_channels + channels.pop(),
                        out_channels_block,
                        time_emb_dim,
                        dropout
                    )
                )
                now_channels = out_channels_block

                # Add attention
                if i in attention_levels:
                    self.up_blocks.append(AttentionBlock(now_channels))

            # Upsample (except last level)
            if i != len(channel_multipliers) - 1:
                self.up_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(now_channels, now_channels, 3, padding=1)
                    )
                )

        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, self.out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, v_meta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Noisy images [B, 3, 256, 256]
            t: Timesteps [B]
            v_meta: Optional metadata [B, 3] (goal_id, format_id, tone)

        Returns:
            Predicted noise [B, 3, 256, 256]
        """
        # Embed time
        time_emb = self.time_embedding(t)  # [B, time_emb_dim]

        # Embed metadata if provided
        if v_meta is not None:
            meta_emb = self.meta_embedding(v_meta)  # [B, meta_emb_dim]
            # Combine time and meta
            combined_emb = torch.cat([time_emb, meta_emb], dim=1)
            cond_emb = self.cond_projection(combined_emb)  # [B, time_emb_dim]
        else:
            cond_emb = time_emb

        # Initial conv
        h = self.conv_in(x)

        # Downsample
        hs = [h]
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, cond_emb)
            else:
                h = module(h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond_emb)

        # Upsample
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, cond_emb)
            else:
                h = module(h)

        # Final conv
        return self.conv_out(h)


if __name__ == '__main__':
    """Test the conditional U-Net"""
    print("=" * 60)
    print("Testing Conditional U-Net Decoder")
    print("=" * 60)

    # Create model
    model = ConditionalUNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        n_goal_classes=10,
        n_format_classes=4
    )

    print(f"\nModel created")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    t = torch.randint(0, 1000, (batch_size,))
    v_meta = torch.tensor([
        [0, 0, 0.5],  # goal=0, format=0, tone=0.5
        [1, 1, 0.7],
        [2, 2, 0.3],
        [3, 0, 0.9]
    ])

    print(f"\nInput shapes:")
    print(f"  x (noisy image): {x.shape}")
    print(f"  t (timesteps): {t.shape}")
    print(f"  v_meta: {v_meta.shape}")

    # Forward pass
    output = model(x, t, v_meta)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output mean/std: {output.mean():.4f} / {output.std():.4f}")

    # Test without conditioning
    output_uncond = model(x, t, None)
    print(f"\nUnconditioned output shape: {output_uncond.shape}")

    print("\n" + "=" * 60)
    print("Conditional U-Net test passed!")
    print("=" * 60)
