import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------------------------------------
# 2D Position Embedding for HxW tokens
# -------------------------------------------------------------
class PositionalEncoding2D(nn.Module):
    """
    2D sine-cosine positional encoding
    output: (1, embed_dim, H, W)
    """
    def __init__(self, embed_dim=256, temperature=10000):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

        self.embed_dim = embed_dim
        self.temperature = temperature
        self.dim_half = embed_dim // 2
        self.dim_quarter = embed_dim // 4

    def forward(self, H, W, device):
        """
        return: (1, H*W, C)
        """

        y = torch.arange(H, device=device).float()
        x = torch.arange(W, device=device).float()

        # (H, W)
        yy = y.unsqueeze(1).repeat(1, W)
        xx = x.unsqueeze(0).repeat(H, 1)

        # frequencies
        dim_t = torch.arange(self.dim_quarter, device=device).float()
        dim_t = self.temperature ** (2 * dim_t / self.dim_quarter)

        # y-position encoding
        pos_y = yy[:, :, None] / dim_t[None, None, :]  # (H,W,C/4)
        pos_x = xx[:, :, None] / dim_t[None, None, :]  # (H,W,C/4)

        pos_y = torch.cat([torch.sin(pos_y), torch.cos(pos_y)], dim=2)  # (H,W,C/2)
        pos_x = torch.cat([torch.sin(pos_x), torch.cos(pos_x)], dim=2)  # (H,W,C/2)

        # final positional encoding
        pos = torch.cat([pos_y, pos_x], dim=2)  # (H, W, C)

        # flatten → (1, H*W, C)
        pos = pos.reshape(1, H * W, self.embed_dim)
        return pos


# -------------------------------------------------------------
# Transformer Aggregator
# -------------------------------------------------------------
class XFeatTransformerAggregator(nn.Module):
    def __init__(self, in_dim=64, embed_dim=256, depth=4, num_heads=8):
        super().__init__()

        # 将 (B, 64) → (B, embed_dim)
        self.input_proj = nn.Linear(in_dim, embed_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,     # output: (B, N, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 2D positional encoding
        self.pos_encoding = PositionalEncoding2D(embed_dim)

    def forward(self, x):
        """
        x: (B, 64, H, W)
        return: (B, embed_dim) global feature
        """
        B, C, H, W = x.shape

        # flatten spatial dims: (B, 64, H*W)
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C)
        # N = H*W tokens

        # project to transformer dim
        x = self.input_proj(x)  # (B, N, embed_dim)

        # positional encoding
        with torch.no_grad():
            pos = self.pos_encoding(H, W, device=x.device)  # (N, embed_dim)  # (N, embed_dim)
        x = x + pos  # (B, N, embed_dim)

        # transformer
        x = self.encoder(x)  # (B, N, embed_dim)

        # global pooling
        global_feat = x.mean(dim=1)  # (B, embed_dim)

        return global_feat