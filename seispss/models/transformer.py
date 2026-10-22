"""Bidirectional Transformer encoder (architecture ablation counterpart of BiMambaEncoder).

Drop-in replacement for :class:`~seispss.models.bimamba.BiMambaEncoder`: same
constructor contract (dim/depth/mlp_ratio/dropout), same ``forward(x, valid_token_mask)``
signature, same ``.layers`` / ``.norm`` attributes, and the same invalid-token zeroing
convention, so stem, positional encoding, masking, predictor, head and export paths are
untouched. Self-attention is inherently bidirectional, so one block replaces the
forward+backward scan pair of a Bi-Mamba block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = float(dropout)
        self.proj_dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn_mask = None
        if valid_token_mask is not None:
            # SDPA boolean mask: True = key participates in attention.
            keep = valid_token_mask
            # A row with no valid key would make the softmax produce NaN; keep one key alive.
            keep = keep | (~keep.any(dim=1, keepdim=True))
            attn_mask = keep[:, None, None, :]

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.proj_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), valid_token_mask=valid_token_mask)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)

        x = x + self.ffn(self.norm2(x))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self._init_weights(depth=depth)

    def _init_weights(self, *, depth: int) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # GPT-2 style residual scaling: keeps the pre-LN residual stream variance bounded
        # with depth, which matters here because the ablation must run at the Bi-Mamba
        # learning rate without a separate stability search.
        scale = 1.0 / math.sqrt(2.0 * depth)
        for block in self.layers:
            block.attn.proj.weight.data.mul_(scale)
            block.ffn.net[3].weight.data.mul_(scale)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask=valid_token_mask)
        x = self.norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        return x
