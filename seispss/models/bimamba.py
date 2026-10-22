"""Bidirectional Mamba encoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
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


class BiMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba2(
            d_model=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.mamba_bwd = Mamba2(
            d_model=dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.cpu_fallback_fwd = FeedForward(dim=dim, hidden_dim=dim, dropout=dropout)
        self.cpu_fallback_bwd = FeedForward(dim=dim, hidden_dim=dim, dropout=dropout)
        self.fuse = nn.Linear(2 * dim, dim)
        self.drop_path = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, hidden_dim=int(dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        x_fwd = self._run_mamba(self.mamba_fwd, self.cpu_fallback_fwd, x_norm)
        x_bwd = torch.flip(
            self._run_mamba(self.mamba_bwd, self.cpu_fallback_bwd, torch.flip(x_norm, dims=[1])),
            dims=[1],
        )
        x = residual + self.drop_path(self.fuse(torch.cat([x_fwd, x_bwd], dim=-1)))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)

        x = x + self.ffn(self.norm2(x))
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        return x

    def _run_mamba(self, module: Mamba2, fallback: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return module(x)
        return fallback(x)


class BiMambaEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    dim=dim,
                    mamba_d_state=mamba_d_state,
                    mamba_d_conv=mamba_d_conv,
                    mamba_expand=mamba_expand,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

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
