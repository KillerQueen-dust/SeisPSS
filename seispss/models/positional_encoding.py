"""Positional encoding helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_1d_sincos_position_embedding(
    *,
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1, length, dim] sinusoidal embeddings."""
    if length <= 0:
        raise ValueError("length must be positive.")

    half_dim = dim // 2
    positions = torch.arange(length, device=device, dtype=torch.float32)
    div_term = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / max(half_dim, 1)
    )
    angles = positions[:, None] * div_term[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb.unsqueeze(0).to(dtype=dtype)
