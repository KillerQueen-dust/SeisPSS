"""Backbone-only model wrapper for downstream tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from seispss.models.encoders import build_encoder
from seispss.models.positional_encoding import build_1d_sincos_position_embedding
from seispss.models.stems import ConvStem1D


class SeisPSSBackbone(nn.Module):
    """Stem + encoder wrapper for downstream transfer."""

    def __init__(self, *, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.stem = ConvStem1D(
            in_channels=model_cfg["in_channels"],
            channels=list(model_cfg["stem_channels"]),
            kernel_sizes=list(model_cfg["stem_kernel_sizes"]),
            strides=list(model_cfg["stem_strides"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.encoder = build_encoder(model_cfg=model_cfg, dim=self.stem.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, valid_token_mask = self.stem(x, valid_time_mask)
        pos_emb = build_1d_sincos_position_embedding(
            length=tokens.shape[1],
            dim=tokens.shape[2],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        hidden = tokens + pos_emb
        hidden = hidden.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        hidden = self.encoder(hidden, valid_token_mask=valid_token_mask)
        return hidden, valid_token_mask

    @classmethod
    def from_export(
        cls,
        export_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple["SeisPSSBackbone", dict[str, Any]]:
        payload = torch.load(export_path, map_location=map_location, weights_only=False)
        model = cls(model_cfg=payload["model_cfg"])
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model, payload
