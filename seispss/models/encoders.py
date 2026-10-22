"""Encoder factory shared by the JEPA pretraining model and the downstream backbone.

``model_cfg["encoder_type"]`` selects the token mixer. Exports written before this key
existed carry no ``encoder_type``, so the default must stay ``bimamba``.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from seispss.models.transformer import TransformerEncoder

ENCODER_TYPES = ("bimamba", "transformer")


def resolve_encoder_type(model_cfg: dict[str, Any]) -> str:
    encoder_type = str(model_cfg.get("encoder_type", "bimamba")).lower()
    if encoder_type not in ENCODER_TYPES:
        raise ValueError(f"Unsupported model.encoder_type: {encoder_type} (expected one of {ENCODER_TYPES}).")
    return encoder_type


def build_encoder(*, model_cfg: dict[str, Any], dim: int) -> nn.Module:
    encoder_type = resolve_encoder_type(model_cfg)
    if encoder_type == "bimamba":
        from seispss.models.bimamba import BiMambaEncoder

        return BiMambaEncoder(
            dim=dim,
            depth=int(model_cfg["encoder_depth"]),
            mamba_d_state=int(model_cfg["mamba_d_state"]),
            mamba_d_conv=int(model_cfg["mamba_d_conv"]),
            mamba_expand=int(model_cfg["mamba_expand"]),
            mlp_ratio=float(model_cfg["encoder_mlp_ratio"]),
            dropout=float(model_cfg["dropout"]),
        )
    return TransformerEncoder(
        dim=dim,
        depth=int(model_cfg["encoder_depth"]),
        num_heads=int(model_cfg["encoder_num_heads"]),
        mlp_ratio=float(model_cfg["encoder_mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
    )
