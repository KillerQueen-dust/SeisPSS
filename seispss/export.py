"""Backbone export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from seispss.models.backbone import SeisPSSBackbone
from seispss.runtime import infer_run_dir_from_ckpt


def normalize_saved_cfg(cfg_like: Any) -> DictConfig:
    if isinstance(cfg_like, DictConfig):
        return cfg_like
    if isinstance(cfg_like, dict) and "cfg" in cfg_like:
        return OmegaConf.create(cfg_like["cfg"])
    return OmegaConf.create(cfg_like)


def build_backbone_export_payload(
    *,
    lightning_ckpt: dict[str, Any],
    variant: str,
    source_ckpt_path: str,
) -> dict[str, Any]:
    cfg = extract_cfg_from_checkpoint(lightning_ckpt)
    state_dict = lightning_ckpt["state_dict"]
    exported_state = select_backbone_state_dict(state_dict=state_dict, variant=variant)
    try:
        run_dir = str(infer_run_dir_from_ckpt(Path(source_ckpt_path)))
    except ValueError:
        run_dir = str(Path(source_ckpt_path).parent)
    return {
        "format_version": "seispss_backbone_v1",
        "variant": variant,
        "source_ckpt_path": source_ckpt_path,
        "source_run_dir": run_dir,
        "run_id": cfg.get("run", {}).get("run_id"),
        "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
        "data_cfg": {
            "sampling_rate": cfg.data.sampling_rate,
            "component_order": cfg.data.component_order,
            "dimension_order": cfg.data.dimension_order,
            "amp_norm_type": cfg.data.amp_norm_type,
            "norm_eps": cfg.data.norm_eps,
        },
        "state_dict": exported_state,
    }


def export_backbone_from_checkpoint(
    *,
    source_ckpt_path: str | Path,
    output_path: str | Path,
    variant: str,
) -> Path:
    source_ckpt_path = Path(source_ckpt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lightning_ckpt = torch.load(source_ckpt_path, map_location="cpu", weights_only=False)
    payload = build_backbone_export_payload(
        lightning_ckpt=lightning_ckpt,
        variant=variant,
        source_ckpt_path=str(source_ckpt_path),
    )
    torch.save(payload, output_path)
    return output_path


def load_exported_backbone(
    export_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(export_path, map_location=map_location, weights_only=False)
    format_version = str(payload.get("format_version", "seispss_backbone_v1"))
    if format_version in {"seispss_backbone_v1", "seismicjepa_backbone_v1"}:
        model = SeisPSSBackbone(model_cfg=payload["model_cfg"])
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model, payload
    raise ValueError(f"Unsupported SeisPSS backbone format: {format_version}")


def extract_cfg_from_checkpoint(lightning_ckpt: dict[str, Any]) -> DictConfig:
    hparams = lightning_ckpt.get("hyper_parameters", {})
    if "cfg" in hparams:
        return normalize_saved_cfg(hparams["cfg"])
    return normalize_saved_cfg(hparams)


def select_backbone_state_dict(state_dict: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    variant = variant.lower()
    if variant == "online":
        stem_prefix = "model.online_stem."
        encoder_prefix = "model.context_encoder."
    elif variant == "ema":
        stem_prefix = "model.target_stem."
        encoder_prefix = "model.target_encoder."
    else:
        raise ValueError(f"Unsupported export variant: {variant}")

    exported: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(stem_prefix):
            exported[f"stem.{key[len(stem_prefix):]}"] = value.detach().cpu()
        elif key.startswith(encoder_prefix):
            exported[f"encoder.{key[len(encoder_prefix):]}"] = value.detach().cpu()
    if not exported:
        raise ValueError(f"No backbone weights found for variant={variant}.")
    return exported
