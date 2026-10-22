"""Factory for dense phase-picking model variants."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from seispss.downstream.phase_picking.model import PhasePickingModel


def build_phase_picking_model(cfg: DictConfig):
    """Build a model that follows the dense P/S/noise phase-picking contract."""

    phase_model_cfg = cfg.get("phase_model")
    model_type = "seispss" if phase_model_cfg is None else str(phase_model_cfg.type).lower()

    if model_type in {"seispss", "seismic_jepa"}:
        return PhasePickingModel(
            model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
            head_cfg=OmegaConf.to_container(cfg.head, resolve=True),
            transfer_cfg=OmegaConf.to_container(cfg.transfer, resolve=True),
        )

    raise ValueError(f"Unsupported phase-picking model type: {model_type}")
