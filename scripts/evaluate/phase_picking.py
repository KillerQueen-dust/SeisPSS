"""Evaluate one complete phase-picking checkpoint on its saved test protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, help="Optional resolved YAML; otherwise use the checkpoint config.")
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    parser.add_argument("--tolerance-seconds", type=float)
    parser.add_argument("--output", type=Path, help="Optional JSON metrics file.")
    args = parser.parse_args()

    import lightning as L
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from seispss.downstream.phase_picking.task import PhasePickingModule

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(payload["hyper_parameters"]["cfg"])
        del payload
    cfg.data._target_ = "seispss.downstream.phase_picking.datamodule.PhasePickingDataModule"
    model_type = str(cfg.phase_model.type).lower()
    if model_type in {"seispss", "seismic_jepa"}:
        cfg.transfer.init = "scratch"
    else:
        raise ValueError(f"Unsupported phase-picking model type: {model_type}")
    if args.tolerance_seconds is not None:
        if args.tolerance_seconds <= 0:
            parser.error("--tolerance-seconds must be positive")
        cfg.eval.tolerance_samples = round(args.tolerance_seconds * float(cfg.data.sampling_rate))
    module = PhasePickingModule.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location="cpu", weights_only=False)
    datamodule = instantiate(cfg.data)
    precision = "32-true" if args.accelerator == "cpu" else str(cfg.trainer.get("precision", "32-true"))
    trainer = L.Trainer(accelerator=args.accelerator, devices=1, precision=precision,
                        logger=False, enable_checkpointing=False)
    results = trainer.test(module, datamodule=datamodule)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
