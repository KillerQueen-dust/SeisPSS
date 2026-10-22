"""Lightning task for dense phase picking."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from seispss.downstream.phase_picking.factory import build_phase_picking_model
from seispss.downstream.phase_picking.metrics import PhasePickingMetrics


class PhasePickingModule(L.LightningModule):
    """Train and evaluate P/S/noise dense phase picking."""

    def __init__(self, *, cfg: DictConfig | dict[str, Any]) -> None:
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.model = build_phase_picking_model(cfg)
        self.register_buffer(
            "class_weights",
            torch.as_tensor(list(cfg.optim.loss.class_weights), dtype=torch.float32),
            persistent=False,
        )
        self.val_metrics = self._build_metrics()
        self.test_metrics = self._build_metrics()
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.set_frozen_backbone_eval()
        return self

    def on_train_epoch_start(self) -> None:
        self.model.set_frozen_backbone_eval()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(
            batch["X"].float(),
            valid_time_mask=batch.get("valid_time_mask"),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits = self(batch)
        loss = self._compute_loss(logits=logits, batch=batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits = self._eval_logits(batch)
        loss = self._compute_loss(logits=logits, batch=batch)
        self.val_metrics.update(logits=logits, batch=batch)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=int(batch["X"].shape[0]),
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self._log_metric_dict(stage="val", metrics=self.val_metrics.compute())

    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        logits = self._eval_logits(batch)
        loss = self._compute_loss(logits=logits, batch=batch)
        self.test_metrics.update(logits=logits, batch=batch)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=int(batch["X"].shape[0]),
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_metric_dict(stage="test", metrics=self.test_metrics.compute())

    def configure_optimizers(self) -> dict[str, Any]:
        param_groups = self.model.optimizer_parameter_groups(self.cfg.optim)
        param_groups = [
            {**group, "params": [param for param in group["params"] if param.requires_grad]}
            for group in param_groups
        ]
        param_groups = [group for group in param_groups if group["params"]]
        if not param_groups:
            raise ValueError("No trainable parameters are available for phase-picking optimization.")

        optimizer = AdamW(
            param_groups,
            betas=tuple(float(beta) for beta in self.cfg.optim.betas),
            eps=float(self.cfg.optim.eps),
        )
        total_steps = self._resolve_total_steps()
        warmup_steps = self._resolve_warmup_steps(total_steps=total_steps)

        def lr_lambda(step: int) -> float:
            min_scale = float(self.cfg.optim.min_lr_scale)
            clamped_step = min(int(step), total_steps)
            if warmup_steps > 0 and clamped_step < warmup_steps:
                return float(clamped_step + 1) / float(max(warmup_steps, 1))
            if total_steps <= warmup_steps:
                return min_scale
            progress = (clamped_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_scale + (1.0 - min_scale) * cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(param_groups))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _compute_loss(self, *, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        targets = batch["y"].to(device=logits.device, dtype=logits.dtype)
        log_probs = F.log_softmax(logits, dim=1)
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1, 1)
        per_time_loss = -(targets * log_probs * weights).sum(dim=1)
        valid_time_mask = batch.get("valid_time_mask")
        if valid_time_mask is None:
            return per_time_loss.mean()
        mask = valid_time_mask.to(device=logits.device, dtype=logits.dtype)
        return (per_time_loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _eval_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["X"].float()
        valid_time_mask = batch.get("valid_time_mask")
        max_window = int(self.cfg.eval.max_inference_window_samples)
        if x.shape[-1] <= max_window:
            return self.model(x, valid_time_mask=valid_time_mask)
        return self.model.sliding_inference(
            x,
            valid_time_mask=valid_time_mask,
            window_samples=max_window,
            stride_samples=int(self.cfg.eval.sliding_stride_samples),
            aggregation=str(self.cfg.eval.get("sliding_aggregation", "avg_logits")),
        )

    def _build_metrics(self) -> PhasePickingMetrics:
        return PhasePickingMetrics(
            threshold_p=float(self.cfg.eval.threshold_p),
            threshold_s=float(self.cfg.eval.threshold_s),
            tolerance_samples=int(self.cfg.eval.tolerance_samples),
            min_peak_distance_samples=int(self.cfg.eval.min_peak_distance_samples),
            sampling_rate=float(self.cfg.data.sampling_rate),
        )

    def _log_metric_dict(self, *, stage: str, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                torch.tensor(float(value), device=self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=name == "f1",
                sync_dist=True,
            )

    def _resolve_total_steps(self) -> int:
        configured = self.cfg.optim.schedule.get("total_steps")
        if configured is not None:
            return max(int(configured), 1)
        trainer_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if trainer_steps is not None and trainer_steps != float("inf"):
            return max(int(trainer_steps), 1)
        return max(int(self.cfg.trainer.max_steps), 1)

    def _resolve_warmup_steps(self, *, total_steps: int) -> int:
        configured = self.cfg.optim.schedule.get("warmup_steps")
        if configured is not None:
            return max(int(configured), 0)
        return max(int(round(total_steps * float(self.cfg.optim.warmup_ratio))), 0)
