"""Lightning module for SeisPSS pretraining."""

from __future__ import annotations

from typing import Any

import lightning as L
import math
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MeanMetric

from seispss.data.augmentations import SAMPLE_TYPE_NAMES, WINDOW_SOURCE_NAMES
from seispss.models.jepa import JEPAForwardOutput, SeisPSS


class SeisPSSPretrainModule(L.LightningModule):
    """Train SeisPSS with automatic optimization and EMA target updates."""

    def __init__(self, *, cfg: DictConfig | dict[str, Any]) -> None:
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        default_loss_cfg = OmegaConf.create(
            {
                "regression_type": "smooth_l1",
                "smooth_l1_beta": 1.0,
                "regression_weight": 1.0,
                "cosine_weight": 0.25,
                "variance_weight": 0.05,
                "prediction_min_std": 0.6,
                "target_reduction": "token_mean",
            }
        )
        default_logging_cfg = OmegaConf.create(
            {
                "log_loss_terms": True,
                "representation_stats": {
                    "train": True,
                    "val": False,
                },
                "window_breakdown": {
                    "train": True,
                    "val": False,
                },
                "sample_type_breakdown": {
                    "train": True,
                    "val": False,
                },
                "dataset_breakdown": {
                    "train": True,
                    "val": False,
                },
                "dataset_stats": {
                    "train": False,
                    "val": False,
                },
                "diagnostics_prefix": "diagnostics/pretrain",
            }
        )
        self.cfg = cfg
        self.loss_cfg = OmegaConf.merge(default_loss_cfg, cfg.get("loss", OmegaConf.create({})))
        logging_cfg = cfg.get("logging", OmegaConf.create({}))
        pretrain_logging_cfg = logging_cfg.get("pretrain", OmegaConf.create({}))
        self.logging_cfg = OmegaConf.merge(default_logging_cfg, pretrain_logging_cfg)
        self.dataset_names = self._make_unique_loader_names(list(cfg.data.get("dataset_names", [])))
        self.model = SeisPSS(
            model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
            mask_cfg=OmegaConf.to_container(cfg.mask, resolve=True),
        )
        self.val_avg_loss = MeanMetric()
        self.val_loader_names: list[str] = []
        self._resolved_schedule_total_steps = 1
        self._resolved_warmup_steps = 0
        self._resolved_ema_ramp_steps = 1
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})

    def on_fit_start(self) -> None:
        datamodule = self.trainer.datamodule
        self.val_loader_names = self._make_unique_loader_names(list(getattr(datamodule, "val_loader_names", [])))

    def on_validation_epoch_start(self) -> None:
        self.val_avg_loss.reset()

    def forward(self, batch: dict[str, torch.Tensor]) -> JEPAForwardOutput:
        return self.model(
            x=batch["X"].float(),
            valid_time_mask=batch.get("valid_time_mask"),
            pseudo_anchor_sample=batch.get("pseudo_anchor_sample"),
            pseudo_anchor_confidence=batch.get("pseudo_anchor_confidence"),
            pseudo_anchor_valid=batch.get("pseudo_anchor_valid"),
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        loss, per_sample_loss, loss_terms = self._compute_loss(outputs)
        self._log_step_metrics(
            stage="train",
            loss=loss,
            per_sample_loss=per_sample_loss,
            loss_terms=loss_terms,
            window_source=batch["window_source"],
            sample_type=batch.get("sample_type"),
            dataset_id=batch.get("dataset_id"),
            outputs=outputs,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        outputs = self(batch)
        loss, per_sample_loss, loss_terms = self._compute_loss(outputs)
        self.val_avg_loss.update(loss.detach(), weight=int(batch["X"].shape[0]))
        loader_name = self._loader_name(dataloader_idx)
        self._log_step_metrics(
            stage=f"val/{loader_name}",
            loss=loss,
            per_sample_loss=per_sample_loss,
            loss_terms=loss_terms,
            window_source=batch["window_source"],
            sample_type=batch.get("sample_type"),
            dataset_id=batch.get("dataset_id"),
            outputs=outputs,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/avg_loss", self.val_avg_loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        base_lr = float(self.cfg.optim.lr)
        backbone_lr = base_lr * float(self.cfg.optim.get("backbone_lr_scale", 1.0))
        predictor_lr = base_lr * float(self.cfg.optim.get("predictor_lr_scale", 1.0))
        weight_decay = float(self.cfg.optim.weight_decay)
        optimizer = AdamW(
            params=self._build_optimizer_param_groups(
                backbone_lr=backbone_lr,
                predictor_lr=predictor_lr,
                weight_decay=weight_decay,
            ),
            betas=tuple(float(beta) for beta in self.cfg.optim.betas),
            eps=float(self.cfg.optim.eps),
        )
        total_steps = self._resolve_schedule_total_steps()
        warmup_steps = self._resolve_warmup_steps(total_steps=total_steps)
        self._resolved_schedule_total_steps = total_steps
        self._resolved_warmup_steps = warmup_steps
        self._resolved_ema_ramp_steps = self._resolve_ema_ramp_steps(total_steps=total_steps)

        def lr_lambda(step: int) -> float:
            min_scale = float(self.cfg.optim.min_lr_scale)
            clamped_step = min(step, total_steps)
            if warmup_steps > 0 and clamped_step < warmup_steps:
                return float(clamped_step + 1) / float(max(warmup_steps, 1))
            if total_steps <= warmup_steps:
                return min_scale
            progress = (clamped_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
            return float(min_scale + (1.0 - min_scale) * cosine)

        scheduler = LambdaLR(optimizer, lr_lambda=[lr_lambda] * len(optimizer.param_groups))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _build_optimizer_param_groups(
        self,
        *,
        backbone_lr: float,
        predictor_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {
            "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone_decay"},
            "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0, "name": "backbone_no_decay"},
            "predictor_decay": {"params": [], "lr": predictor_lr, "weight_decay": weight_decay, "name": "predictor_decay"},
            "predictor_no_decay": {"params": [], "lr": predictor_lr, "weight_decay": 0.0, "name": "predictor_no_decay"},
        }

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            prefix = "predictor" if name.startswith("predictor.") else "backbone"
            decay_type = "no_decay" if self._should_skip_weight_decay(name=name, parameter=parameter) else "decay"
            groups[f"{prefix}_{decay_type}"]["params"].append(parameter)

        return [group for group in groups.values() if group["params"]]

    @staticmethod
    def _should_skip_weight_decay(*, name: str, parameter: torch.nn.Parameter) -> bool:
        lower_name = name.lower()
        return (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in lower_name
            or "mask_token" in lower_name
        )

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        momentum = self._ema_momentum()
        self.model.update_target_encoder(momentum=momentum)
        self.log("train/ema_momentum", momentum, on_step=True, prog_bar=False, sync_dist=True)

    def _compute_loss(
        self,
        outputs: JEPAForwardOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        predictions = outputs.predictions.float()
        targets = outputs.targets.float()

        per_token_regression = self._compute_regression_loss(predictions=predictions, targets=targets)
        per_token_cosine = 1.0 - F.cosine_similarity(
            predictions,
            targets,
            dim=-1,
        )

        regression_loss, regression_per_sample = self._reduce_target_loss(per_token_regression, outputs=outputs)
        cosine_loss, cosine_per_sample = self._reduce_target_loss(per_token_cosine, outputs=outputs)

        pred_std_per_dim = predictions.reshape(-1, predictions.shape[-1]).std(dim=0, unbiased=False)
        variance_floor = float(self.loss_cfg.prediction_min_std)
        variance_loss = torch.relu(variance_floor - pred_std_per_dim).pow(2).mean()

        regression_weight = float(self.loss_cfg.regression_weight)
        cosine_weight = float(self.loss_cfg.cosine_weight)
        variance_weight = float(self.loss_cfg.variance_weight)
        per_sample_loss = (
            regression_weight * regression_per_sample
            + cosine_weight * cosine_per_sample
        )
        loss = per_sample_loss.mean() + variance_weight * variance_loss
        loss_terms = {
            "align": regression_loss,
            "cosine_aux": cosine_loss,
            "variance": variance_loss,
        }
        return loss, per_sample_loss, loss_terms

    def _reduce_target_loss(
        self,
        per_token_loss: torch.Tensor,
        *,
        outputs: JEPAForwardOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reduction = str(self.loss_cfg.get("target_reduction", "token_mean")).lower()
        horizon_ids = outputs.mask.target_horizon_ids
        if reduction in {"token_mean", "legacy"} or horizon_ids is None:
            per_sample = per_token_loss.mean(dim=1)
            return per_sample.mean(), per_sample
        if reduction != "horizon_balanced":
            raise ValueError(f"Unsupported loss.target_reduction: {reduction}")

        horizon_ids = horizon_ids.to(device=per_token_loss.device)
        valid = horizon_ids >= 0
        if not bool(valid.any()):
            per_sample = per_token_loss.mean(dim=1)
            return per_sample.mean(), per_sample
        num_horizons = int(horizon_ids[valid].max().item()) + 1
        safe_ids = horizon_ids.clamp(min=0, max=max(num_horizons - 1, 0))
        weights = valid.to(dtype=per_token_loss.dtype)
        sums = per_token_loss.new_zeros(per_token_loss.shape[0], num_horizons)
        counts = per_token_loss.new_zeros(per_token_loss.shape[0], num_horizons)
        sums.scatter_add_(1, safe_ids, per_token_loss * weights)
        counts.scatter_add_(1, safe_ids, weights)
        active = counts > 0
        horizon_means = sums / counts.clamp_min(1.0)
        per_sample = (horizon_means * active.to(dtype=per_token_loss.dtype)).sum(dim=1)
        per_sample = per_sample / active.to(dtype=per_token_loss.dtype).sum(dim=1).clamp_min(1.0)
        return per_sample.mean(), per_sample

    def _compute_regression_loss(
        self,
        *,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        regression_type = str(self.loss_cfg.get("regression_type", "mse")).lower()
        if regression_type in {"mse", "l2"}:
            return F.mse_loss(
                predictions,
                targets,
                reduction="none",
            ).mean(dim=-1)
        if regression_type in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(
                predictions,
                targets,
                beta=float(self.loss_cfg.get("smooth_l1_beta", 1.0)),
                reduction="none",
            ).mean(dim=-1)
        raise ValueError(f"Unsupported regression_type: {regression_type}")

    def _log_step_metrics(
        self,
        *,
        stage: str,
        loss: torch.Tensor,
        per_sample_loss: torch.Tensor,
        loss_terms: dict[str, torch.Tensor],
        window_source: torch.Tensor,
        sample_type: torch.Tensor | None,
        dataset_id: torch.Tensor | None,
        outputs: JEPAForwardOutput,
    ) -> None:
        is_train = stage.startswith("train")
        self.log(f"{stage}/loss", loss, on_step=is_train, on_epoch=not is_train, sync_dist=True)
        if bool(self.logging_cfg.get("log_loss_terms", True)):
            self.log(
                f"{stage}/loss_align",
                loss_terms["align"],
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            self.log(
                f"{stage}/loss_cosine_aux",
                loss_terms["cosine_aux"],
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            self.log(
                f"{stage}/loss_variance",
                loss_terms["variance"],
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )

        diagnostic_stage = self._diagnostic_stage(stage)
        self._log_mask_diagnostics(
            stage=diagnostic_stage,
            outputs=outputs,
            is_train=is_train,
        )
        if self._stage_logging_enabled("representation_stats", is_train=is_train):
            stats = self._representation_stats(outputs=outputs)
            self._log_mapping(
                prefix=diagnostic_stage,
                values=stats,
                on_step=is_train,
                on_epoch=not is_train,
            )
        if self._stage_logging_enabled("window_breakdown", is_train=is_train):
            self._log_window_source_breakdown(
                stage=diagnostic_stage,
                window_source=window_source,
                per_sample_loss=per_sample_loss,
                is_train=is_train,
            )
        if sample_type is not None and self._stage_logging_enabled("sample_type_breakdown", is_train=is_train):
            self._log_sample_type_breakdown(
                stage=diagnostic_stage,
                sample_type=sample_type,
                per_sample_loss=per_sample_loss,
                is_train=is_train,
            )
        if dataset_id is not None and self._stage_logging_enabled("dataset_breakdown", is_train=is_train):
            self._log_dataset_breakdown(
                stage=diagnostic_stage,
                dataset_id=dataset_id,
                per_sample_loss=per_sample_loss,
                outputs=outputs,
                is_train=is_train,
            )

    def _log_window_source_breakdown(
        self,
        *,
        stage: str,
        window_source: torch.Tensor,
        per_sample_loss: torch.Tensor,
        is_train: bool,
    ) -> None:
        window_source = window_source.view(-1)
        for source_id, source_name in WINDOW_SOURCE_NAMES.items():
            source_mask = window_source == source_id
            if not bool(source_mask.any()):
                continue
            source_loss = per_sample_loss[source_mask].mean()
            self.log(
                f"{stage}/window_source/{source_name}/loss",
                source_loss,
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            self.log(
                f"{stage}/window_source/{source_name}/fraction",
                source_mask.float().mean(),
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )

    def _log_sample_type_breakdown(
        self,
        *,
        stage: str,
        sample_type: torch.Tensor,
        per_sample_loss: torch.Tensor,
        is_train: bool,
    ) -> None:
        sample_type = sample_type.view(-1)
        for sample_type_id, sample_type_name in SAMPLE_TYPE_NAMES.items():
            source_mask = sample_type == sample_type_id
            if not bool(source_mask.any()):
                continue
            source_loss = per_sample_loss[source_mask].mean()
            self.log(
                f"{stage}/sample_type/{sample_type_name}/loss",
                source_loss,
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            self.log(
                f"{stage}/sample_type/{sample_type_name}/fraction",
                source_mask.float().mean(),
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )

    def _log_dataset_breakdown(
        self,
        *,
        stage: str,
        dataset_id: torch.Tensor,
        per_sample_loss: torch.Tensor,
        outputs: JEPAForwardOutput,
        is_train: bool,
    ) -> None:
        dataset_id = dataset_id.view(-1)
        log_stats = self._stage_logging_enabled("dataset_stats", is_train=is_train)
        for dataset_idx, dataset_name in enumerate(self.dataset_names):
            dataset_mask = dataset_id == dataset_idx
            if not bool(dataset_mask.any()):
                continue
            prefix = f"{stage}/dataset/{dataset_name}"
            self.log(
                f"{prefix}/loss",
                per_sample_loss[dataset_mask].mean(),
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            self.log(
                f"{prefix}/fraction",
                dataset_mask.float().mean(),
                on_step=is_train,
                on_epoch=not is_train,
                sync_dist=True,
            )
            if log_stats:
                stats = self._compact_representation_stats(outputs=outputs, sample_mask=dataset_mask)
                self._log_mapping(
                    prefix=prefix,
                    values=stats,
                    on_step=is_train,
                    on_epoch=not is_train,
                )

    def _log_mask_diagnostics(
        self,
        *,
        stage: str,
        outputs: JEPAForwardOutput,
        is_train: bool,
    ) -> None:
        mask = outputs.mask
        valid_tokens = mask.valid_token_mask.float().sum(dim=1).clamp_min(1.0)
        target_ratio = (mask.target_mask.float().sum(dim=1) / valid_tokens).mean()
        guard_ratio = outputs.predictions.new_tensor(0.0)
        if mask.guard_mask is not None:
            guard_ratio = (mask.guard_mask.float().sum(dim=1) / valid_tokens).mean()
        visible_ratio = (mask.context_visible_mask.float().sum(dim=1) / valid_tokens).mean()
        values: dict[str, torch.Tensor] = {
            "target_ratio": target_ratio,
            "guard_ratio": guard_ratio,
            "visible_ratio": visible_ratio,
        }
        if mask.anchor_token is not None:
            anchor_valid = mask.anchor_token >= 0
            values["anchor_valid_fraction"] = anchor_valid.float().mean()
            if mask.anchor_confidence is not None and bool(anchor_valid.any()):
                values["anchor_confidence_mean"] = mask.anchor_confidence[anchor_valid].float().mean()
            else:
                values["anchor_confidence_mean"] = outputs.predictions.new_tensor(0.0)
        if mask.anchored_branch_mask is not None:
            values["anchored_branch_fraction"] = mask.anchored_branch_mask.float().mean()
        if mask.target_horizon_ids is not None:
            horizon_ids = mask.target_horizon_ids
            total = horizon_ids.numel()
            horizon_names = getattr(self.model.mask_generator, "horizon_names", None)
            if horizon_names is None:
                horizon_names = [str(idx) for idx in range(int(horizon_ids.max().item()) + 1)]
            for horizon_idx, horizon_name in enumerate(horizon_names):
                fraction = (horizon_ids == horizon_idx).float().sum() / max(float(total), 1.0)
                values[f"horizon_{horizon_name}_fraction"] = fraction
        self._log_mapping(
            prefix=f"{stage}/mask",
            values=values,
            on_step=is_train,
            on_epoch=not is_train,
        )

    def _log_mapping(
        self,
        *,
        prefix: str,
        values: dict[str, torch.Tensor],
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in values.items():
            self.log(f"{prefix}/{name}", value, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def _representation_stats(
        self,
        *,
        outputs: JEPAForwardOutput,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        predictions = outputs.predictions.detach().float()
        targets = outputs.targets.detach().float()
        valid_token_mask = outputs.valid_token_mask
        target_mask = outputs.mask.target_mask
        if sample_mask is not None:
            predictions = predictions[sample_mask]
            targets = targets[sample_mask]
            valid_token_mask = valid_token_mask[sample_mask]
            target_mask = target_mask[sample_mask]

        valid_tokens = valid_token_mask.float().sum(dim=1)
        target_tokens = target_mask.float().sum(dim=1)
        target_ratio = (target_tokens / valid_tokens.clamp_min(1.0)).mean()
        pred_std = predictions.std(dim=(0, 1), unbiased=False).mean()
        target_std = targets.std(dim=(0, 1), unbiased=False).mean()
        pred_target_std_ratio = pred_std / target_std.clamp_min(1.0e-6)
        cosine = F.cosine_similarity(predictions, targets, dim=-1).mean()
        return {
            "target_ratio": target_ratio,
            "pred_std": pred_std,
            "target_std": target_std,
            "pred_target_std_ratio": pred_target_std_ratio,
            "cosine": cosine,
        }

    def _compact_representation_stats(
        self,
        *,
        outputs: JEPAForwardOutput,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        stats = self._representation_stats(outputs=outputs, sample_mask=sample_mask)
        return {
            "pred_target_std_ratio": stats["pred_target_std_ratio"],
            "cosine": stats["cosine"],
        }

    def _stage_logging_enabled(self, key: str, *, is_train: bool) -> bool:
        value = self.logging_cfg.get(key, False)
        if isinstance(value, DictConfig):
            stage_key = "train" if is_train else "val"
            return bool(value.get(stage_key, False))
        return bool(value)

    def _diagnostic_stage(self, stage: str) -> str:
        prefix = str(self.logging_cfg.get("diagnostics_prefix", "diagnostics/pretrain")).strip("/")
        return f"{prefix}/{stage}"

    def _loader_name(self, dataloader_idx: int) -> str:
        if dataloader_idx < len(self.val_loader_names):
            return self.val_loader_names[dataloader_idx]
        return f"loader_{dataloader_idx}"

    @staticmethod
    def _make_unique_loader_names(loader_names: list[str]) -> list[str]:
        counts = {name: loader_names.count(name) for name in loader_names}
        seen: dict[str, int] = {}
        unique_names: list[str] = []
        for name in loader_names:
            seen[name] = seen.get(name, 0) + 1
            if counts[name] == 1:
                unique_names.append(name)
            else:
                unique_names.append(f"{name}_{seen[name] - 1}")
        return unique_names

    def _resolve_schedule_total_steps(self) -> int:
        schedule_cfg = self.cfg.optim.get("schedule", OmegaConf.create({}))
        configured_total_steps = schedule_cfg.get("total_steps")
        if configured_total_steps is not None and int(configured_total_steps) > 0:
            return int(configured_total_steps)

        max_steps = int(getattr(self.trainer, "max_steps", -1))
        if max_steps > 0:
            return max_steps

        return max(int(self.trainer.estimated_stepping_batches), 1)

    def _resolve_warmup_steps(self, *, total_steps: int) -> int:
        schedule_cfg = self.cfg.optim.get("schedule", OmegaConf.create({}))
        configured_warmup_steps = schedule_cfg.get("warmup_steps")
        if configured_warmup_steps is not None:
            return max(min(int(configured_warmup_steps), total_steps), 0)

        warmup_ratio = float(self.cfg.optim.get("warmup_ratio", 0.0))
        return max(min(int(total_steps * warmup_ratio), total_steps), 0)

    def _resolve_ema_ramp_steps(self, *, total_steps: int) -> int:
        ema_cfg = self.cfg.optim.get("ema", OmegaConf.create({}))
        configured_ramp_steps = ema_cfg.get("ramp_steps")
        if configured_ramp_steps is not None and int(configured_ramp_steps) > 0:
            return int(configured_ramp_steps)
        return total_steps

    def _ema_momentum(self) -> float:
        start = float(self.cfg.optim.ema.start)
        end = float(self.cfg.optim.ema.end)
        ramp_steps = max(int(self._resolved_ema_ramp_steps), 1)
        progress = min(float(self.global_step) / float(max(ramp_steps - 1, 1)), 1.0)
        return start + (end - start) * progress

    def build_backbone_export_payload(self, *, variant: str, source_ckpt_path: str | None = None) -> dict[str, Any]:
        return {
            "format_version": "seispss_backbone_v1",
            "variant": variant,
            "source_ckpt_path": source_ckpt_path,
            "run_id": self.cfg.get("run", {}).get("run_id"),
            "model_cfg": OmegaConf.to_container(self.cfg.model, resolve=True),
            "data_cfg": {
                "sampling_rate": self.cfg.data.sampling_rate,
                "component_order": self.cfg.data.component_order,
                "dimension_order": self.cfg.data.dimension_order,
                "amp_norm_type": self.cfg.data.amp_norm_type,
                "norm_eps": self.cfg.data.norm_eps,
            },
            "state_dict": self.model.get_backbone_state_dict(variant=variant),
        }
