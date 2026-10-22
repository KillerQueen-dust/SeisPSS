"""Callbacks for experiment bookkeeping."""

from __future__ import annotations

import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm, convert_inf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT

from seispss.export import export_backbone_from_checkpoint
from seispss.runtime import write_run_state


def _backbone_export_filename(*, export_kind: str, variant: str) -> str:
    if export_kind in {"seispss", "jepa"}:
        return f"backbone-{variant}.pt"
    raise ValueError(f"Unsupported backbone export kind: {export_kind}")


def _export_checkpoint_backbone(*, export_kind: str, source_ckpt_path: Path, output_path: Path, variant: str) -> None:
    if export_kind in {"seispss", "jepa"}:
        export_backbone_from_checkpoint(
            source_ckpt_path=source_ckpt_path,
            output_path=output_path,
            variant=variant,
        )
        return
    raise ValueError(f"Unsupported backbone export kind: {export_kind}")


class StepProgressBar(TQDMProgressBar):
    """Step-first progress bar for long pretraining runs."""

    TRAIN_BAR_FORMAT = "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def __init__(
        self,
        *,
        monitor: str = "val/avg_loss",
        mode: str = "min",
        refresh_rate: int = 1,
        process_position: int = 0,
        leave: bool = False,
        ncols: int | None = None,
        dynamic_ncols: bool = True,
    ) -> None:
        super().__init__(
            refresh_rate=refresh_rate,
            process_position=process_position,
            leave=leave,
        )
        self.monitor = monitor
        self.mode = mode.lower()
        self.ncols = ncols
        self.dynamic_ncols = dynamic_ncols
        self.best_score: float | None = None
        self.best_step: int | None = None

    def init_train_tqdm(self):  # type: ignore[override]
        return self._init_tqdm(
            desc=self.train_description,
            position=2 * self.process_position,
            leave=self._leave,
            bar_format=self.TRAIN_BAR_FORMAT,
            smoothing=0,
        )

    def init_validation_tqdm(self):  # type: ignore[override]
        has_main_bar = self.trainer.state.fn != "validate"
        return self._init_tqdm(
            desc=self.validation_description,
            position=2 * self.process_position + has_main_bar,
            leave=not has_main_bar,
        )

    def init_test_tqdm(self):  # type: ignore[override]
        return self._init_tqdm(
            desc="Testing",
            position=2 * self.process_position,
            leave=True,
        )

    def init_predict_tqdm(self):  # type: ignore[override]
        return self._init_tqdm(
            desc=self.predict_description,
            position=2 * self.process_position,
            leave=True,
            smoothing=0,
        )

    def on_train_epoch_start(self, trainer: Trainer, *args: Any) -> None:  # type: ignore[override]
        if self._leave:
            self.train_progress_bar = self.init_train_tqdm()
        total = convert_inf(self._total_train_steps(trainer))
        self.train_progress_bar.reset(total=total)
        self.train_progress_bar.n = int(trainer.global_step)
        self.train_progress_bar.initial = int(trainer.global_step)
        self.train_progress_bar.set_description(self._step_description(trainer))
        self.train_progress_bar.refresh()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:  # type: ignore[override]
        del outputs, batch
        batch_total = convert_inf(self.total_train_batches) or batch_idx + 1
        if self.train_progress_bar is not None and self._should_update(batch_idx + 1, batch_total):
            self.train_progress_bar.n = int(trainer.global_step)
            self.train_progress_bar.set_description(self._step_description(trainer))
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.train_progress_bar.refresh()

    def on_validation_end(self, trainer: Trainer, pl_module) -> None:  # type: ignore[override]
        if not trainer.sanity_checking:
            self._update_best_score(trainer)
        super().on_validation_end(trainer, pl_module)

    def get_metrics(self, trainer: Trainer, pl_module) -> dict[str, int | float]:  # type: ignore[override]
        del pl_module
        metrics: dict[str, int | float] = {}
        monitor_value = self._metric_to_float(trainer.callback_metrics.get(self.monitor))
        if monitor_value is not None:
            metrics[self.monitor] = monitor_value
        if self.best_step is not None:
            metrics["best_step"] = self.best_step
        return metrics

    def _update_best_score(self, trainer: Trainer) -> None:
        score = self._metric_to_float(trainer.callback_metrics.get(self.monitor))
        if score is None:
            return
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.best_step = int(trainer.global_step)

    def _is_better(self, score: float, best_score: float) -> bool:
        if self.mode == "max":
            return score > best_score
        return score < best_score

    @staticmethod
    def _metric_to_float(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return None
        return metric if math.isfinite(metric) else None

    @staticmethod
    def _total_train_steps(trainer: Trainer) -> int | float:
        max_steps = int(getattr(trainer, "max_steps", -1))
        if max_steps > 0:
            return max_steps
        return trainer.num_training_batches

    def _step_description(self, trainer: Trainer) -> str:
        max_steps = int(getattr(trainer, "max_steps", -1))
        if max_steps > 0:
            return f"Step {int(trainer.global_step)}/{max_steps}"
        return f"Step {int(trainer.global_step)}"

    def _init_tqdm(
        self,
        *,
        desc: str,
        position: int,
        leave: bool,
        bar_format: str | None = None,
        smoothing: float | None = None,
    ) -> Tqdm:
        kwargs: dict[str, Any] = {
            "desc": desc,
            "position": position,
            "disable": self.is_disabled,
            "leave": leave,
            "dynamic_ncols": self.dynamic_ncols,
            "file": sys.stdout,
            "bar_format": self.BAR_FORMAT if bar_format is None else bar_format,
        }
        if self.ncols is not None:
            kwargs["ncols"] = self.ncols
        if smoothing is not None:
            kwargs["smoothing"] = smoothing
        return Tqdm(**kwargs)


class RunStateCallback(Callback):
    """Persist run metadata for resume and downstream export workflows."""

    def __init__(
        self,
        *,
        run_id: str,
        run_name: str,
        run_dir: str,
        state_path: str,
        latest_ckpt_path: str,
        resume_ckpt_path: str | None,
        hydra_output_dir: str | None,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.run_name = run_name
        self.run_dir = run_dir
        self.state_path = Path(state_path)
        self.latest_ckpt_path = latest_ckpt_path
        self.resume_ckpt_path = resume_ckpt_path
        self.hydra_output_dir = hydra_output_dir

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        self._write_state(trainer=trainer, status="running")

    def on_save_checkpoint(self, trainer: Trainer, pl_module, checkpoint: dict[str, Any]) -> None:  # type: ignore[override]
        del pl_module, checkpoint
        self._write_state(trainer=trainer, status="running")

    def on_train_end(self, trainer: Trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        self._write_state(trainer=trainer, status="finished")

    def on_exception(self, trainer: Trainer, pl_module, exception: BaseException) -> None:  # type: ignore[override]
        del pl_module
        self._write_state(trainer=trainer, status="failed", exception=str(exception))

    def _write_state(
        self,
        *,
        trainer: Trainer,
        status: str,
        exception: str | None = None,
    ) -> None:
        logger_info = self._extract_logger_info(trainer)
        checkpoint_info = self._extract_checkpoint_info(trainer)
        state = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "latest_ckpt_path": self.latest_ckpt_path,
            "resume_ckpt_path": self.resume_ckpt_path,
            "hydra_output_dir": self.hydra_output_dir,
            "wandb": logger_info,
            "checkpoints": checkpoint_info,
        }
        if exception is not None:
            state["exception"] = exception
        write_run_state(self.state_path, state)

    def _extract_logger_info(self, trainer: Trainer) -> dict[str, Any]:
        if isinstance(trainer.logger, WandbLogger):
            try:
                experiment = trainer.logger.experiment
            except Exception as exc:  # pragma: no cover - defensive fallback for logger init failures
                return {"id": None, "name": self.run_name, "project": None, "error": str(exc)}
            return {
                "id": getattr(experiment, "id", None),
                "name": getattr(experiment, "name", self.run_name),
                "project": getattr(experiment, "project", None),
            }
        return {}

    def _extract_checkpoint_info(self, trainer: Trainer) -> list[dict[str, Any]]:
        checkpoint_info = []
        for callback in trainer.checkpoint_callbacks:
            checkpoint_info.append(
                {
                    "dirpath": str(callback.dirpath) if callback.dirpath is not None else None,
                    "best_model_path": callback.best_model_path,
                    "best_model_score": float(callback.best_model_score) if callback.best_model_score is not None else None,
                    "last_model_path": getattr(callback, "last_model_path", None),
                }
            )
        return checkpoint_info


class BackboneExportCallback(Callback):
    """Export downstream-ready backbone artifacts whenever checkpoints are saved."""

    _STEP_PATTERN = re.compile(r"step=(\d+)")

    def __init__(
        self,
        *,
        checkpoints_dir: str,
        history_dir: str,
        exports_dir: str,
        latest_ckpt_path: str,
        variant: str = "ema",
        export_kind: str = "seispss",
    ) -> None:
        super().__init__()
        self.checkpoints_dir = Path(checkpoints_dir)
        self.history_dir = Path(history_dir)
        self.exports_dir = Path(exports_dir)
        self.latest_ckpt_path = Path(latest_ckpt_path)
        self.variant = variant
        self.export_kind = str(export_kind).lower()
        self._synced_latest_signature: tuple[str, float, int] | None = None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:  # type: ignore[override]
        del pl_module, outputs, batch, batch_idx
        if trainer.sanity_checking:
            return
        self._sync_checkpoint_artifacts()

    def on_validation_end(self, trainer: Trainer, pl_module) -> None:  # type: ignore[override]
        del pl_module
        if trainer.sanity_checking:
            return
        self._sync_checkpoint_artifacts()

    def on_train_end(self, trainer: Trainer, pl_module) -> None:  # type: ignore[override]
        del trainer, pl_module
        self._sync_checkpoint_artifacts()

    def _sync_checkpoint_artifacts(self) -> None:
        self._refresh_latest_checkpoint()
        for ckpt_path in self._checkpoint_export_sources():
            output_path = self._export_path_for_checkpoint(ckpt_path)
            if output_path is None or not self._needs_export(ckpt_path, output_path):
                continue
            self._export_checkpoint(source_ckpt_path=ckpt_path, output_path=output_path)

    def _refresh_latest_checkpoint(self) -> None:
        latest_source = self._latest_step_checkpoint()
        if latest_source is None or latest_source.resolve() == self.latest_ckpt_path.resolve():
            return
        latest_stat = latest_source.stat()
        latest_signature = (str(latest_source.resolve()), latest_stat.st_mtime, latest_stat.st_size)
        if latest_signature == self._synced_latest_signature:
            return
        self.latest_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.latest_ckpt_path.with_name(f".{self.latest_ckpt_path.name}.tmp")
        shutil.copy2(latest_source, tmp_path)
        tmp_path.replace(self.latest_ckpt_path)
        self._synced_latest_signature = latest_signature

    def _checkpoint_export_sources(self) -> list[Path]:
        sources = []
        sources.extend(sorted(self.history_dir.glob("step=*.ckpt"), key=self._checkpoint_sort_key))
        sources.extend(sorted(self.checkpoints_dir.glob("best-step=*.ckpt"), key=self._checkpoint_sort_key))
        if self.latest_ckpt_path.exists():
            sources.append(self.latest_ckpt_path)
        return sources

    def _latest_step_checkpoint(self) -> Path | None:
        candidates = [
            path
            for path in self._checkpoint_export_sources()
            if path.name != self.latest_ckpt_path.name and self._step_from_name(path) is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=self._checkpoint_sort_key)

    def _export_path_for_checkpoint(self, ckpt_path: Path) -> Path | None:
        if ckpt_path.parent.resolve() == self.history_dir.resolve() and ckpt_path.name.startswith("step="):
            return self.exports_dir / ckpt_path.with_suffix(".pt").name
        if ckpt_path.parent.resolve() == self.checkpoints_dir.resolve() and ckpt_path.name.startswith("best-step="):
            return self.exports_dir / ckpt_path.with_suffix(".pt").name
        if ckpt_path.resolve() == self.latest_ckpt_path.resolve():
            return self.exports_dir / _backbone_export_filename(export_kind=self.export_kind, variant=self.variant)
        return None

    def _export_checkpoint(self, *, source_ckpt_path: Path, output_path: Path) -> None:
        _export_checkpoint_backbone(
            export_kind=self.export_kind,
            source_ckpt_path=source_ckpt_path,
            output_path=output_path,
            variant=self.variant,
        )

    @staticmethod
    def _needs_export(ckpt_path: Path, output_path: Path) -> bool:
        if not ckpt_path.exists():
            return False
        if not output_path.exists():
            return True
        return ckpt_path.stat().st_mtime > output_path.stat().st_mtime

    @classmethod
    def _checkpoint_sort_key(cls, ckpt_path: Path) -> tuple[int, float]:
        step = cls._step_from_name(ckpt_path)
        if step is None:
            step = -1
        return step, ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0

    @classmethod
    def _step_from_name(cls, ckpt_path: Path) -> int | None:
        match = cls._STEP_PATTERN.search(ckpt_path.name)
        if match is None:
            return None
        return int(match.group(1))
