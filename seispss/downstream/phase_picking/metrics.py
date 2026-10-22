"""Peak-based metrics for dense phase-picking outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class PhaseMetricState:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    errors: list[float] = field(default_factory=list)

    def update(self, *, predictions: list[int], true_pick: int, tolerance_samples: int) -> None:
        if true_pick < 0:
            self.fp += len(predictions)
            return
        if not predictions:
            self.fn += 1
            return

        distances = [abs(int(prediction) - int(true_pick)) for prediction in predictions]
        best_idx = int(np.argmin(distances))
        best_distance = int(distances[best_idx])
        if best_distance <= int(tolerance_samples):
            self.tp += 1
            self.fp += max(len(predictions) - 1, 0)
            self.errors.append(float(best_distance))
        else:
            self.fp += len(predictions)
            self.fn += 1

    def compute(self) -> dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        mae = float(np.mean(self.errors)) if self.errors else 0.0
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mae_samples": mae,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
        }


class PhasePickingMetrics:
    """Accumulate P and S peak metrics over an epoch."""

    def __init__(
        self,
        *,
        threshold_p: float,
        threshold_s: float,
        tolerance_samples: int,
        min_peak_distance_samples: int,
        sampling_rate: float,
    ) -> None:
        self.thresholds = {"p": float(threshold_p), "s": float(threshold_s)}
        self.tolerance_samples = int(tolerance_samples)
        self.min_peak_distance_samples = int(min_peak_distance_samples)
        self.sampling_rate = float(sampling_rate)
        self.reset()

    def reset(self) -> None:
        self.states = {
            "p": PhaseMetricState(),
            "s": PhaseMetricState(),
        }

    @torch.no_grad()
    def update(self, *, logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> None:
        probabilities = torch.softmax(logits.detach(), dim=1).float().cpu().numpy()
        p_true = _batch_pick_values(batch, "p_pick_sample", "target_p_sample")
        s_true = _batch_pick_values(batch, "s_pick_sample", "target_s_sample")
        valid_time_mask = None
        if batch.get("valid_time_mask") is not None:
            valid_time_mask = batch["valid_time_mask"].detach().bool().cpu().numpy()
        for batch_idx in range(probabilities.shape[0]):
            for phase, class_idx, true_picks in (
                ("p", 0, p_true),
                ("s", 1, s_true),
            ):
                phase_values = probabilities[batch_idx, class_idx].copy()
                if valid_time_mask is not None:
                    phase_values[~valid_time_mask[batch_idx]] = 0.0
                predictions = find_peaks_1d(
                    phase_values,
                    threshold=self.thresholds[phase],
                    min_distance=self.min_peak_distance_samples,
                )
                self.states[phase].update(
                    predictions=predictions,
                    true_pick=int(true_picks[batch_idx]),
                    tolerance_samples=self.tolerance_samples,
                )

    def compute(self) -> dict[str, float]:
        p_metrics = self.states["p"].compute()
        s_metrics = self.states["s"].compute()
        all_errors = self.states["p"].errors + self.states["s"].errors
        mae_samples = float(np.mean(all_errors)) if all_errors else 0.0
        metrics = {
            "precision": 0.5 * (p_metrics["precision"] + s_metrics["precision"]),
            "recall": 0.5 * (p_metrics["recall"] + s_metrics["recall"]),
            "f1": 0.5 * (p_metrics["f1"] + s_metrics["f1"]),
            "mae_samples": mae_samples,
            "mae_seconds": mae_samples / self.sampling_rate,
        }
        for phase, phase_metrics in (("p", p_metrics), ("s", s_metrics)):
            for name, value in phase_metrics.items():
                metrics[f"{phase}/{name}"] = value
            metrics[f"{phase}/mae_seconds"] = phase_metrics["mae_samples"] / self.sampling_rate
        return metrics


def find_peaks_1d(
    values: np.ndarray,
    *,
    threshold: float,
    min_distance: int,
) -> list[int]:
    """Find thresholded local maxima with score-ordered non-maximum suppression."""

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return []

    candidates = values >= float(threshold)
    if values.size > 1:
        left = np.r_[values[0], values[:-1]]
        right = np.r_[values[1:], values[-1]]
        candidates &= (values >= left) & (values >= right)
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size == 0:
        return []

    order = candidate_indices[np.argsort(values[candidate_indices])[::-1]]
    selected: list[int] = []
    min_distance = max(int(min_distance), 1)
    for idx in order:
        idx = int(idx)
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)
    return sorted(selected)


def _batch_pick_values(
    batch: dict[str, torch.Tensor],
    primary_key: str,
    fallback_key: str,
) -> np.ndarray:
    if primary_key in batch:
        values = batch[primary_key]
    else:
        values = batch[fallback_key]
    return values.detach().cpu().numpy().astype(np.int64)
