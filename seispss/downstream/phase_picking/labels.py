"""Dense P/S/noise label generation for phase picking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seispss.utils.phase_metadata import SEISBENCH_PHASE_METADATA_TO_LABEL

PHASE_CLASS_NAMES: tuple[str, str, str] = ("P", "S", "N")
PHASE_TO_INDEX: dict[str, int] = {"P": 0, "S": 1, "N": 2, "Noise": 2}

P_PHASE_KEYS: tuple[str, ...] = tuple(
    key for key, label in SEISBENCH_PHASE_METADATA_TO_LABEL.items() if label == "P"
)
S_PHASE_KEYS: tuple[str, ...] = tuple(
    key for key, label in SEISBENCH_PHASE_METADATA_TO_LABEL.items() if label == "S"
)
PHASE_PICK_KEYS: dict[str, tuple[str, ...]] = {
    "P": P_PHASE_KEYS,
    "S": S_PHASE_KEYS,
}


def metadata_value_to_samples(value: Any, *, sample_scale: float = 1.0) -> list[int]:
    """Convert a SeisBench metadata pick value to finite integer samples."""

    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []
    if values.size == 0:
        return []
    values = values[np.isfinite(values)]
    return [int(round(float(sample) * float(sample_scale))) for sample in values]


def _numeric_metadata_value(metadata: dict[str, Any], key: str) -> float | None:
    if key not in metadata:
        return None
    try:
        value = np.asarray(metadata[key], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if value.size == 0 or np.isnan(value).all():
        return None
    return float(np.nanmean(value))


def _metadata_sampling_rate(metadata: dict[str, Any], default_sampling_rate: float) -> float:
    sampling_rate = _numeric_metadata_value(metadata, "trace_sampling_rate_hz")
    if sampling_rate is not None and sampling_rate > 0:
        return sampling_rate
    trace_dt = _numeric_metadata_value(metadata, "trace_dt_s")
    if trace_dt is not None and trace_dt > 0:
        return 1.0 / trace_dt
    return float(default_sampling_rate)


def _sample_scale(metadata: dict[str, Any], target_sampling_rate: float | None) -> float:
    if target_sampling_rate is None:
        return 1.0
    source_sampling_rate = _metadata_sampling_rate(
        metadata,
        default_sampling_rate=float(target_sampling_rate),
    )
    return float(target_sampling_rate) / source_sampling_rate


def extract_phase_picks(
    metadata: dict[str, Any],
    *,
    trace_length: int | None = None,
    target_sampling_rate: float | None = None,
) -> dict[str, list[int]]:
    """Extract P and S picks from SeisBench metadata aliases."""

    sample_scale = _sample_scale(metadata, target_sampling_rate)
    picks: dict[str, list[int]] = {"P": [], "S": []}
    for key, phase in SEISBENCH_PHASE_METADATA_TO_LABEL.items():
        if key not in metadata:
            continue
        for sample in metadata_value_to_samples(metadata[key], sample_scale=sample_scale):
            if trace_length is not None and not 0 <= sample < int(trace_length):
                continue
            picks[phase].append(sample)
    return {phase: sorted(set(samples)) for phase, samples in picks.items()}


def first_pick_in_window(
    picks: list[int],
    *,
    length: int,
    valid_time_mask: np.ndarray | None = None,
) -> int:
    """Return the first valid pick in a window, or -1 when absent."""

    for sample in sorted(set(picks)):
        if sample < 0 or sample >= length:
            continue
        if valid_time_mask is not None and not bool(valid_time_mask[sample]):
            continue
        return int(sample)
    return -1


def gaussian_peak(*, center: int, length: int, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    positions = np.arange(length, dtype=np.float32)
    return np.exp(-0.5 * ((positions - float(center)) / float(sigma)) ** 2)


def build_phase_label(
    *,
    length: int,
    metadata: dict[str, Any],
    sigma_p: float,
    sigma_s: float,
    valid_time_mask: np.ndarray | None = None,
    target_sampling_rate: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build a soft P/S/noise label tensor with shape ``[3, length]``."""

    picks = extract_phase_picks(
        metadata,
        trace_length=length,
        target_sampling_rate=target_sampling_rate,
    )
    y = np.zeros((len(PHASE_CLASS_NAMES), length), dtype=np.float32)
    target_samples = {
        "P": first_pick_in_window(picks["P"], length=length, valid_time_mask=valid_time_mask),
        "S": first_pick_in_window(picks["S"], length=length, valid_time_mask=valid_time_mask),
    }

    for sample in picks["P"]:
        if 0 <= sample < length and (valid_time_mask is None or bool(valid_time_mask[sample])):
            y[PHASE_TO_INDEX["P"]] = np.maximum(
                y[PHASE_TO_INDEX["P"]],
                gaussian_peak(center=sample, length=length, sigma=sigma_p),
            )
    for sample in picks["S"]:
        if 0 <= sample < length and (valid_time_mask is None or bool(valid_time_mask[sample])):
            y[PHASE_TO_INDEX["S"]] = np.maximum(
                y[PHASE_TO_INDEX["S"]],
                gaussian_peak(center=sample, length=length, sigma=sigma_s),
            )

    phase_sum = y[PHASE_TO_INDEX["P"]] + y[PHASE_TO_INDEX["S"]]
    overlap = phase_sum > 1.0
    if np.any(overlap):
        y[PHASE_TO_INDEX["P"], overlap] /= phase_sum[overlap]
        y[PHASE_TO_INDEX["S"], overlap] /= phase_sum[overlap]
        phase_sum = y[PHASE_TO_INDEX["P"]] + y[PHASE_TO_INDEX["S"]]

    y[PHASE_TO_INDEX["N"]] = np.clip(1.0 - phase_sum, 0.0, 1.0)
    if valid_time_mask is not None:
        invalid = ~valid_time_mask.astype(bool)
        y[:, invalid] = 0.0
        y[PHASE_TO_INDEX["N"], invalid] = 1.0
    return y, target_samples


class PhaseLabelGenerator:
    """Generate P/S Gaussian targets and a complementary noise channel."""

    def __init__(
        self,
        *,
        sigma_p: float,
        sigma_s: float,
        sampling_rate: float,
        key: str = "X",
        label_key: str = "y",
    ) -> None:
        self.sigma_p = float(sigma_p)
        self.sigma_s = float(sigma_s)
        self.sampling_rate = float(sampling_rate)
        self.key = key
        self.label_key = label_key

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key]
        length = int(x.shape[-1])
        valid_time_mask = None
        if "valid_time_mask" in state_dict:
            valid_time_mask = np.asarray(state_dict["valid_time_mask"][0]).astype(bool)
        labels, target_samples = build_phase_label(
            length=length,
            metadata=metadata,
            sigma_p=self.sigma_p,
            sigma_s=self.sigma_s,
            valid_time_mask=valid_time_mask,
            target_sampling_rate=self.sampling_rate,
        )
        state_dict[self.label_key] = (labels, None)
        state_dict["p_pick_sample"] = (np.int64(target_samples["P"]), None)
        state_dict["s_pick_sample"] = (np.int64(target_samples["S"]), None)
        state_dict["target_p_sample"] = (np.int64(target_samples["P"]), None)
        state_dict["target_s_sample"] = (np.int64(target_samples["S"]), None)


@dataclass
class PhasePickLabeler(PhaseLabelGenerator):
    """Backward-compatible name used by the older ETHZ phase-picking code."""

    sigma_p: float
    sigma_s: float
    key: str = "X"
    label_key: str = "y"
    valid_mask_key: str = "valid_time_mask"
    sampling_rate: float | None = None

    def __post_init__(self) -> None:
        self.sigma_p = float(self.sigma_p)
        self.sigma_s = float(self.sigma_s)
        self.sampling_rate = 100.0 if self.sampling_rate is None else float(self.sampling_rate)

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key]
        valid_time_mask = None
        if self.valid_mask_key in state_dict:
            valid_time_mask = np.asarray(state_dict[self.valid_mask_key][0]).astype(bool)
        labels, target_samples = build_phase_label(
            length=int(x.shape[-1]),
            metadata=metadata,
            sigma_p=self.sigma_p,
            sigma_s=self.sigma_s,
            valid_time_mask=valid_time_mask,
            target_sampling_rate=self.sampling_rate,
        )
        state_dict[self.label_key] = (labels.astype(np.float32, copy=False), None)
        state_dict["p_pick_sample"] = (np.int64(target_samples["P"]), None)
        state_dict["s_pick_sample"] = (np.int64(target_samples["S"]), None)
        state_dict["target_p_sample"] = (np.int64(target_samples["P"]), None)
        state_dict["target_s_sample"] = (np.int64(target_samples["S"]), None)
