"""Window selectors for dense phase-picking supervision."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from seispss.utils.phase_metadata import SEISBENCH_PHASE_METADATA_TO_LABEL


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _as_rng(deterministic: bool, *seed_parts: Any) -> np.random.Generator:
    if deterministic:
        return np.random.default_rng(_stable_seed(*seed_parts))
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)


def _as_key_pair(key: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(key, str):
        return (key, key)
    return key


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


def _sample_scale(metadata: dict[str, Any], target_sampling_rate: float) -> float:
    source_sampling_rate = _metadata_sampling_rate(
        metadata,
        default_sampling_rate=target_sampling_rate,
    )
    return float(target_sampling_rate) / source_sampling_rate


def extract_phase_picks(
    metadata: dict[str, Any],
    *,
    trace_length: int,
    target_sampling_rate: float,
) -> dict[str, list[int]]:
    """Return P/S pick samples after resampling to the target rate."""

    scale = _sample_scale(metadata=metadata, target_sampling_rate=target_sampling_rate)
    picks: dict[str, list[int]] = {"P": [], "S": []}
    for key, label in SEISBENCH_PHASE_METADATA_TO_LABEL.items():
        raw_pick = _numeric_metadata_value(metadata, key)
        if raw_pick is None:
            continue
        pick = int(round(raw_pick * scale))
        if 0 <= pick < trace_length:
            picks.setdefault(label, []).append(pick)
    return {label: sorted(set(label_picks)) for label, label_picks in picks.items()}


def _random_start(*, trace_length: int, window_length: int, rng: np.random.Generator) -> int:
    max_start = trace_length - window_length
    if max_start < 0:
        raise ValueError(
            f"Trace length {trace_length} is shorter than requested window length {window_length}."
        )
    if max_start == 0:
        return 0
    return int(rng.integers(0, max_start + 1))


def _slice_without_padding(*, x: np.ndarray, start: int, window_length: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got {x.shape}.")
    end = start + window_length
    if start < 0 or end > x.shape[-1]:
        raise ValueError(
            f"Window [{start}, {end}) exceeds trace length {x.shape[-1]}; no-padding crop is required."
        )
    return x[:, start:end].copy()


def _shift_sample_metadata(
    *,
    metadata: dict[str, Any],
    start: int,
    target_sampling_rate: float,
) -> None:
    scale = _sample_scale(metadata=metadata, target_sampling_rate=target_sampling_rate)
    source_start = float(start) / scale
    for key, value in list(metadata.items()):
        if not key.endswith("_sample"):
            continue
        try:
            metadata[key] = value - source_start
        except TypeError:
            continue


def _write_window_state(
    state_dict: dict[str, Any],
    *,
    key: tuple[str, str],
    x: np.ndarray,
    metadata: dict[str, Any],
    start: int,
    window_length: int,
    target_sampling_rate: float,
) -> None:
    window = _slice_without_padding(x=x, start=start, window_length=window_length)
    _shift_sample_metadata(
        metadata=metadata,
        start=start,
        target_sampling_rate=target_sampling_rate,
    )
    state_dict[key[1]] = (window, metadata)
    state_dict["valid_time_mask"] = (np.ones(window.shape[-1], dtype=np.float32), None)
    state_dict["window_start_sample"] = (np.int64(start), None)
    state_dict["window_end_sample"] = (np.int64(start + window.shape[-1]), None)


def _sample_pick_conditioned_start(
    *,
    pick: int,
    trace_length: int,
    window_length: int,
    relative_low: int,
    relative_high: int,
    rng: np.random.Generator,
) -> int | None:
    max_start = trace_length - window_length
    if max_start < 0:
        raise ValueError(
            f"Trace length {trace_length} is shorter than requested window length {window_length}."
        )
    start_min = max(0, pick - relative_high)
    start_max = min(max_start, pick - relative_low)
    if start_min > start_max:
        return None
    if start_min == start_max:
        return int(start_min)
    return int(rng.integers(start_min, start_max + 1))


class PhaseWindowSelector:
    """Sample fixed-length windows around P/S picks or uniformly at random."""

    def __init__(
        self,
        *,
        window_length: int,
        sampling_rate: float,
        p_probability: float = 0.4,
        s_probability: float = 0.4,
        random_probability: float = 0.2,
        min_pick_position_fraction: float = 0.3,
        max_pick_position_fraction: float = 0.7,
        deterministic: bool = False,
        key: str | tuple[str, str] = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        self.min_pick_position = int(round(float(min_pick_position_fraction) * self.window_length))
        self.max_pick_position = int(round(float(max_pick_position_fraction) * self.window_length))
        if not 0 <= self.min_pick_position <= self.max_pick_position <= self.window_length:
            raise ValueError("Pick position fractions must satisfy 0 <= min <= max <= 1.")
        probabilities = np.asarray(
            [p_probability, s_probability, random_probability],
            dtype=np.float64,
        )
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("Window sampling probabilities must be non-negative and sum positive.")
        self.probabilities = probabilities / probabilities.sum()
        self.deterministic = bool(deterministic)

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = dict(metadata)
        trace_name = metadata.get("trace_name", "unknown_trace")
        if x.shape[-1] < self.window_length:
            raise ValueError(
                f"Trace '{trace_name}' length {x.shape[-1]} is shorter than requested "
                f"window length {self.window_length}."
            )
        rng = _as_rng(self.deterministic, "phase_window", trace_name, self.window_length)
        phase_picks = extract_phase_picks(
            metadata,
            trace_length=x.shape[-1],
            target_sampling_rate=self.sampling_rate,
        )
        mode = str(rng.choice(["P", "S", "random"], p=self.probabilities))
        start = self._phase_start(mode=mode, phase_picks=phase_picks, trace_length=x.shape[-1], rng=rng)
        if start is None:
            start = _random_start(trace_length=x.shape[-1], window_length=self.window_length, rng=rng)
        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=self.window_length,
            target_sampling_rate=self.sampling_rate,
        )

    def _phase_start(
        self,
        *,
        mode: str,
        phase_picks: dict[str, list[int]],
        trace_length: int,
        rng: np.random.Generator,
    ) -> int | None:
        if mode not in {"P", "S"}:
            return None
        picks = phase_picks.get(mode, [])
        if not picks:
            return None
        pick = int(rng.choice(picks))
        return _sample_pick_conditioned_start(
            pick=pick,
            trace_length=trace_length,
            window_length=self.window_length,
            relative_low=self.min_pick_position,
            relative_high=self.max_pick_position,
            rng=rng,
        )


class PLeadWindowSelector:
    """Place the P arrival a configured number of seconds after the window start."""

    def __init__(
        self,
        *,
        window_length: int,
        sampling_rate: float,
        min_p_lead_seconds: float,
        max_p_lead_seconds: float,
        deterministic: bool = False,
        key: str | tuple[str, str] = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        self.min_p_lead = int(round(float(min_p_lead_seconds) * self.sampling_rate))
        self.max_p_lead = int(round(float(max_p_lead_seconds) * self.sampling_rate))
        if not 0 <= self.min_p_lead <= self.max_p_lead <= self.window_length:
            raise ValueError("P-lead seconds must place P within the window.")
        self.deterministic = bool(deterministic)

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = dict(metadata)
        trace_name = metadata.get("trace_name", "unknown_trace")
        if x.shape[-1] < self.window_length:
            raise ValueError(
                f"Trace '{trace_name}' length {x.shape[-1]} is shorter than requested "
                f"window length {self.window_length}."
            )
        rng = _as_rng(self.deterministic, "p_lead_window", trace_name, self.window_length)
        phase_picks = extract_phase_picks(
            metadata,
            trace_length=x.shape[-1],
            target_sampling_rate=self.sampling_rate,
        )
        start = None
        if phase_picks.get("P"):
            pick = int(rng.choice(phase_picks["P"]))
            start = _sample_pick_conditioned_start(
                pick=pick,
                trace_length=x.shape[-1],
                window_length=self.window_length,
                relative_low=self.min_p_lead,
                relative_high=self.max_p_lead,
                rng=rng,
            )
        if start is None:
            start = _random_start(trace_length=x.shape[-1], window_length=self.window_length, rng=rng)
        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=self.window_length,
            target_sampling_rate=self.sampling_rate,
        )


class EvalWindowSelector:
    """Build deterministic evaluation windows."""

    def __init__(
        self,
        *,
        window_length: int,
        sampling_rate: float,
        strategy: str = "full",
        key: str | tuple[str, str] = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        self.strategy = str(strategy)

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = dict(metadata)
        trace_length = int(x.shape[-1])
        if trace_length < self.window_length:
            raise ValueError(
                f"Trace length {trace_length} is shorter than requested window length {self.window_length}."
            )
        if self.strategy == "full":
            start = 0
            window_length = trace_length
        elif self.strategy == "prefix":
            start = 0
            window_length = self.window_length
        elif self.strategy == "center":
            start = (trace_length - self.window_length) // 2
            window_length = self.window_length
        else:
            raise ValueError(f"Unsupported evaluation window strategy: {self.strategy}")
        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=window_length,
            target_sampling_rate=self.sampling_rate,
        )
