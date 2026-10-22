"""Custom data augmentations for SeisPSS pretraining."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np

from seispss.data.event_metadata import MISSING_UID, event_station_uids
from seispss.utils.phase_metadata import SEISBENCH_PHASE_METADATA_TO_LABEL

WINDOW_SOURCE_NAMES = {
    0: "event_random",
    1: "phase_containing",
    2: "noise_random",
}

WINDOW_SOURCE_IDS = {name: idx for idx, name in WINDOW_SOURCE_NAMES.items()}

SAMPLE_TYPE_NAMES = {
    0: "event",
    1: "noise",
}

SAMPLE_TYPE_IDS = {name: idx for idx, name in SAMPLE_TYPE_NAMES.items()}


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _as_rng(deterministic: bool, *seed_parts: Any) -> np.random.Generator:
    if deterministic:
        return np.random.default_rng(_stable_seed(*seed_parts))
    seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)


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
    source_sampling_rate = _metadata_sampling_rate(metadata, default_sampling_rate=target_sampling_rate)
    return float(target_sampling_rate) / source_sampling_rate


def _extract_phase_picks(
    metadata: dict[str, Any],
    *,
    trace_length: int,
    target_sampling_rate: float,
) -> dict[str, list[int]]:
    sample_scale = _sample_scale(metadata=metadata, target_sampling_rate=target_sampling_rate)
    picks: dict[str, list[int]] = {"P": [], "S": []}
    for key, label in SEISBENCH_PHASE_METADATA_TO_LABEL.items():
        raw_pick = _numeric_metadata_value(metadata, key)
        if raw_pick is None:
            continue
        pick = int(round(raw_pick * sample_scale))
        if 0 <= pick < trace_length:
            picks.setdefault(label, []).append(pick)
    return {label: sorted(set(label_picks)) for label, label_picks in picks.items()}


def _flatten_picks(phase_picks: dict[str, list[int]]) -> list[int]:
    return sorted(set(pick for label_picks in phase_picks.values() for pick in label_picks))


def _is_noise_trace(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("trace_category", "")).strip().lower() == "noise"


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
    sample_scale = _sample_scale(metadata=metadata, target_sampling_rate=target_sampling_rate)
    source_start = start / sample_scale
    for key, value in metadata.items():
        if key.endswith("_sample"):
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
    source_id: int,
    sample_type_id: int,
    trace_has_pick: bool,
    target_sampling_rate: float,
) -> None:
    window = _slice_without_padding(x=x, start=start, window_length=window_length)
    _shift_sample_metadata(
        metadata=metadata,
        start=start,
        target_sampling_rate=target_sampling_rate,
    )
    state_dict[key[1]] = (window, metadata)
    state_dict["window_source"] = (np.int64(source_id), None)
    state_dict["sample_type"] = (np.int64(sample_type_id), None)
    state_dict["trace_has_pick"] = (np.int64(trace_has_pick), None)
    state_dict["valid_time_mask"] = (np.ones(window_length, dtype=np.float32), None)
    state_dict["window_start_sample"] = (np.int64(start), None)
    state_dict["window_end_sample"] = (np.int64(start + window_length), None)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _sample_from_intervals(
    intervals: list[tuple[int, int]],
    *,
    rng: np.random.Generator,
) -> int | None:
    merged = _merge_intervals(intervals)
    if not merged:
        return None
    interval_lengths = np.asarray([end - start + 1 for start, end in merged], dtype=np.int64)
    total_length = int(interval_lengths.sum())
    offset = int(rng.integers(0, total_length))
    for (start, end), interval_length in zip(merged, interval_lengths):
        if offset < interval_length:
            return int(start + offset)
        offset -= int(interval_length)
    return int(merged[-1][1])


def _as_key_pair(key: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(key, str):
        return (key, key)
    return key


class EventWindowSelector:
    """Select no-padding pretraining windows from event traces."""

    def __init__(
        self,
        *,
        window_length: int,
        random_probability: float,
        phase_containing_probability: float,
        pick_margin_sec: tuple[float, float] | list[float],
        sampling_rate: float,
        deterministic: bool = False,
        key: str = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        total = random_probability + phase_containing_probability
        if total <= 0:
            raise ValueError("Event window probabilities must sum to a positive value.")
        self.probabilities = np.array(
            [random_probability, phase_containing_probability],
            dtype=np.float64,
        )
        self.probabilities /= self.probabilities.sum()
        if len(pick_margin_sec) != 2:
            raise ValueError("pick_margin_sec must contain [low_sec, high_sec].")
        self.pick_margin_low = int(round(float(pick_margin_sec[0]) * self.sampling_rate))
        self.pick_margin_high = int(round(float(pick_margin_sec[1]) * self.sampling_rate))
        if not 0 <= self.pick_margin_low <= self.pick_margin_high <= self.window_length:
            raise ValueError(
                "pick_margin_sec must satisfy 0 <= low <= high <= window length."
            )
        self.deterministic = deterministic

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = copy.deepcopy(metadata)

        trace_name = metadata.get("trace_name", "unknown_trace")
        self._validate_trace_length(trace_name=trace_name, trace_length=x.shape[-1])
        rng = _as_rng(self.deterministic, "event_window", trace_name, self.window_length)
        phase_picks = _extract_phase_picks(
            metadata,
            trace_length=x.shape[-1],
            target_sampling_rate=self.sampling_rate,
        )
        valid_picks = _flatten_picks(phase_picks)

        should_sample_phase = bool(rng.choice([False, True], p=self.probabilities))
        source_id = WINDOW_SOURCE_IDS["event_random"]
        start = None
        if should_sample_phase:
            start = self._sample_phase_containing_start(
                trace_length=x.shape[-1],
                valid_picks=valid_picks,
                rng=rng,
            )
            if start is not None:
                source_id = WINDOW_SOURCE_IDS["phase_containing"]
        if start is None:
            start = _random_start(
                trace_length=x.shape[-1],
                window_length=self.window_length,
                rng=rng,
            )

        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=self.window_length,
            source_id=source_id,
            sample_type_id=SAMPLE_TYPE_IDS["event"],
            trace_has_pick=bool(valid_picks),
            target_sampling_rate=self.sampling_rate,
        )

    def _sample_phase_containing_start(
        self,
        *,
        trace_length: int,
        valid_picks: list[int],
        rng: np.random.Generator,
    ) -> int | None:
        max_start = trace_length - self.window_length
        if max_start < 0:
            raise ValueError(
                f"Trace length {trace_length} is shorter than requested window length {self.window_length}."
            )
        intervals = []
        for pick in valid_picks:
            start_min = max(0, pick - self.pick_margin_high)
            start_max = min(max_start, pick - self.pick_margin_low)
            if start_min <= start_max:
                intervals.append((start_min, start_max))
        return _sample_from_intervals(intervals, rng=rng)

    def _validate_trace_length(self, *, trace_name: Any, trace_length: int) -> None:
        if trace_length < self.window_length:
            raise ValueError(
                f"Trace '{trace_name}' length {trace_length} is shorter than requested "
                f"window length {self.window_length}. This sample should have been filtered before training."
            )


class NoiseWindowSelector:
    """Select no-padding random pretraining windows from noise traces."""

    def __init__(
        self,
        *,
        window_length: int,
        sampling_rate: float,
        deterministic: bool = False,
        key: str = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        self.deterministic = deterministic

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = copy.deepcopy(metadata)
        trace_name = metadata.get("trace_name", "unknown_trace")
        if x.shape[-1] < self.window_length:
            raise ValueError(
                f"Trace '{trace_name}' length {x.shape[-1]} is shorter than requested "
                f"window length {self.window_length}. This noise sample should have been filtered before training."
            )
        rng = _as_rng(self.deterministic, "noise_window", trace_name, self.window_length)
        phase_picks = _extract_phase_picks(
            metadata,
            trace_length=x.shape[-1],
            target_sampling_rate=self.sampling_rate,
        )
        start = _random_start(
            trace_length=x.shape[-1],
            window_length=self.window_length,
            rng=rng,
        )
        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=self.window_length,
            source_id=WINDOW_SOURCE_IDS["noise_random"],
            sample_type_id=SAMPLE_TYPE_IDS["noise"],
            trace_has_pick=bool(_flatten_picks(phase_picks)),
            target_sampling_rate=self.sampling_rate,
        )


class PretrainValidationWindowSelector:
    """Select deterministic no-padding validation windows."""

    def __init__(
        self,
        *,
        window_length: int,
        sampling_rate: float,
        p_position_samples: int,
        s_position_samples: int,
        deterministic_manifest: bool = True,
        key: str = "X",
    ) -> None:
        self.key = _as_key_pair(key)
        self.window_length = int(window_length)
        self.sampling_rate = float(sampling_rate)
        self.p_position_samples = int(p_position_samples)
        self.s_position_samples = int(s_position_samples)
        self.deterministic_manifest = deterministic_manifest

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        metadata = copy.deepcopy(metadata)
        trace_name = metadata.get("trace_name", "unknown_trace")
        if x.shape[-1] < self.window_length:
            raise ValueError(
                f"Trace '{trace_name}' length {x.shape[-1]} is shorter than requested "
                f"window length {self.window_length}. This validation sample should have been filtered before validation."
            )
        rng = _as_rng(
            self.deterministic_manifest,
            "pretrain_val",
            trace_name,
            self.window_length,
        )
        phase_picks = _extract_phase_picks(
            metadata,
            trace_length=x.shape[-1],
            target_sampling_rate=self.sampling_rate,
        )
        max_start = x.shape[-1] - self.window_length
        if max_start < 0:
            raise ValueError(
                f"Trace length {x.shape[-1]} is shorter than requested window length {self.window_length}."
            )

        source_id = WINDOW_SOURCE_IDS["event_random"]
        sample_type_id = SAMPLE_TYPE_IDS["event"]
        if _is_noise_trace(metadata):
            start = _random_start(
                trace_length=x.shape[-1],
                window_length=self.window_length,
                rng=rng,
            )
            source_id = WINDOW_SOURCE_IDS["noise_random"]
            sample_type_id = SAMPLE_TYPE_IDS["noise"]
        elif phase_picks.get("P"):
            start = int(np.clip(phase_picks["P"][0] - self.p_position_samples, 0, max_start))
            source_id = WINDOW_SOURCE_IDS["phase_containing"]
        elif phase_picks.get("S"):
            start = int(np.clip(phase_picks["S"][0] - self.s_position_samples, 0, max_start))
            source_id = WINDOW_SOURCE_IDS["phase_containing"]
        else:
            start = _random_start(
                trace_length=x.shape[-1],
                window_length=self.window_length,
                rng=rng,
            )

        _write_window_state(
            state_dict,
            key=self.key,
            x=x,
            metadata=metadata,
            start=start,
            window_length=self.window_length,
            source_id=source_id,
            sample_type_id=sample_type_id,
            trace_has_pick=bool(_flatten_picks(phase_picks)),
            target_sampling_rate=self.sampling_rate,
        )


class CaptureValidComponentMask:
    """Expose which waveform components contain non-zero signal."""

    def __init__(self, key: str = "X") -> None:
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, _ = state_dict[self.key[0]]
        valid_components = (np.std(x, axis=-1) > 0).astype(np.float32)
        state_dict["valid_component_mask"] = (valid_components, None)


class AddPseudoArrivalMetadata:
    """Expose crop-local pseudo-arrival anchor metadata as batch tensors."""

    def __init__(
        self,
        *,
        window_length: int,
        waveform_key: str = "X",
    ) -> None:
        self.window_length = int(window_length)
        self.waveform_key = waveform_key

    def __call__(self, state_dict: dict[str, Any]) -> None:
        metadata: dict[str, Any] = {}
        if self.waveform_key in state_dict:
            _, metadata = state_dict[self.waveform_key]

        sample = _numeric_metadata_value(metadata, "pseudo_anchor_sample")
        confidence = _numeric_metadata_value(metadata, "pseudo_anchor_confidence")
        z_peak = _numeric_metadata_value(metadata, "pseudo_anchor_z_peak")
        persistence = _numeric_metadata_value(metadata, "pseudo_anchor_persistence")
        status = _numeric_metadata_value(metadata, "pseudo_anchor_status")
        valid = self._metadata_bool(metadata.get("pseudo_anchor_valid", False))

        is_in_window = sample is not None and 0 <= int(round(sample)) < self.window_length
        if not valid or not is_in_window:
            state_dict["pseudo_anchor_sample"] = (np.int64(-1), None)
            state_dict["pseudo_anchor_confidence"] = (np.float32(0.0), None)
            state_dict["pseudo_anchor_valid"] = (np.int64(0), None)
        else:
            state_dict["pseudo_anchor_sample"] = (np.int64(round(float(sample))), None)
            state_dict["pseudo_anchor_confidence"] = (
                np.float32(max(float(confidence or 0.0), 0.0)),
                None,
            )
            state_dict["pseudo_anchor_valid"] = (np.int64(1), None)

        state_dict["pseudo_anchor_z_peak"] = (np.float32(float(z_peak or 0.0)), None)
        state_dict["pseudo_anchor_persistence"] = (np.float32(float(persistence or 0.0)), None)
        state_dict["pseudo_anchor_status"] = (np.int64(int(status or -1)), None)

    @staticmethod
    def _metadata_bool(value: Any) -> bool:
        try:
            return bool(np.asarray(value).astype(bool).item())
        except (TypeError, ValueError):
            return False


class FillMissingComponents:
    """Copy valid channels over missing ones to avoid dead-channel artifacts."""

    def __init__(self, key: str = "X", deterministic: bool = False) -> None:
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.deterministic = deterministic

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        trace_name = metadata.get("trace_name", "unknown_trace")
        window_start_sample = int(state_dict.get("window_start_sample", (0, None))[0])
        state_dict[self.key[1]] = (
            self._fill_missing_component(
                x,
                trace_name=trace_name,
                window_start_sample=window_start_sample,
            ),
            metadata,
        )

    def _fill_missing_component(
        self,
        x: np.ndarray,
        *,
        trace_name: str,
        window_start_sample: int,
    ) -> np.ndarray:
        std_devs = np.std(x, axis=-1)
        zero_std_rows = np.where(std_devs == 0)[0]
        non_zero_std_rows = np.where(std_devs != 0)[0]

        if len(zero_std_rows) == 0 or len(non_zero_std_rows) == 0:
            return x

        x = x.copy()
        rng = _as_rng(
            self.deterministic,
            trace_name,
            window_start_sample,
            tuple(zero_std_rows.tolist()),
            tuple(non_zero_std_rows.tolist()),
        )
        for row_idx in zero_std_rows:
            replacement_row = int(rng.choice(non_zero_std_rows))
            x[row_idx] = x[replacement_row]
        return x


class StdSafeNormalize:
    """Normalize waveforms using only valid time steps when available."""

    def __init__(
        self,
        *,
        amp_norm_type: str = "std",
        eps: float = 1e-6,
        key: str = "X",
        valid_mask_key: str = "valid_time_mask",
    ) -> None:
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.amp_norm_type = amp_norm_type
        self.eps = eps
        self.valid_mask_key = valid_mask_key

    def __call__(self, state_dict: dict[str, Any]) -> None:
        x, metadata = state_dict[self.key[0]]
        valid_mask = None
        if self.valid_mask_key in state_dict:
            valid_mask = np.asarray(state_dict[self.valid_mask_key][0]).astype(bool)
        state_dict[self.key[1]] = (self._normalize(x, valid_mask), metadata)

    def _normalize(
        self,
        x: np.ndarray,
        valid_mask: np.ndarray | None,
    ) -> np.ndarray:
        x = x.astype(np.float32, copy=True)
        if valid_mask is None or bool(np.all(valid_mask)):
            x -= x.mean(axis=-1, keepdims=True)
            if self.amp_norm_type == "peak":
                amp = np.max(np.abs(x), axis=-1, keepdims=True)
            elif self.amp_norm_type == "std":
                amp = x.std(axis=-1, keepdims=True)
            else:
                raise ValueError(f"Unsupported amp_norm_type: {self.amp_norm_type}")
            amp = np.where(amp == 0, 1.0, amp)
            return x / (amp + self.eps)

        mask = np.broadcast_to(valid_mask[None, :], x.shape).astype(np.float32)
        count = np.maximum(mask.sum(axis=-1, keepdims=True), 1.0)
        mean = (x * mask).sum(axis=-1, keepdims=True) / count
        x = x - mean

        if self.amp_norm_type == "peak":
            amp = np.max(np.abs(x) * mask, axis=-1, keepdims=True)
        elif self.amp_norm_type == "std":
            amp = np.sqrt(((x**2) * mask).sum(axis=-1, keepdims=True) / count)
        else:
            raise ValueError(f"Unsupported amp_norm_type: {self.amp_norm_type}")

        amp = np.where(amp == 0, 1.0, amp)
        normalized = x / (amp + self.eps)
        return normalized * mask


class ChangeDtype:
    """Cast selected numeric entries to a target dtype."""

    def __init__(self, dtype: np.dtype = np.float32, keys: tuple[str, ...] = ("X", "valid_time_mask")) -> None:
        self.dtype = dtype
        self.keys = keys

    def __call__(self, state_dict: dict[str, Any]) -> None:
        for key in self.keys:
            if key not in state_dict:
                continue
            value, metadata = state_dict[key]
            if isinstance(value, np.ndarray):
                state_dict[key] = (value.astype(self.dtype, copy=False), metadata)


class AddConstantMetadata:
    """Expose constant per-generator metadata as model-consumable data."""

    def __init__(self, key: str, value: int) -> None:
        self.key = key
        self.value = value

    def __call__(self, state_dict: dict[str, Any]) -> None:
        state_dict[self.key] = (np.int64(self.value), None)


class AddEventMetadata:
    """Expose stable event and station identifiers for metadata-level objectives."""

    def __init__(
        self,
        *,
        dataset_name: str,
        event_key: str = "event_uid",
        station_key: str = "station_uid",
        waveform_key: str = "X",
    ) -> None:
        self.dataset_name = dataset_name
        self.event_key = event_key
        self.station_key = station_key
        self.waveform_key = waveform_key

    def __call__(self, state_dict: dict[str, Any]) -> None:
        if self.waveform_key not in state_dict:
            state_dict[self.event_key] = (MISSING_UID, None)
            state_dict[self.station_key] = (MISSING_UID, None)
            return

        _, metadata = state_dict[self.waveform_key]
        event_uid, station_uid = event_station_uids(metadata, dataset_name=self.dataset_name)
        state_dict[self.event_key] = (event_uid, None)
        state_dict[self.station_key] = (station_uid, None)


def _colored_noise(
    rng: np.random.Generator,
    length: int,
    *,
    exponent: float,
) -> np.ndarray:
    """Unit-std noise with power spectral density ~ 1/f**exponent (0=white, 1≈pink)."""
    white = rng.standard_normal(length).astype(np.float64)
    if exponent <= 0.0 or length < 4:
        std = float(white.std())
        return (white / std).astype(np.float32) if std > 0 else white.astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = freqs[nonzero] ** (-exponent / 2.0)
    scale[0] = 0.0  # drop DC component
    shaped = np.fft.irfft(spectrum * scale, n=length)
    std = float(shaped.std())
    if std <= 0:
        return np.zeros(length, dtype=np.float32)
    return (shaped / std).astype(np.float32)


class WaveformAugment:
    """Train-time, label-preserving 3C waveform augmentation.

    Applied AFTER ``FillMissingComponents`` and BEFORE per-channel std-normalization.
    Only the waveform *values* in ``X`` are modified -- never the time axis or the
    metadata pick fields -- so the P/S Gaussian targets built downstream are
    unchanged. Because normalization is per-channel std, amplitude/gain scaling is
    absorbed and is therefore intentionally NOT provided here; the augmentations
    below alter waveform shape / SNR / channel presence / polarity, which survive it.

    Sub-augmentations (each gated by its own probability):
      * ``polarity_flip``: global sign flip of all components (preserves 3C structure).
      * ``add_noise``: additive colored noise scaled to a random target SNR, per channel.
      * ``time_mask``: zero short random segments, guarded away from the P/S picks.
      * ``channel_dropout``: zero one valid component (never the last), also clearing
        it in ``valid_component_mask``.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any] | None,
        sampling_rate: float,
        deterministic: bool = False,
        key: str = "X",
        valid_component_mask_key: str = "valid_component_mask",
    ) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.sampling_rate = float(sampling_rate)
        self.deterministic = bool(deterministic)
        self.key = key
        self.valid_component_mask_key = valid_component_mask_key

        noise = dict(cfg.get("add_noise", {}) or {})
        self._noise_prob = float(noise.get("prob", 0.0))
        snr = noise.get("snr_db", [5.0, 25.0])
        self._noise_snr_lo, self._noise_snr_hi = float(snr[0]), float(snr[1])
        self._noise_exponent = 1.0 if str(noise.get("color", "pink")) == "pink" else 0.0

        flip = dict(cfg.get("polarity_flip", {}) or {})
        self._flip_prob = float(flip.get("prob", 0.0))

        drop = dict(cfg.get("channel_dropout", {}) or {})
        self._drop_prob = float(drop.get("prob", 0.0))

        tmask = dict(cfg.get("time_mask", {}) or {})
        self._tmask_prob = float(tmask.get("prob", 0.0))
        self._tmask_max_frac = float(tmask.get("max_frac", 0.1))
        nrange = tmask.get("n_masks", [1, 2])
        self._tmask_n_lo, self._tmask_n_hi = int(nrange[0]), int(nrange[1])
        self._tmask_guard = int(round(float(tmask.get("guard_seconds", 0.5)) * self.sampling_rate))

    def __call__(self, state_dict: dict[str, Any]) -> None:
        if not self.enabled or self.key not in state_dict:
            return
        x, metadata = state_dict[self.key]
        x = np.asarray(x, dtype=np.float32).copy()
        metadata = metadata or {}
        rng = _as_rng(self.deterministic, metadata.get("trace_name", "unknown"), "waveform_augment")

        if self._flip_prob > 0.0 and rng.random() < self._flip_prob:
            x = -x
        if self._noise_prob > 0.0 and rng.random() < self._noise_prob:
            x = self._add_noise(x, rng)
        if self._tmask_prob > 0.0 and rng.random() < self._tmask_prob:
            x = self._time_mask(x, rng, metadata)
        dropped: int | None = None
        if self._drop_prob > 0.0 and rng.random() < self._drop_prob:
            x, dropped = self._channel_dropout(x, rng)

        state_dict[self.key] = (x, metadata)
        if dropped is not None and self.valid_component_mask_key in state_dict:
            vcm, vmeta = state_dict[self.valid_component_mask_key]
            vcm = np.asarray(vcm, dtype=np.float32).copy()
            if dropped < vcm.shape[0]:
                vcm[dropped] = 0.0
            state_dict[self.valid_component_mask_key] = (vcm, vmeta)

    def _add_noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        snr_db = float(rng.uniform(self._noise_snr_lo, self._noise_snr_hi))
        ratio = 10.0 ** (-snr_db / 20.0)
        length = x.shape[-1]
        for c in range(x.shape[0]):
            sig_rms = float(np.sqrt(np.mean(x[c] ** 2)))
            if sig_rms <= 0.0:
                continue
            noise = _colored_noise(rng, length, exponent=self._noise_exponent)
            x[c] = x[c] + (sig_rms * ratio) * noise
        return x

    def _time_mask(
        self, x: np.ndarray, rng: np.random.Generator, metadata: dict[str, Any]
    ) -> np.ndarray:
        length = x.shape[-1]
        max_len = max(1, int(round(self._tmask_max_frac * length)))
        picks = _flatten_picks(
            _extract_phase_picks(
                metadata, trace_length=length, target_sampling_rate=self.sampling_rate
            )
        )
        n = (
            int(rng.integers(self._tmask_n_lo, self._tmask_n_hi + 1))
            if self._tmask_n_hi >= self._tmask_n_lo
            else 1
        )
        for _ in range(n):
            seg = int(rng.integers(1, max_len + 1))
            start = int(rng.integers(0, max(1, length - seg + 1)))
            end = start + seg
            if any(
                start < pick + self._tmask_guard and end > pick - self._tmask_guard
                for pick in picks
            ):
                continue  # keep a guard band around P/S picks
            x[:, start:end] = 0.0
        return x

    def _channel_dropout(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, int | None]:
        valid = [c for c in range(x.shape[0]) if float(np.std(x[c])) > 0.0]
        if len(valid) <= 1:
            return x, None  # never drop the last remaining component
        c = int(rng.choice(valid))
        x[c] = 0.0
        return x, c
