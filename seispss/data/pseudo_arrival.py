"""Pseudo-arrival anchor estimation and cache utilities."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATUS_SUCCESS = np.int8(0)
STATUS_NO_TRIGGER = np.int8(1)
STATUS_NO_VALID_CHANNEL = np.int8(2)
STATUS_ERROR = np.int8(3)

STATUS_NAMES = {
    int(STATUS_SUCCESS): "success",
    int(STATUS_NO_TRIGGER): "no_trigger",
    int(STATUS_NO_VALID_CHANNEL): "no_valid_channel",
    int(STATUS_ERROR): "error",
}


@dataclass(frozen=True)
class PseudoArrivalEstimate:
    """Single-trace pseudo-arrival estimate."""

    anchor_sample: int
    confidence: float
    valid: bool
    z_peak: float
    persistence: float
    status: np.int8


@dataclass(frozen=True)
class PseudoArrivalCachePaths:
    npz_path: Path
    manifest_path: Path
    lock_path: Path


class RobustEnergyOnsetEstimator:
    """Robust, label-free energy-onset detector for seismic waveforms."""

    def __init__(self, **cfg: Any) -> None:
        self.name = str(cfg.get("name", "robust_energy_onset_v1"))
        self.version = int(cfg.get("version", 1))
        self.clip_k = float(cfg.get("clip_k", 8.0))
        self.short_windows_sec = tuple(float(v) for v in cfg.get("short_windows_sec", (0.2, 0.5)))
        self.long_window_sec = float(cfg.get("long_window_sec", 5.0))
        self.long_window_max_fraction = float(cfg.get("long_window_max_fraction", 0.15))
        self.theta_on = float(cfg.get("theta_on", 2.75))
        self.theta_hold = float(cfg.get("theta_hold", 1.25))
        self.persistence_sec = float(cfg.get("persistence_sec", 0.20))
        self.persistence_fraction = float(cfg.get("persistence_fraction", 0.60))
        self.confidence_lo = float(cfg.get("confidence_lo", 2.75))
        self.confidence_hi = float(cfg.get("confidence_hi", 6.0))
        self.confidence_peak_window_sec = float(cfg.get("confidence_peak_window_sec", 2.0))
        self.eps = float(cfg.get("eps", 1.0e-6))

    def estimate(self, x: np.ndarray, *, sampling_rate: float) -> PseudoArrivalEstimate:
        """Estimate the earliest persistent energy rise.

        Args:
            x: Waveform array in [channels, time] layout after SeisBench resampling.
            sampling_rate: Target waveform sampling rate in Hz.
        """

        try:
            waveform = np.asarray(x, dtype=np.float32)
            if waveform.ndim != 2:
                raise ValueError(f"Expected waveform shape [channels, time], got {waveform.shape}.")
            normalized = self._robust_normalize(waveform)
            if normalized.size == 0:
                return _invalid_estimate(status=STATUS_NO_VALID_CHANNEL)

            score = self._score(normalized, sampling_rate=float(sampling_rate))
            if score.size == 0 or not np.isfinite(score).any():
                return _invalid_estimate(status=STATUS_NO_TRIGGER)
            anchor, persistence = self._find_trigger(score, sampling_rate=float(sampling_rate))
            if anchor is None:
                return _invalid_estimate(status=STATUS_NO_TRIGGER)

            peak_window = max(int(round(self.confidence_peak_window_sec * float(sampling_rate))), 1)
            peak_end = min(int(anchor) + peak_window, score.shape[0])
            z_peak = float(np.nanmax(score[int(anchor) : peak_end])) if peak_end > int(anchor) else float(score[int(anchor)])
            confidence = self._confidence(z_peak=z_peak, persistence=persistence)
            return PseudoArrivalEstimate(
                anchor_sample=int(anchor),
                confidence=float(confidence),
                valid=True,
                z_peak=z_peak,
                persistence=float(persistence),
                status=STATUS_SUCCESS,
            )
        except Exception:
            return _invalid_estimate(status=STATUS_ERROR)

    def _robust_normalize(self, waveform: np.ndarray) -> np.ndarray:
        finite_waveform = np.where(np.isfinite(waveform), waveform, 0.0)
        std = finite_waveform.std(axis=-1)
        valid_channels = np.isfinite(std) & (std > self.eps)
        if not bool(valid_channels.any()):
            return np.empty((0, waveform.shape[-1]), dtype=np.float32)

        x = finite_waveform[valid_channels].astype(np.float32, copy=True)
        median = np.median(x, axis=-1, keepdims=True)
        centered = x - median
        mad = np.median(np.abs(centered), axis=-1, keepdims=True)
        scale = 1.4826 * mad
        fallback = x.std(axis=-1, keepdims=True)
        scale = np.where(scale > self.eps, scale, fallback)
        scale = np.where(scale > self.eps, scale, 1.0)
        normalized = centered / scale
        return np.clip(normalized, -self.clip_k, self.clip_k)

    def _score(self, normalized: np.ndarray, *, sampling_rate: float) -> np.ndarray:
        energy = np.mean(np.square(normalized), axis=0).astype(np.float32, copy=False)
        length = int(energy.shape[0])
        if length <= 1:
            return np.zeros(length, dtype=np.float32)

        short_windows = [
            max(int(round(window_sec * sampling_rate)), 1)
            for window_sec in self.short_windows_sec
        ]
        short_windows = [min(window, length) for window in short_windows]
        long_window = int(round(self.long_window_sec * sampling_rate))
        long_window = min(long_window, max(int(round(self.long_window_max_fraction * length)), 1))
        long_window = max(long_window, 4 * max(short_windows))
        long_window = min(long_window, length)

        baseline = _causal_mean(energy, long_window)
        ratios = []
        for short_window in short_windows:
            short_energy = _causal_mean(energy, short_window)
            ratios.append(np.log(short_energy + self.eps) - np.log(baseline + self.eps))
        raw_score = np.mean(np.stack(ratios, axis=0), axis=0)

        median = float(np.nanmedian(raw_score))
        mad = float(np.nanmedian(np.abs(raw_score - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= self.eps:
            scale = float(np.nanstd(raw_score))
        if not np.isfinite(scale) or scale <= self.eps:
            return np.zeros_like(raw_score, dtype=np.float32)
        return ((raw_score - median) / (scale + self.eps)).astype(np.float32, copy=False)

    def _find_trigger(self, score: np.ndarray, *, sampling_rate: float) -> tuple[int | None, float]:
        hold = max(int(round(self.persistence_sec * sampling_rate)), 1)
        above_hold = (score >= self.theta_hold).astype(np.int32)
        cumulative = np.concatenate([[0], np.cumsum(above_hold, dtype=np.int64)])
        starts = np.arange(score.shape[0], dtype=np.int64)
        ends = np.minimum(starts + hold, score.shape[0])
        counts = cumulative[ends] - cumulative[starts]
        denominators = np.maximum(ends - starts, 1)
        persistence = counts / denominators
        candidates = np.flatnonzero((score >= self.theta_on) & (persistence >= self.persistence_fraction))
        if candidates.size == 0:
            return None, 0.0
        anchor = int(candidates[0])
        return anchor, float(persistence[anchor])

    def _confidence(self, *, z_peak: float, persistence: float) -> float:
        denom = max(self.confidence_hi - self.confidence_lo, self.eps)
        peak_confidence = (float(z_peak) - self.confidence_lo) / denom
        peak_confidence = min(max(peak_confidence, 0.0), 1.0)
        persistence_confidence = min(max(float(persistence), 0.0), 1.0)
        return float(peak_confidence * persistence_confidence)


def ensure_pseudo_arrival_cache(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    cache_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Load a cache if valid; otherwise build it under a simple file lock."""

    estimator_cfg = dict(cache_cfg.get("estimator", {}))
    estimator = RobustEnergyOnsetEstimator(**estimator_cfg)
    row_hash = metadata_row_hashes(dataset.metadata, dataset_name=dataset_name, split=split)
    row_hash_digest = digest_row_hashes(row_hash)
    fingerprint = fingerprint_anchor_cache(
        dataset=dataset,
        dataset_name=dataset_name,
        split=split,
        sampling_rate=sampling_rate,
        component_order=component_order,
        dimension_order=dimension_order,
        estimator_cfg=estimator_cfg,
        row_hash_digest=row_hash_digest,
    )
    paths = pseudo_arrival_cache_paths(
        root=cache_cfg.get("root", "outputs/cache/pseudo_arrival"),
        estimator_name=estimator.name,
        dataset_name=dataset_name,
        split=split,
        fingerprint=fingerprint,
    )
    force_rebuild = bool(cache_cfg.get("force_rebuild", False))
    if not force_rebuild and _cache_is_valid(paths=paths, expected_row_hash_digest=row_hash_digest):
        print(f"[pseudo-arrival] cache hit: {dataset_name}/{split} {paths.npz_path}")
        return load_pseudo_arrival_cache(
            dataset,
            dataset_name=dataset_name,
            split=split,
            sampling_rate=sampling_rate,
            component_order=component_order,
            dimension_order=dimension_order,
            cache_cfg=cache_cfg,
        )

    _build_with_lock(
        dataset,
        dataset_name=dataset_name,
        split=split,
        sampling_rate=sampling_rate,
        component_order=component_order,
        dimension_order=dimension_order,
        cache_cfg=cache_cfg,
        estimator=estimator,
        row_hash=row_hash,
        row_hash_digest=row_hash_digest,
        fingerprint=fingerprint,
        paths=paths,
    )
    return _load_cache_payload(paths=paths, expected_row_hash=row_hash)


def load_pseudo_arrival_cache(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    cache_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Load and row-align a pseudo-arrival cache for a dataset split."""

    estimator_cfg = dict(cache_cfg.get("estimator", {}))
    estimator = RobustEnergyOnsetEstimator(**estimator_cfg)
    row_hash = metadata_row_hashes(dataset.metadata, dataset_name=dataset_name, split=split)
    row_hash_digest = digest_row_hashes(row_hash)
    fingerprint = fingerprint_anchor_cache(
        dataset=dataset,
        dataset_name=dataset_name,
        split=split,
        sampling_rate=sampling_rate,
        component_order=component_order,
        dimension_order=dimension_order,
        estimator_cfg=estimator_cfg,
        row_hash_digest=row_hash_digest,
    )
    paths = pseudo_arrival_cache_paths(
        root=cache_cfg.get("root", "outputs/cache/pseudo_arrival"),
        estimator_name=estimator.name,
        dataset_name=dataset_name,
        split=split,
        fingerprint=fingerprint,
    )
    return _load_cache_payload(paths=paths, expected_row_hash=row_hash)


def inject_pseudo_arrival_cache(dataset: Any, cache_payload: dict[str, Any]) -> None:
    """Inject cache arrays into SeisBench split metadata after row-hash validation."""

    row_hash = np.asarray(cache_payload["row_hash"], dtype=np.uint64)
    if len(dataset.metadata) != int(row_hash.shape[0]):
        raise ValueError(
            f"Pseudo-arrival cache length {row_hash.shape[0]} does not match split metadata length "
            f"{len(dataset.metadata)}."
        )
    manifest = cache_payload.get("manifest", {})
    dataset_name = manifest.get("dataset_name")
    split = manifest.get("split")
    if dataset_name is not None and split is not None:
        current_row_hash = metadata_row_hashes(dataset.metadata, dataset_name=str(dataset_name), split=str(split))
        if not np.array_equal(current_row_hash, row_hash):
            raise ValueError("Pseudo-arrival cache row_hash mismatch for current split metadata.")

    metadata = getattr(dataset, "_metadata", dataset.metadata)
    if "pseudo_anchor_row_hash" in metadata.columns:
        current = np.asarray(metadata["pseudo_anchor_row_hash"], dtype=np.uint64)
        if not np.array_equal(current, row_hash):
            raise ValueError("Existing pseudo-arrival row hashes do not match the cache payload.")

    for key in (
        "pseudo_anchor_sample",
        "pseudo_anchor_confidence",
        "pseudo_anchor_valid",
        "pseudo_anchor_z_peak",
        "pseudo_anchor_persistence",
        "pseudo_anchor_status",
    ):
        metadata[key] = cache_payload[key]
    metadata["pseudo_anchor_row_hash"] = row_hash


def metadata_row_hashes(metadata: Any, *, dataset_name: str, split: str) -> np.ndarray:
    """Build stable uint64 row hashes for SeisBench metadata alignment."""

    fields = (
        "trace_name",
        "source_id",
        "event_id",
        "station_network_code",
        "station_code",
        "station_location_code",
        "trace_npts",
        "trace_number_of_samples",
        "trace_length_samples",
        "trace_sampling_rate_hz",
        "trace_dt_s",
        "trace_chunk",
    )
    hashes = np.empty(len(metadata), dtype=np.uint64)
    for row_position, (_, row) in enumerate(metadata.iterrows()):
        parts = [dataset_name, split, f"row={row_position}"]
        for field in fields:
            if field in metadata.columns:
                parts.append(f"{field}={_normalise_value(row.get(field))}")
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        hashes[row_position] = np.uint64(int(digest[:16], 16))
    return hashes


def digest_row_hashes(row_hash: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(row_hash, dtype=np.uint64).tobytes()).hexdigest()


def fingerprint_anchor_cache(
    *,
    dataset: Any,
    dataset_name: str,
    split: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    estimator_cfg: dict[str, Any],
    row_hash_digest: str,
) -> str:
    payload = {
        "dataset_name": dataset_name,
        "split": split,
        "sampling_rate": int(sampling_rate),
        "component_order": component_order,
        "dimension_order": dimension_order,
        "dataset_class": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
        "estimator": _json_safe(estimator_cfg),
        "row_hash_digest": row_hash_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def pseudo_arrival_cache_paths(
    *,
    root: str | Path,
    estimator_name: str,
    dataset_name: str,
    split: str,
    fingerprint: str,
) -> PseudoArrivalCachePaths:
    cache_dir = Path(root) / _safe_path(estimator_name) / _safe_path(dataset_name) / _safe_path(split)
    return PseudoArrivalCachePaths(
        npz_path=cache_dir / f"{fingerprint}.npz",
        manifest_path=cache_dir / f"{fingerprint}.manifest.json",
        lock_path=cache_dir / f"{fingerprint}.lock",
    )


def _build_with_lock(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    cache_cfg: dict[str, Any],
    estimator: RobustEnergyOnsetEstimator,
    row_hash: np.ndarray,
    row_hash_digest: str,
    fingerprint: str,
    paths: PseudoArrivalCachePaths,
) -> None:
    paths.npz_path.parent.mkdir(parents=True, exist_ok=True)
    poll_sec = float(cache_cfg.get("lock_poll_sec", 10.0))
    while True:
        try:
            fd = os.open(paths.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()} time={time.time()}\n")
            break
        except FileExistsError:
            if _cache_is_valid(paths=paths, expected_row_hash_digest=row_hash_digest):
                return
            print(f"[pseudo-arrival] waiting for cache lock: {paths.lock_path}")
            time.sleep(max(poll_sec, 1.0))

    try:
        if not bool(cache_cfg.get("force_rebuild", False)) and _cache_is_valid(
            paths=paths,
            expected_row_hash_digest=row_hash_digest,
        ):
            return
        _build_pseudo_arrival_cache(
            dataset,
            dataset_name=dataset_name,
            split=split,
            sampling_rate=sampling_rate,
            component_order=component_order,
            dimension_order=dimension_order,
            cache_cfg=cache_cfg,
            estimator=estimator,
            row_hash=row_hash,
            row_hash_digest=row_hash_digest,
            fingerprint=fingerprint,
            paths=paths,
        )
    finally:
        try:
            paths.lock_path.unlink()
        except FileNotFoundError:
            pass


def _build_pseudo_arrival_cache(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    cache_cfg: dict[str, Any],
    estimator: RobustEnergyOnsetEstimator,
    row_hash: np.ndarray,
    row_hash_digest: str,
    fingerprint: str,
    paths: PseudoArrivalCachePaths,
) -> None:
    length = len(dataset)
    anchor_sample = np.full(length, -1, dtype=np.int32)
    confidence = np.zeros(length, dtype=np.float32)
    valid = np.zeros(length, dtype=np.bool_)
    z_peak = np.zeros(length, dtype=np.float32)
    persistence = np.zeros(length, dtype=np.float32)
    status = np.full(length, STATUS_ERROR, dtype=np.int8)
    log_every = max(int(cache_cfg.get("log_every_n", 1000)), 1)
    start_time = time.time()
    print(f"[pseudo-arrival] building cache: {dataset_name}/{split}, rows={length}, path={paths.npz_path}")
    for idx in range(length):
        x, _ = dataset.get_sample(idx)
        estimate = estimator.estimate(x, sampling_rate=float(sampling_rate))
        anchor_sample[idx] = int(estimate.anchor_sample)
        confidence[idx] = float(estimate.confidence)
        valid[idx] = bool(estimate.valid)
        z_peak[idx] = float(estimate.z_peak)
        persistence[idx] = float(estimate.persistence)
        status[idx] = np.int8(estimate.status)
        if (idx + 1) % log_every == 0 or (idx + 1) == length:
            elapsed = max(time.time() - start_time, 1.0e-6)
            rate = float(idx + 1) / elapsed
            eta = (length - idx - 1) / max(rate, 1.0e-6)
            print(
                f"[pseudo-arrival] {dataset_name}/{split}: {idx + 1}/{length} "
                f"({rate:.1f} traces/s, eta={eta / 60.0:.1f} min)"
            )

    summary = _cache_summary(
        anchor_sample=anchor_sample,
        confidence=confidence,
        valid=valid,
        z_peak=z_peak,
        status=status,
    )
    manifest = {
        "format_version": "pseudo_arrival_cache_v1",
        "fingerprint": fingerprint,
        "dataset_name": dataset_name,
        "split": split,
        "sampling_rate": int(sampling_rate),
        "component_order": component_order,
        "dimension_order": dimension_order,
        "dataset_class": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
        "estimator": _json_safe(dict(cache_cfg.get("estimator", {}))),
        "row_count": int(length),
        "row_hash_digest": row_hash_digest,
        "created_unix_time": time.time(),
        "summary": summary,
    }

    tmp_npz = paths.npz_path.with_suffix(paths.npz_path.suffix + ".tmp")
    tmp_manifest = paths.manifest_path.with_suffix(paths.manifest_path.suffix + ".tmp")
    with open(tmp_npz, "wb") as handle:
        np.savez(
            handle,
            pseudo_anchor_sample=anchor_sample,
            pseudo_anchor_confidence=confidence,
            pseudo_anchor_valid=valid,
            pseudo_anchor_z_peak=z_peak,
            pseudo_anchor_persistence=persistence,
            pseudo_anchor_status=status,
            row_hash=np.asarray(row_hash, dtype=np.uint64),
        )
    with open(tmp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_npz, paths.npz_path)
    os.replace(tmp_manifest, paths.manifest_path)
    print(f"[pseudo-arrival] cache built: {dataset_name}/{split} summary={summary}")


def _load_cache_payload(
    *,
    paths: PseudoArrivalCachePaths,
    expected_row_hash: np.ndarray,
) -> dict[str, Any]:
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        raise FileNotFoundError(f"Missing pseudo-arrival cache files for {paths.npz_path}")
    with np.load(paths.npz_path, allow_pickle=False) as payload:
        result = {key: payload[key] for key in payload.files}
    row_hash = np.asarray(result.get("row_hash"), dtype=np.uint64)
    if row_hash.shape != expected_row_hash.shape or not np.array_equal(row_hash, expected_row_hash):
        raise ValueError(f"Pseudo-arrival cache row_hash mismatch: {paths.npz_path}")
    with open(paths.manifest_path, "r", encoding="utf-8") as handle:
        result["manifest"] = json.load(handle)
    return result


def _cache_is_valid(*, paths: PseudoArrivalCachePaths, expected_row_hash_digest: str) -> bool:
    if not paths.npz_path.exists() or not paths.manifest_path.exists():
        return False
    try:
        with open(paths.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("row_hash_digest") != expected_row_hash_digest:
            return False
        if int(manifest.get("row_count", -1)) < 0:
            return False
        with np.load(paths.npz_path, allow_pickle=False) as payload:
            if "row_hash" not in payload.files:
                return False
            if digest_row_hashes(payload["row_hash"]) != expected_row_hash_digest:
                return False
        return True
    except Exception:
        return False


def _cache_summary(
    *,
    anchor_sample: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    z_peak: np.ndarray,
    status: np.ndarray,
) -> dict[str, Any]:
    valid_mask = np.asarray(valid, dtype=bool)
    summary: dict[str, Any] = {
        "coverage": float(valid_mask.mean()) if valid_mask.size else 0.0,
        "status_counts": {
            STATUS_NAMES.get(int(status_id), str(int(status_id))): int((status == status_id).sum())
            for status_id in np.unique(status)
        },
    }
    if bool(valid_mask.any()):
        summary["anchor_position_quantiles"] = _quantiles(anchor_sample[valid_mask].astype(np.float64))
        summary["confidence_quantiles"] = _quantiles(confidence[valid_mask].astype(np.float64))
        summary["z_peak_quantiles"] = _quantiles(z_peak[valid_mask].astype(np.float64))
    else:
        summary["anchor_position_quantiles"] = {}
        summary["confidence_quantiles"] = {}
        summary["z_peak_quantiles"] = {}
    return summary


def _quantiles(values: np.ndarray) -> dict[str, float]:
    quantile_points = (0.05, 0.25, 0.50, 0.75, 0.95)
    quantiles = np.quantile(values, quantile_points)
    return {f"q{int(point * 100):02d}": float(value) for point, value in zip(quantile_points, quantiles)}


def _causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    cumsum = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    indices = np.arange(values.shape[0], dtype=np.int64)
    starts = np.maximum(indices - window + 1, 0)
    sums = cumsum[indices + 1] - cumsum[starts]
    counts = indices + 1 - starts
    return (sums / np.maximum(counts, 1)).astype(np.float32, copy=False)


def _invalid_estimate(*, status: np.int8) -> PseudoArrivalEstimate:
    return PseudoArrivalEstimate(
        anchor_sample=-1,
        confidence=0.0,
        valid=False,
        z_peak=0.0,
        persistence=0.0,
        status=np.int8(status),
    )


def _normalise_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(np.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:.12g}" if np.isfinite(numeric) else ""
    return str(value).strip()


def _safe_path(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
