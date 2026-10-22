"""Lazy SeisBench dataset helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


_TRACE_NPTS_PATTERN = re.compile(r":(?P<npts>\d+)\s*$")


def _import_seisbench_data() -> Any:
    import seisbench.data as sbd

    return sbd


def _import_seisbench_generator() -> Any:
    import seisbench.generate as sbg

    return sbg


def _import_worker_seeding():
    from seisbench.util import worker_seeding

    return worker_seeding


def resolve_dataset_class(name: str) -> Any:
    sbd = _import_seisbench_data()
    try:
        return getattr(sbd, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown SeisBench dataset '{name}'.") from exc


def apply_training_fraction(training_fraction: float, train_data: Any) -> None:
    if not 0.0 < training_fraction <= 1.0:
        raise ValueError("training_fraction must be in (0, 1].")

    if training_fraction >= 1.0:
        return

    if "trace_name" not in train_data.metadata.columns:
        raise ValueError("Dataset metadata must contain 'trace_name' for block-wise subsampling.")

    blocks = train_data["trace_name"].apply(lambda trace_name: trace_name.split("$")[0])
    unique_blocks = np.asarray(blocks.unique(), dtype=object)
    np.random.shuffle(unique_blocks)
    keep_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
    mask = blocks.isin(set(keep_blocks))
    train_data.filter(mask, inplace=True)


def filter_min_trace_length(
    data: Any,
    *,
    min_length_samples: int,
    target_sampling_rate: float,
) -> None:
    """Drop traces that are shorter than the requested no-padding window."""

    estimated_lengths = estimate_resampled_trace_lengths(
        metadata=data.metadata,
        target_sampling_rate=target_sampling_rate,
    )
    mask = np.isfinite(estimated_lengths) & (estimated_lengths >= int(min_length_samples))
    data.filter(mask, inplace=True)


def filter_event_traces(data: Any) -> None:
    """Keep non-noise traces for event-style pretraining windows."""

    if "trace_category" not in data.metadata.columns:
        return
    categories = data.metadata["trace_category"].astype(str).str.lower()
    data.filter(categories != "noise", inplace=True)


def filter_noise_traces(data: Any) -> None:
    """Keep only SeisBench traces explicitly marked as noise."""

    if "trace_category" not in data.metadata.columns:
        data.filter(np.zeros(len(data), dtype=bool), inplace=True)
        return
    categories = data.metadata["trace_category"].astype(str).str.lower()
    data.filter(categories == "noise", inplace=True)


def estimate_resampled_trace_lengths(
    *,
    metadata: Any,
    target_sampling_rate: float,
) -> np.ndarray:
    """Estimate waveform lengths after SeisBench resampling."""

    source_lengths = _metadata_trace_lengths(metadata)
    source_sampling_rates = _metadata_sampling_rates(
        metadata,
        default_sampling_rate=float(target_sampling_rate),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = float(target_sampling_rate) / source_sampling_rates
        estimated_lengths = np.rint(source_lengths * scale)
    estimated_lengths[~np.isfinite(estimated_lengths)] = np.nan
    return estimated_lengths


def _metadata_trace_lengths(metadata: Any) -> np.ndarray:
    for column in (
        "trace_npts",
        "trace_number_of_samples",
        "trace_length_samples",
    ):
        if column in metadata.columns:
            return _to_float_array(metadata[column])

    if "trace_name" not in metadata.columns:
        return np.full(len(metadata), np.nan, dtype=np.float64)

    lengths = np.full(len(metadata), np.nan, dtype=np.float64)
    for idx, trace_name in enumerate(metadata["trace_name"]):
        match = _TRACE_NPTS_PATTERN.search(str(trace_name))
        if match is not None:
            lengths[idx] = float(match.group("npts"))
    return lengths


def _metadata_sampling_rates(
    metadata: Any,
    *,
    default_sampling_rate: float,
) -> np.ndarray:
    if "trace_sampling_rate_hz" in metadata.columns:
        sampling_rates = _to_float_array(metadata["trace_sampling_rate_hz"])
    elif "trace_dt_s" in metadata.columns:
        trace_dt = _to_float_array(metadata["trace_dt_s"])
        with np.errstate(divide="ignore", invalid="ignore"):
            sampling_rates = 1.0 / trace_dt
    else:
        sampling_rates = np.full(len(metadata), default_sampling_rate, dtype=np.float64)

    invalid = ~np.isfinite(sampling_rates) | (sampling_rates <= 0)
    sampling_rates[invalid] = default_sampling_rate
    return sampling_rates


def _to_float_array(values: Any) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for idx, value in enumerate(values):
        try:
            result[idx] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def ensure_split(dataset: Any) -> None:
    if "split" in dataset.metadata.columns:
        return

    if "event_id" in dataset.metadata.columns:
        split = _build_group_split(dataset.metadata["event_id"], len(dataset))
    else:
        split = _build_row_order_split(len(dataset))

    dataset._metadata["split"] = split  # pylint: disable=protected-access


def _build_row_order_split(length: int) -> np.ndarray:
    split = np.array(["train"] * length)
    split[int(0.6 * length) : int(0.7 * length)] = "dev"
    split[int(0.7 * length) :] = "test"
    return split


def _build_group_split(group_values: Any, length: int) -> np.ndarray:
    groups = [
        _normalise_group_value(value=value, row_idx=row_idx)
        for row_idx, value in enumerate(group_values)
    ]
    if len(groups) != length:
        raise ValueError("Split group values must have the same length as the dataset.")

    unique_groups = sorted(set(groups), key=_stable_hash)
    train_end = int(0.6 * len(unique_groups))
    dev_end = int(0.7 * len(unique_groups))

    group_to_split = {}
    for idx, group in enumerate(unique_groups):
        if idx < train_end:
            group_to_split[group] = "train"
        elif idx < dev_end:
            group_to_split[group] = "dev"
        else:
            group_to_split[group] = "test"

    return np.asarray([group_to_split[group] for group in groups], dtype=object)


def _normalise_group_value(*, value: Any, row_idx: int) -> str:
    if value is None:
        return f"__row_{row_idx}"
    try:
        if bool(np.isnan(value)):
            return f"__row_{row_idx}"
    except (TypeError, ValueError):
        pass
    return str(value)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_generator(dataset: Any) -> Any:
    sbg = _import_seisbench_generator()
    return sbg.GenericGenerator(dataset)


def build_dataset(
    *,
    dataset_name: str,
    sampling_rate: int,
    component_order: str,
    dimension_order: str,
    cache: str | None,
) -> Any:
    dataset_cls = resolve_dataset_class(dataset_name)
    dataset = dataset_cls(
        sampling_rate=sampling_rate,
        component_order=component_order,
        dimension_order=dimension_order,
        cache=cache,
    )
    ensure_split(dataset)
    return dataset


def get_worker_init_fn():
    return _import_worker_seeding()
