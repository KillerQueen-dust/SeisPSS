"""SeisBench data module for dense phase picking."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from seispss.data.augmentations import (
    CaptureValidComponentMask,
    ChangeDtype,
    FillMissingComponents,
    StdSafeNormalize,
    WaveformAugment,
)
from seispss.data.seisbench import (
    apply_training_fraction,
    build_dataset,
    build_generator,
    filter_min_trace_length,
    get_worker_init_fn,
)
from seispss.downstream.phase_picking.labels import PhaseLabelGenerator
from seispss.downstream.phase_picking.windowing import (
    EvalWindowSelector,
    PLeadWindowSelector,
    PhaseWindowSelector,
)
from seispss.utils.phase_metadata import SEISBENCH_PHASE_PICK_KEYS


GEOFON_EVAL_WINDOW_SAMPLES = 12_000


class PhasePickingDataModule(L.LightningDataModule):
    """Prepare SeisBench traces for P/S/noise dense labeling."""

    def __init__(
        self,
        *,
        dataset_name: str,
        sampling_rate: int,
        component_order: str,
        dimension_order: str,
        batch_size: int,
        val_batch_size: int | None,
        num_workers: int,
        training_fraction: float,
        cache: str | None,
        prefetch_factor: int | None,
        amp_norm_type: str,
        norm_eps: float,
        require_phase: bool = True,
        pin_memory: bool = True,
        eval_split: str = "test",
        window: dict[str, Any],
        labels: dict[str, Any],
        preload_train: bool = True,
        preload_eval: bool = False,
        augment: dict[str, Any] | None = None,
        train_subset_manifest: str | None = None,
        loader_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = str(dataset_name)
        self.sampling_rate = int(sampling_rate)
        self.component_order = component_order
        self.dimension_order = dimension_order
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size or batch_size)
        self.num_workers = int(num_workers)
        self.training_fraction = float(training_fraction)
        self.cache = cache
        self.preload_train = bool(preload_train)
        self.preload_eval = bool(preload_eval)
        self.prefetch_factor = prefetch_factor
        self.amp_norm_type = amp_norm_type
        self.norm_eps = float(norm_eps)
        self.require_phase = bool(require_phase)
        self.pin_memory = bool(pin_memory)
        self.eval_split = "dev" if str(eval_split) == "val" else str(eval_split)
        self.window = window
        self.labels = labels
        self.augment = dict(augment) if augment else {}
        self.train_subset_manifest = train_subset_manifest
        self.loader_seed = loader_seed

        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def setup(self, stage: str | None = None) -> None:
        if stage is None:
            dataset = self._base_dataset()
            self.train_generator = self._build_train_generator(dataset=dataset)
            self.val_generator = self._build_eval_generator(
                split_name=self.eval_split,
                dataset=dataset,
            )
            self.test_generator = self._build_eval_generator(
                split_name="test",
                dataset=dataset,
            )
        elif stage == "fit":
            dataset = self._base_dataset()
            self.train_generator = self._build_train_generator(dataset=dataset)
            self.val_generator = self._build_eval_generator(
                split_name=self.eval_split,
                dataset=dataset,
            )
        elif stage == "validate":
            self.val_generator = self._build_eval_generator(split_name=self.eval_split)
        elif stage == "test":
            self.test_generator = self._build_eval_generator(split_name="test")

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(
            self.train_generator,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(
            self.val_generator,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(
            self.test_generator,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=False,
        )

    def _build_train_generator(self, *, dataset: Any | None = None) -> Any:
        if dataset is None:
            dataset = self._base_dataset()
        split = dataset.train()
        expected_names = None
        if self.train_subset_manifest:
            if self.training_fraction != 1.0:
                raise ValueError("A frozen train subset requires training_fraction=1.0.")
            manifest = pd.read_csv(Path(self.train_subset_manifest), keep_default_na=False)
            expected_names = manifest["trace_name"].astype(str)
            available = split.metadata["trace_name"].astype(str)
            if expected_names.duplicated().any() or available.duplicated().any():
                raise ValueError("Frozen subset trace names must be unique.")
            if "split" in manifest and not manifest["split"].eq("train").all():
                raise ValueError("Frozen train subset contains a non-train record.")
            if not expected_names.isin(available).all():
                raise ValueError("Frozen train subset contains missing or non-train trace names.")
            split.filter(available.isin(expected_names).to_numpy(), inplace=True)
        else:
            apply_training_fraction(self.training_fraction, split)
        self._filter_split(
            split,
            min_length_samples=int(self.window["length_samples"]),
        )
        if expected_names is not None and set(split.metadata["trace_name"].astype(str)) != set(expected_names):
            raise ValueError("Frozen subset eligibility changed; rebuild and version its protocol.")
        if self.cache and self.preload_train:
            split.preload_waveforms(pbar=True)
        generator = build_generator(split)
        generator.add_augmentations(self._build_train_augmentations())
        if len(generator) == 0:
            raise ValueError("No phase-picking training samples remain after filtering.")
        return generator

    def _build_eval_generator(self, *, split_name: str, dataset: Any | None = None) -> Any:
        if dataset is None:
            dataset = self._base_dataset()
        if split_name == "test":
            split = dataset.test()
        elif split_name == "dev":
            split = dataset.dev()
        else:
            raise ValueError("eval_split must be one of: dev, val, test.")
        eval_window_samples = self._eval_window_samples()
        self._filter_split(
            split,
            min_length_samples=eval_window_samples,
            phase_prefix_samples=eval_window_samples if self._uses_geofon_prefix_eval() else None,
        )
        if self.cache and self.preload_eval:
            split.preload_waveforms(pbar=True)
        generator = build_generator(split)
        generator.add_augmentations(self._build_eval_augmentations())
        if len(generator) == 0:
            raise ValueError("No phase-picking evaluation samples remain after filtering.")
        return generator

    def _base_dataset(self) -> Any:
        return build_dataset(
            dataset_name=self.dataset_name,
            sampling_rate=self.sampling_rate,
            component_order=self.component_order,
            dimension_order=self.dimension_order,
            cache=self.cache,
        )

    def _filter_split(
        self,
        split: Any,
        *,
        min_length_samples: int,
        phase_prefix_samples: int | None = None,
    ) -> None:
        filter_min_trace_length(
            split,
            min_length_samples=int(min_length_samples),
            target_sampling_rate=self.sampling_rate,
        )
        if self.require_phase:
            if phase_prefix_samples is None:
                phase_mask = _phase_available_mask(split.metadata)
            else:
                phase_mask = _phase_in_prefix_mask(
                    split.metadata,
                    prefix_samples=int(phase_prefix_samples),
                    target_sampling_rate=self.sampling_rate,
                )
            split.filter(phase_mask, inplace=True)

    def _build_train_augmentations(self) -> list[Any]:
        train_cfg = dict(self.window.get("train", {}))
        strategy = str(train_cfg.get("strategy", "phase"))
        if strategy == "p_lead":
            window_selector = PLeadWindowSelector(
                window_length=int(self.window["length_samples"]),
                sampling_rate=self.sampling_rate,
                min_p_lead_seconds=float(train_cfg["min_p_lead_seconds"]),
                max_p_lead_seconds=float(train_cfg["max_p_lead_seconds"]),
                deterministic=False,
            )
        elif strategy in {"phase", "mixed"}:
            window_selector = PhaseWindowSelector(
                window_length=int(self.window["length_samples"]),
                sampling_rate=self.sampling_rate,
                p_probability=float(train_cfg.get("p_probability", 0.4)),
                s_probability=float(train_cfg.get("s_probability", 0.4)),
                random_probability=float(train_cfg.get("random_probability", 0.2)),
                min_pick_position_fraction=float(train_cfg.get("min_pick_position_fraction", 0.3)),
                max_pick_position_fraction=float(train_cfg.get("max_pick_position_fraction", 0.7)),
                deterministic=False,
            )
        else:
            raise ValueError(f"Unsupported phase-picking train window strategy: {strategy}")
        return [
            window_selector,
            *self._post_window_augmentations(deterministic=False),
        ]

    def _build_eval_augmentations(self) -> list[Any]:
        eval_cfg = dict(self.window.get("eval", {}))
        window_length = self._eval_window_samples()
        strategy = "prefix" if self._uses_geofon_prefix_eval() else str(eval_cfg.get("strategy", "full"))
        return [
            EvalWindowSelector(
                window_length=window_length,
                sampling_rate=self.sampling_rate,
                strategy=strategy,
            ),
            *self._post_window_augmentations(deterministic=True),
        ]

    def _eval_window_samples(self) -> int:
        if self._uses_geofon_prefix_eval():
            return GEOFON_EVAL_WINDOW_SAMPLES
        return int(self.window["length_samples"])

    def _uses_geofon_prefix_eval(self) -> bool:
        eval_cfg = dict(self.window.get("eval", {}))
        return self.dataset_name.lower() == "geofon" and str(eval_cfg.get("strategy", "full")) == "full"

    def _augment_enabled(self) -> bool:
        return bool(self.augment.get("enabled", False))

    def _post_window_augmentations(self, *, deterministic: bool) -> list[Any]:
        transforms: list[Any] = [
            CaptureValidComponentMask(),
            FillMissingComponents(deterministic=deterministic),
        ]
        # Train-only, label-preserving waveform augmentation, applied before
        # normalization. Never added to the eval pipeline (deterministic=True).
        if not deterministic and self._augment_enabled():
            transforms.append(
                WaveformAugment(
                    config=self.augment,
                    sampling_rate=self.sampling_rate,
                    deterministic=False,
                )
            )
        transforms += [
            StdSafeNormalize(amp_norm_type=self.amp_norm_type, eps=self.norm_eps),
            PhaseLabelGenerator(
                sigma_p=float(self.labels["sigma_p"]),
                sigma_s=float(self.labels["sigma_s"]),
                sampling_rate=self.sampling_rate,
            ),
            ChangeDtype(keys=("X", "valid_time_mask", "y")),
        ]
        return transforms

    def _build_loader(
        self,
        dataset: Any,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=get_worker_init_fn(),
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,
        )
        if self.loader_seed is not None:
            kwargs["generator"] = torch.Generator().manual_seed(int(self.loader_seed))
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)


def _phase_available_mask(metadata: Any) -> np.ndarray:
    mask = np.zeros(len(metadata), dtype=np.bool_)
    for key in SEISBENCH_PHASE_PICK_KEYS:
        if key not in metadata.columns:
            continue
        values = metadata[key]
        for idx, value in enumerate(values):
            if _is_finite_number(value):
                mask[idx] = True
    return mask


def _phase_in_prefix_mask(
    metadata: Any,
    *,
    prefix_samples: int,
    target_sampling_rate: float,
) -> np.ndarray:
    mask = np.zeros(len(metadata), dtype=np.bool_)
    scales = _metadata_sample_scales(
        metadata,
        target_sampling_rate=target_sampling_rate,
    )
    for key in SEISBENCH_PHASE_PICK_KEYS:
        if key not in metadata.columns:
            continue
        values = metadata[key]
        for idx, value in enumerate(values):
            if mask[idx]:
                continue
            for sample in _metadata_value_to_samples(value):
                sample = int(round(float(sample) * float(scales[idx])))
                if 0 <= sample < int(prefix_samples):
                    mask[idx] = True
                    break
    return mask


def _metadata_sample_scales(metadata: Any, *, target_sampling_rate: float) -> np.ndarray:
    sampling_rates = np.full(len(metadata), float(target_sampling_rate), dtype=np.float64)
    if "trace_sampling_rate_hz" in metadata.columns:
        sampling_rates = _metadata_numeric_array(metadata["trace_sampling_rate_hz"])
    elif "trace_dt_s" in metadata.columns:
        trace_dt = _metadata_numeric_array(metadata["trace_dt_s"])
        with np.errstate(divide="ignore", invalid="ignore"):
            sampling_rates = 1.0 / trace_dt
    invalid = ~np.isfinite(sampling_rates) | (sampling_rates <= 0)
    sampling_rates[invalid] = float(target_sampling_rate)
    return float(target_sampling_rate) / sampling_rates


def _metadata_numeric_array(values: Any) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for idx, value in enumerate(values):
        samples = _metadata_value_to_samples(value)
        if samples:
            result[idx] = float(samples[0])
    return result


def _metadata_value_to_samples(value: Any) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return []
    if array.size == 0:
        return []
    array = array[np.isfinite(array)]
    return [float(sample) for sample in array]


def _is_finite_number(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return array.size > 0 and bool(np.isfinite(array).any())


ETHZPhasePickingDataModule = PhasePickingDataModule
