"""Lightning data module for SeisPSS pretraining."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from seispss.data.augmentations import (
    AddConstantMetadata,
    AddEventMetadata,
    AddPseudoArrivalMetadata,
    CaptureValidComponentMask,
    ChangeDtype,
    EventWindowSelector,
    FillMissingComponents,
    NoiseWindowSelector,
    PretrainValidationWindowSelector,
    StdSafeNormalize,
)
from seispss.data.event_metadata import MISSING_UID, event_station_uids
from seispss.data.pseudo_arrival import ensure_pseudo_arrival_cache, inject_pseudo_arrival_cache
from seispss.data.sampling import EventPairedBatchSampler, build_temperature_probabilities, make_group_sampler
from seispss.data.seisbench import (
    apply_training_fraction,
    build_dataset,
    build_generator,
    filter_event_traces,
    filter_min_trace_length,
    filter_noise_traces,
    get_worker_init_fn,
)


class SeisPSSPretrainDataModule(L.LightningDataModule):
    """Prepare SeisBench datasets for JEPA pretraining."""

    def __init__(
        self,
        *,
        dataset_names: list[str],
        sampling_rate: int,
        component_order: str,
        dimension_order: str,
        batch_size: int,
        val_batch_size: int | None,
        num_workers: int,
        training_fraction: float,
        cache: str | None,
        prefetch_factor: int | None,
        dataset_sampling: dict[str, Any],
        batch_sampling: dict[str, Any] | None = None,
        window: dict[str, Any],
        amp_norm_type: str,
        norm_eps: float,
        pin_memory: bool = True,
        anchor_cache: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_names = dataset_names
        self.sampling_rate = sampling_rate
        self.component_order = component_order
        self.dimension_order = dimension_order
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.training_fraction = training_fraction
        self.cache = cache
        self.prefetch_factor = prefetch_factor
        self.dataset_sampling = dataset_sampling
        self.batch_sampling = batch_sampling or {"earthquake_fraction": 1.0, "noise_fraction": 0.0}
        self.window = window
        self.amp_norm_type = amp_norm_type
        self.norm_eps = norm_eps
        self.pin_memory = pin_memory
        if anchor_cache is None:
            self.anchor_cache = {}
        elif OmegaConf.is_config(anchor_cache):
            self.anchor_cache = OmegaConf.to_container(anchor_cache, resolve=True)
        else:
            self.anchor_cache = dict(anchor_cache)

        self.train_dataset = None
        self.train_sampler = None
        self.train_batch_sampler = None
        self.val_generators: list[Any] = []
        self.val_loader_names: list[str] = []

    def prepare_data(self) -> None:
        if not self._anchor_cache_enabled():
            return

        for dataset_name in self.dataset_names:
            dataset = build_dataset(
                dataset_name=dataset_name,
                sampling_rate=self.sampling_rate,
                component_order=self.component_order,
                dimension_order=self.dimension_order,
                cache=self.cache,
            )
            for split_name in self._anchor_cache_splits():
                split_dataset = self._dataset_split(dataset, split_name)
                if split_name == "train":
                    apply_training_fraction(self.training_fraction, split_dataset)
                ensure_pseudo_arrival_cache(
                    split_dataset,
                    dataset_name=dataset_name,
                    split=split_name,
                    sampling_rate=self.sampling_rate,
                    component_order=self.component_order,
                    dimension_order=self.dimension_order,
                    cache_cfg=self.anchor_cache,
                )

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit"):
            return

        event_generators = []
        event_lengths: list[int] = []
        noise_generators = []
        noise_lengths: list[int] = []
        event_station_groups_by_dataset: list[list[list[list[int]]]] = []
        self.val_generators = []
        self.val_loader_names = []

        window_length = int(self.window["length_samples"])
        noise_dataset_name = str(self.batch_sampling.get("noise_dataset", "STEAD")).lower()
        noise_fraction = float(self.batch_sampling.get("noise_fraction", 0.0))

        for dataset_id, dataset_name in enumerate(self.dataset_names):
            dataset = build_dataset(
                dataset_name=dataset_name,
                sampling_rate=self.sampling_rate,
                component_order=self.component_order,
                dimension_order=self.dimension_order,
                cache=self.cache,
            )
            train_split = dataset.train()
            val_split = dataset.dev()
            apply_training_fraction(self.training_fraction, train_split)
            self._inject_anchor_cache_if_enabled(train_split, dataset_name=dataset_name, split_name="train")
            self._inject_anchor_cache_if_enabled(val_split, dataset_name=dataset_name, split_name="dev")
            filter_min_trace_length(
                train_split,
                min_length_samples=window_length,
                target_sampling_rate=self.sampling_rate,
            )
            filter_min_trace_length(
                val_split,
                min_length_samples=window_length,
                target_sampling_rate=self.sampling_rate,
            )

            event_train = self._copy_dataset(train_split)
            filter_event_traces(event_train)

            if self.cache:
                event_train.preload_waveforms(pbar=True)
                val_split.preload_waveforms(pbar=True)

            if len(event_train) > 0:
                event_station_groups_by_dataset.append(
                    self._build_event_station_groups(
                        event_train,
                        dataset_name=dataset_name,
                        global_offset=sum(event_lengths),
                    )
                )
                event_generator = build_generator(event_train)
                event_generator.add_augmentations(
                    self._build_train_event_augmentations(
                        dataset_id=dataset_id,
                        dataset_name=dataset_name,
                    )
                )
                event_generators.append(event_generator)
                event_lengths.append(len(event_generator))
            else:
                event_station_groups_by_dataset.append([])

            if noise_fraction > 0 and dataset_name.lower() == noise_dataset_name:
                noise_train = self._copy_dataset(train_split)
                filter_noise_traces(noise_train)
                if len(noise_train) > 0:
                    if self.cache:
                        noise_train.preload_waveforms(pbar=True)
                    noise_generator = build_generator(noise_train)
                    noise_generator.add_augmentations(
                        self._build_train_noise_augmentations(
                            dataset_id=dataset_id,
                            dataset_name=dataset_name,
                        )
                    )
                    noise_generators.append(noise_generator)
                    noise_lengths.append(len(noise_generator))

            val_generator = build_generator(val_split)
            val_generator.add_augmentations(
                self._build_val_augmentations(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                )
            )
            self.val_generators.append(val_generator)
            self.val_loader_names.append(dataset_name)

        train_generators = event_generators + noise_generators
        train_lengths = event_lengths + noise_lengths
        if not train_generators:
            raise ValueError("No pretraining samples remain after event/noise filtering.")
        self.train_dataset = ConcatDataset(train_generators)
        configured_num_samples = int(self.dataset_sampling.get("num_samples_per_epoch", 0))
        num_samples = configured_num_samples if configured_num_samples > 0 else len(self.train_dataset)
        if self._event_pairing_enabled():
            self.train_batch_sampler = self._build_event_paired_batch_sampler(
                event_station_groups_by_dataset=event_station_groups_by_dataset,
                num_samples=num_samples,
            )
            self.train_sampler = None
        else:
            group_probabilities = self._build_train_group_probabilities(
                event_lengths=event_lengths,
                noise_lengths=noise_lengths,
            )
            self.train_sampler = make_group_sampler(
                lengths=train_lengths,
                group_probabilities=group_probabilities,
                num_samples=num_samples,
            )
            self.train_batch_sampler = None

    def train_dataloader(self) -> DataLoader:
        if self.train_batch_sampler is not None:
            kwargs = dict(
                dataset=self.train_dataset,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=get_worker_init_fn(),
                persistent_workers=self.num_workers > 0,
            )
            if self.num_workers > 0 and self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
            return DataLoader(**kwargs)

        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=get_worker_init_fn(),
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(
            **kwargs,
        )

    def val_dataloader(self) -> list[DataLoader]:
        return [
            self._build_val_loader(generator)
            for generator in self.val_generators
        ]

    def _build_train_event_augmentations(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
    ) -> list[Any]:
        event_cfg = self.window["event"]
        return [
            EventWindowSelector(
                window_length=self.window["length_samples"],
                random_probability=event_cfg["random_probability"],
                phase_containing_probability=event_cfg["phase_containing_probability"],
                pick_margin_sec=event_cfg["pick_margin_sec"],
                sampling_rate=self.sampling_rate,
                deterministic=False,
            ),
            *self._build_post_window_augmentations(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                deterministic=False,
            ),
        ]

    def _build_train_noise_augmentations(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
    ) -> list[Any]:
        return [
            NoiseWindowSelector(
                window_length=self.window["length_samples"],
                sampling_rate=self.sampling_rate,
                deterministic=False,
            ),
            *self._build_post_window_augmentations(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                deterministic=False,
            ),
        ]

    def _build_val_augmentations(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
    ) -> list[Any]:
        val_cfg = self.window["validation"]
        return [
            PretrainValidationWindowSelector(
                window_length=self.window["length_samples"],
                sampling_rate=self.sampling_rate,
                p_position_samples=val_cfg["p_position_samples"],
                s_position_samples=val_cfg["s_position_samples"],
                deterministic_manifest=bool(val_cfg.get("deterministic_manifest", True)),
            ),
            *self._build_post_window_augmentations(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                deterministic=True,
            ),
        ]

    def _build_post_window_augmentations(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
        deterministic: bool,
    ) -> list[Any]:
        augmentations: list[Any] = [
            CaptureValidComponentMask(),
        ]
        if self._anchor_cache_enabled():
            augmentations.append(
                AddPseudoArrivalMetadata(
                    window_length=int(self.window["length_samples"]),
                )
            )
        augmentations.extend(
            [
            FillMissingComponents(deterministic=deterministic),
            StdSafeNormalize(
                amp_norm_type=self.amp_norm_type,
                eps=self.norm_eps,
            ),
            ChangeDtype(),
            AddConstantMetadata("dataset_id", dataset_id),
            AddEventMetadata(dataset_name=dataset_name),
            ]
        )
        return augmentations

    def _inject_anchor_cache_if_enabled(
        self,
        split_dataset: Any,
        *,
        dataset_name: str,
        split_name: str,
    ) -> None:
        if not self._anchor_cache_enabled() or split_name not in self._anchor_cache_splits():
            return
        payload = ensure_pseudo_arrival_cache(
            split_dataset,
            dataset_name=dataset_name,
            split=split_name,
            sampling_rate=self.sampling_rate,
            component_order=self.component_order,
            dimension_order=self.dimension_order,
            cache_cfg=self.anchor_cache,
        )
        inject_pseudo_arrival_cache(split_dataset, payload)

    def _anchor_cache_enabled(self) -> bool:
        return bool(self.anchor_cache.get("enabled", False))

    def _anchor_cache_splits(self) -> tuple[str, ...]:
        splits = self.anchor_cache.get("splits", ("train", "dev"))
        return tuple(str(split) for split in splits)

    @staticmethod
    def _dataset_split(dataset: Any, split_name: str) -> Any:
        split_name = str(split_name)
        method_name = {
            "val": "dev",
            "valid": "dev",
            "validation": "dev",
        }.get(split_name, split_name)
        if not hasattr(dataset, method_name):
            raise ValueError(f"Unsupported pseudo-arrival cache split: {split_name}")
        return getattr(dataset, method_name)()

    def _build_train_group_probabilities(
        self,
        *,
        event_lengths: list[int],
        noise_lengths: list[int],
    ) -> np.ndarray:
        event_fraction = float(
            self.batch_sampling.get("event_fraction", self.batch_sampling.get("earthquake_fraction", 1.0))
        )
        noise_fraction = float(self.batch_sampling.get("noise_fraction", 0.0))
        if event_fraction < 0 or noise_fraction < 0:
            raise ValueError("earthquake_fraction/event_fraction and noise_fraction must be non-negative.")
        if event_fraction > 0 and not event_lengths:
            raise ValueError("earthquake_fraction/event_fraction > 0 but no event traces are available.")
        if noise_fraction > 0 and not noise_lengths:
            raise ValueError(
                "noise_fraction > 0 but no matching noise traces are available. "
                "Check batch_sampling.noise_dataset and trace_category metadata."
            )

        temperature = float(self.dataset_sampling.get("temperature", 0.0))
        group_probabilities: list[float] = []
        if event_lengths:
            event_probs = build_temperature_probabilities(
                lengths=event_lengths,
                temperature=temperature,
            )
            group_probabilities.extend((event_fraction * event_probs).tolist())
        if noise_lengths:
            noise_probs = build_temperature_probabilities(
                lengths=noise_lengths,
                temperature=0.0,
            )
            group_probabilities.extend((noise_fraction * noise_probs).tolist())
        return np.asarray(group_probabilities, dtype=np.float64)

    def _event_pairing_enabled(self) -> bool:
        pairing_cfg = self.batch_sampling.get("event_pairing", {})
        if not isinstance(pairing_cfg, Mapping):
            return False
        return bool(pairing_cfg.get("enabled", False))

    def _build_event_paired_batch_sampler(
        self,
        *,
        event_station_groups_by_dataset: list[list[list[list[int]]]],
        num_samples: int,
    ) -> EventPairedBatchSampler:
        noise_fraction = float(self.batch_sampling.get("noise_fraction", 0.0))
        if noise_fraction > 0:
            raise ValueError("event_pairing is only supported with noise_fraction=0.0.")

        pairing_cfg = self.batch_sampling.get("event_pairing", {})
        views_per_event = int(pairing_cfg.get("views_per_event", 2))
        event_group_counts = [len(groups) for groups in event_station_groups_by_dataset]
        if not any(count > 0 for count in event_group_counts):
            raise ValueError("event_pairing enabled but no events with >=2 stations are available.")

        temperature = float(self.dataset_sampling.get("temperature", 0.0))
        dataset_probs = build_temperature_probabilities(
            lengths=[count for count in event_group_counts if count > 0],
            temperature=temperature,
        )
        event_groups: list[list[list[int]]] = []
        event_probs: list[float] = []
        prob_idx = 0
        for groups, count in zip(event_station_groups_by_dataset, event_group_counts):
            if count <= 0:
                continue
            dataset_prob = float(dataset_probs[prob_idx])
            prob_idx += 1
            per_event_prob = dataset_prob / float(count)
            event_groups.extend(groups)
            event_probs.extend([per_event_prob] * count)

        num_batches = max(int(num_samples) // int(self.batch_size), 1)
        return EventPairedBatchSampler(
            event_station_groups=event_groups,
            group_probabilities=event_probs,
            batch_size=int(self.batch_size),
            views_per_event=views_per_event,
            num_batches=num_batches,
        )

    def _build_event_station_groups(
        self,
        dataset: Any,
        *,
        dataset_name: str,
        global_offset: int,
    ) -> list[list[list[int]]]:
        event_to_stations: dict[int, dict[int, list[int]]] = {}
        for local_idx, (_, row) in enumerate(dataset.metadata.iterrows()):
            event_uid, station_uid = event_station_uids(row, dataset_name=dataset_name)
            if event_uid == MISSING_UID or station_uid == MISSING_UID:
                continue
            station_to_indices = event_to_stations.setdefault(int(event_uid), {})
            station_to_indices.setdefault(int(station_uid), []).append(int(global_offset + local_idx))

        groups: list[list[list[int]]] = []
        for station_to_indices in event_to_stations.values():
            if len(station_to_indices) < 2:
                continue
            groups.append(list(station_to_indices.values()))
        return groups

    @staticmethod
    def _copy_dataset(dataset: Any) -> Any:
        return dataset.filter(np.ones(len(dataset), dtype=bool), inplace=False)

    def _build_val_loader(self, generator: Any) -> DataLoader:
        kwargs = dict(
            dataset=generator,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=get_worker_init_fn(),
            persistent_workers=self.num_workers > 0,
        )
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**kwargs)
