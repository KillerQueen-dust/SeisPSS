"""Sampling helpers for multi-dataset pretraining."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def build_temperature_weights(lengths: list[int], temperature: float) -> tuple[torch.DoubleTensor, np.ndarray]:
    dataset_probs = build_temperature_probabilities(lengths=lengths, temperature=temperature)
    weights = build_group_weights(lengths=lengths, group_probabilities=dataset_probs)
    return weights, dataset_probs


def build_temperature_probabilities(lengths: list[int], temperature: float) -> np.ndarray:
    if not lengths:
        raise ValueError("Expected at least one dataset length.")

    sizes = np.asarray(lengths, dtype=np.float64)
    if np.any(sizes <= 0):
        raise ValueError(f"Invalid dataset sizes: {lengths}")

    dataset_probs = np.power(sizes, temperature)
    dataset_probs /= dataset_probs.sum()
    return dataset_probs


def build_group_weights(lengths: list[int], group_probabilities: list[float] | np.ndarray) -> torch.DoubleTensor:
    if not lengths:
        raise ValueError("Expected at least one group length.")
    if len(lengths) != len(group_probabilities):
        raise ValueError("lengths and group_probabilities must have the same length.")

    sizes = np.asarray(lengths, dtype=np.float64)
    if np.any(sizes <= 0):
        raise ValueError(f"Invalid group sizes: {lengths}")

    group_probs = np.asarray(group_probabilities, dtype=np.float64)
    if np.any(group_probs < 0):
        raise ValueError(f"Group probabilities must be non-negative: {group_probabilities}")
    probability_sum = group_probs.sum()
    if probability_sum <= 0:
        raise ValueError("Group probabilities must sum to a positive value.")
    group_probs = group_probs / probability_sum

    per_sample_probs = group_probs / sizes
    weights = np.concatenate(
        [np.full(length, per_sample_probs[idx], dtype=np.float64) for idx, length in enumerate(lengths)]
    )
    return torch.as_tensor(weights, dtype=torch.double)


def make_temperature_sampler(
    *,
    lengths: list[int],
    temperature: float,
    num_samples: int,
) -> WeightedRandomSampler:
    weights, _ = build_temperature_weights(lengths=lengths, temperature=temperature)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


def make_group_sampler(
    *,
    lengths: list[int],
    group_probabilities: list[float] | np.ndarray,
    num_samples: int,
) -> WeightedRandomSampler:
    weights = build_group_weights(lengths=lengths, group_probabilities=group_probabilities)
    return WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)


class EventPairedBatchSampler(Sampler[list[int]]):
    """Sample batches containing multiple stations from the same event."""

    def __init__(
        self,
        *,
        event_station_groups: list[list[list[int]]],
        group_probabilities: list[float] | np.ndarray,
        batch_size: int,
        views_per_event: int,
        num_batches: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if views_per_event < 2:
            raise ValueError("views_per_event must be >= 2 for event-paired sampling.")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")
        if not event_station_groups:
            raise ValueError("event_station_groups must contain at least one eligible event.")
        if len(event_station_groups) != len(group_probabilities):
            raise ValueError("event_station_groups and group_probabilities must have the same length.")

        self.event_station_groups = event_station_groups
        self.batch_size = int(batch_size)
        self.views_per_event = int(views_per_event)
        self.num_batches = int(num_batches)
        probabilities = torch.as_tensor(group_probabilities, dtype=torch.double)
        if torch.any(probabilities < 0):
            raise ValueError("group_probabilities must be non-negative.")
        probability_sum = probabilities.sum()
        if float(probability_sum) <= 0:
            raise ValueError("group_probabilities must sum to a positive value.")
        self.group_probabilities = probabilities / probability_sum

    def __iter__(self):
        groups_per_batch = int(np.ceil(self.batch_size / self.views_per_event))
        for _ in range(self.num_batches):
            batch: list[int] = []
            group_ids = torch.multinomial(
                self.group_probabilities,
                num_samples=groups_per_batch,
                replacement=True,
            )
            for group_id in group_ids.tolist():
                batch.extend(self._sample_event_views(self.event_station_groups[group_id]))
                if len(batch) >= self.batch_size:
                    break
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches

    def _sample_event_views(self, station_groups: list[list[int]]) -> list[int]:
        station_count = len(station_groups)
        if station_count >= self.views_per_event:
            station_ids = torch.randperm(station_count)[: self.views_per_event].tolist()
        else:
            station_ids = torch.randperm(station_count).tolist()
            extra_count = self.views_per_event - station_count
            station_ids.extend(torch.randint(0, station_count, size=(extra_count,)).tolist())

        sample_indices: list[int] = []
        for station_id in station_ids:
            candidates = station_groups[station_id]
            candidate_id = int(torch.randint(0, len(candidates), size=(1,)).item())
            sample_indices.append(int(candidates[candidate_id]))
        return sample_indices
