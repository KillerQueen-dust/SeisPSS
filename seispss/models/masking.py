"""1D time-span masking utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpanMask:
    target_indices: torch.LongTensor
    target_mask: torch.BoolTensor
    context_visible_mask: torch.BoolTensor
    valid_token_mask: torch.BoolTensor
    guard_mask: torch.BoolTensor | None = None
    target_horizon_ids: torch.LongTensor | None = None
    anchor_token: torch.LongTensor | None = None
    anchor_confidence: torch.FloatTensor | None = None
    anchored_branch_mask: torch.BoolTensor | None = None

    def to(
        self,
        *,
        device: torch.device | str,
        valid_token_mask: torch.Tensor | None = None,
    ) -> "SpanMask":
        return SpanMask(
            target_indices=self.target_indices.to(device=device),
            target_mask=self.target_mask.to(device=device),
            context_visible_mask=self.context_visible_mask.to(device=device),
            valid_token_mask=valid_token_mask if valid_token_mask is not None else self.valid_token_mask.to(device=device),
            guard_mask=None if self.guard_mask is None else self.guard_mask.to(device=device),
            target_horizon_ids=None if self.target_horizon_ids is None else self.target_horizon_ids.to(device=device),
            anchor_token=None if self.anchor_token is None else self.anchor_token.to(device=device),
            anchor_confidence=None if self.anchor_confidence is None else self.anchor_confidence.to(device=device),
            anchored_branch_mask=None
            if self.anchored_branch_mask is None
            else self.anchored_branch_mask.to(device=device),
        )


def gather_tokens(sequence: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(*indices.shape, sequence.shape[-1])
    return torch.gather(sequence, dim=1, index=gather_index)


def gather_span_tokens(sequence: torch.Tensor, span_indices: torch.Tensor) -> torch.Tensor:
    batch_size, num_targets, span_length = span_indices.shape
    flat_indices = span_indices.reshape(batch_size, num_targets * span_length)
    gathered = gather_tokens(sequence, flat_indices)
    return gathered.reshape(batch_size, num_targets, span_length, sequence.shape[-1])


def span_indices_to_mask(span_indices: torch.Tensor, sequence_length: int) -> torch.Tensor:
    mask = torch.zeros(
        span_indices.shape[0],
        sequence_length,
        device=span_indices.device,
        dtype=torch.bool,
    )
    flat_indices = span_indices.reshape(span_indices.shape[0], -1)
    mask.scatter_(dim=1, index=flat_indices, value=True)
    return mask


class TimeSpanMaskGenerator1D(nn.Module):
    """Sample non-overlapping target spans within valid tokens."""

    def __init__(
        self,
        *,
        num_targets: int,
        span_length: int,
        min_gap_tokens: int,
        context_gap_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_targets = num_targets
        self.span_length = span_length
        self.min_gap_tokens = min_gap_tokens
        self.context_gap_tokens = int(context_gap_tokens)
        if self.context_gap_tokens < 0:
            raise ValueError("context_gap_tokens must be non-negative.")

    def forward(self, valid_token_mask: torch.Tensor) -> SpanMask:
        batch_size, sequence_length = valid_token_mask.shape
        candidate_starts_per_sample = self._find_candidate_starts(valid_token_mask)
        target_indices = [
            self._sample_single(
                candidate_starts=candidate_starts_per_sample[idx],
                sequence_length=sequence_length,
                device=valid_token_mask.device,
            )
            for idx in range(batch_size)
        ]
        target_indices_tensor = torch.stack(target_indices, dim=0)
        target_mask = span_indices_to_mask(target_indices_tensor, sequence_length)
        context_block_mask = self._context_block_mask(target_mask)
        context_visible_mask = valid_token_mask & ~context_block_mask
        return SpanMask(
            target_indices=target_indices_tensor,
            target_mask=target_mask,
            context_visible_mask=context_visible_mask,
            valid_token_mask=valid_token_mask,
            guard_mask=context_block_mask & ~target_mask,
        )

    def _find_candidate_starts(self, valid_token_mask: torch.Tensor) -> list[torch.Tensor]:
        batch_size, sequence_length = valid_token_mask.shape
        max_start = sequence_length - self.span_length + 1
        if max_start <= 0:
            raise ValueError("span_length exceeds the token sequence length.")

        if bool(valid_token_mask.all()):
            full_range = torch.arange(max_start, device=valid_token_mask.device, dtype=torch.long)
            return [full_range] * batch_size

        span_kernel = torch.ones(
            1,
            1,
            self.span_length,
            device=valid_token_mask.device,
            dtype=torch.float32,
        )
        valid_spans = F.conv1d(valid_token_mask[:, None, :].to(dtype=torch.float32), span_kernel).squeeze(1)
        return [torch.nonzero(valid_spans[idx] == self.span_length, as_tuple=False).flatten() for idx in range(batch_size)]

    def _sample_single(
        self,
        *,
        candidate_starts: torch.Tensor,
        sequence_length: int,
        device: torch.device,
    ) -> torch.LongTensor:
        del sequence_length

        if candidate_starts.numel() == 0:
            raise ValueError(
                "No valid target span can be sampled. Reduce span_length or ensure longer valid windows."
            )

        starts = self._pick_non_overlapping_starts(candidate_starts.tolist(), device=device)
        spans = [
            torch.arange(start, start + self.span_length, device=device, dtype=torch.long)
            for start in starts
        ]
        return torch.stack(spans, dim=0)

    def _pick_non_overlapping_starts(
        self,
        candidate_starts: list[int],
        *,
        device: torch.device,
    ) -> list[int]:
        for gap in range(self.min_gap_tokens, -1, -1):
            perm = torch.randperm(len(candidate_starts), device=device)
            selected: list[int] = []
            for idx in perm.tolist():
                start = candidate_starts[idx]
                if all(abs(start - prev) >= (self.span_length + gap) for prev in selected):
                    selected.append(start)
                if len(selected) == self.num_targets:
                    return sorted(selected)

        perm = torch.randperm(len(candidate_starts), device=device)
        selected = [candidate_starts[idx] for idx in perm[: self.num_targets].tolist()]
        while len(selected) < self.num_targets:
            selected.append(selected[-1])
        return sorted(selected)

    def _context_block_mask(self, target_mask: torch.Tensor) -> torch.Tensor:
        if self.context_gap_tokens <= 0:
            return target_mask
        kernel_size = 2 * self.context_gap_tokens + 1
        kernel = torch.ones(
            1,
            1,
            kernel_size,
            device=target_mask.device,
            dtype=torch.float32,
        )
        expanded = F.conv1d(
            target_mask[:, None, :].to(dtype=torch.float32),
            kernel,
            padding=self.context_gap_tokens,
        )
        return expanded.squeeze(1) > 0


class PseudoArrivalTargetMaskGenerator1D(nn.Module):
    """Sample multi-horizon targets around pseudo-arrival anchors."""

    def __init__(
        self,
        *,
        target_lengths_tokens: list[int],
        guard_ratios: list[float],
        stem_guard_tokens: int,
        horizons: dict[str, dict[str, object]],
        sample_to_token_stride: int,
        min_visible_ratio: float = 0.55,
        max_resample_attempts: int = 32,
        alpha_max: float = 0.65,
        anchor_jitter_fraction: float = 0.015,
        anchor_mode: str = "full",
    ) -> None:
        super().__init__()
        self.target_lengths = [int(length) for length in target_lengths_tokens]
        if not self.target_lengths or any(length <= 0 for length in self.target_lengths):
            raise ValueError("target_lengths_tokens must contain positive integers.")
        if len(guard_ratios) != len(self.target_lengths):
            raise ValueError("guard_ratios must have the same length as target_lengths_tokens.")
        self.guard_tokens = [
            max(int(stem_guard_tokens), int(math.ceil(float(ratio) * length)))
            for ratio, length in zip(guard_ratios, self.target_lengths)
        ]
        horizon_items = list(horizons.items())
        if len(horizon_items) != len(self.target_lengths):
            raise ValueError("horizons must define one entry per target length.")
        self.horizon_names = [name for name, _ in horizon_items]
        self.center_scales = [str(cfg.get("center_scale", "sequence")) for _, cfg in horizon_items]
        self.center_ranges = [
            (float(cfg.get("center_range", [0.0, 1.0])[0]), float(cfg.get("center_range", [0.0, 1.0])[1]))
            for _, cfg in horizon_items
        ]
        self.sample_to_token_stride = max(int(sample_to_token_stride), 1)
        self.min_visible_ratio = float(min_visible_ratio)
        self.max_resample_attempts = max(int(max_resample_attempts), 1)
        self.alpha_max = float(alpha_max)
        self.anchor_jitter_fraction = float(anchor_jitter_fraction)
        self.anchor_mode = str(anchor_mode).lower()
        if self.anchor_mode not in {"full", "random", "shuffled"}:
            raise ValueError("anchor_mode must be one of: full, random, shuffled.")

    def forward(
        self,
        valid_token_mask: torch.Tensor,
        pseudo_anchor_sample: torch.Tensor | None = None,
        pseudo_anchor_confidence: torch.Tensor | None = None,
        pseudo_anchor_valid: torch.Tensor | None = None,
    ) -> SpanMask:
        valid_token_mask = valid_token_mask.bool()
        batch_size, sequence_length = valid_token_mask.shape
        device = valid_token_mask.device

        if self.anchor_mode == "shuffled" and pseudo_anchor_sample is not None:
            permutation = torch.randperm(batch_size, device=device)
            pseudo_anchor_sample = pseudo_anchor_sample.to(device=device)[permutation]
            if pseudo_anchor_confidence is not None:
                pseudo_anchor_confidence = pseudo_anchor_confidence.to(device=device)[permutation]
            if pseudo_anchor_valid is not None:
                pseudo_anchor_valid = pseudo_anchor_valid.to(device=device)[permutation]

        anchor_token, anchor_confidence = self._prepare_anchor_inputs(
            valid_token_mask=valid_token_mask,
            pseudo_anchor_sample=pseudo_anchor_sample,
            pseudo_anchor_confidence=pseudo_anchor_confidence,
            pseudo_anchor_valid=pseudo_anchor_valid,
        )

        allow_anchor = self.anchor_mode != "random"
        layout = self._sample_batch(
            valid_token_mask=valid_token_mask,
            anchor_token=anchor_token,
            anchor_confidence=anchor_confidence,
            allow_anchor=allow_anchor,
        )

        return SpanMask(
            target_indices=layout["target_indices"],
            target_mask=layout["target_mask"],
            context_visible_mask=layout["context_visible_mask"],
            valid_token_mask=valid_token_mask,
            guard_mask=layout["guard_mask"],
            target_horizon_ids=layout["horizon_ids"],
            anchor_token=anchor_token,
            anchor_confidence=anchor_confidence,
            anchored_branch_mask=layout["anchored_branch_mask"],
        )

    def _prepare_anchor_inputs(
        self,
        *,
        valid_token_mask: torch.Tensor,
        pseudo_anchor_sample: torch.Tensor | None,
        pseudo_anchor_confidence: torch.Tensor | None,
        pseudo_anchor_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = valid_token_mask.shape
        device = valid_token_mask.device
        anchor_token = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        anchor_confidence = torch.zeros(batch_size, device=device, dtype=torch.float32)
        if pseudo_anchor_sample is None:
            return anchor_token, anchor_confidence

        samples = pseudo_anchor_sample.to(device=device).view(-1).float()
        confidence = (
            pseudo_anchor_confidence.to(device=device).view(-1).float()
            if pseudo_anchor_confidence is not None
            else torch.ones_like(samples)
        )
        valid = samples >= 0
        if pseudo_anchor_valid is not None:
            valid = valid & pseudo_anchor_valid.to(device=device).view(-1).bool()
        tokens = torch.round(samples / float(self.sample_to_token_stride)).long()
        batch_indices = torch.arange(batch_size, device=device)
        valid = valid & (tokens >= 0) & (tokens < sequence_length)
        if bool(valid.any()):
            valid_indices = batch_indices[valid]
            valid = valid.clone()
            valid[valid_indices] = valid_token_mask[valid_indices, tokens[valid_indices]]
        anchor_token[valid] = tokens[valid]
        anchor_confidence[valid] = confidence[valid].clamp(0.0, 1.0)
        return anchor_token, anchor_confidence

    def _sample_batch(
        self,
        *,
        valid_token_mask: torch.Tensor,
        anchor_token: torch.Tensor,
        anchor_confidence: torch.Tensor,
        allow_anchor: bool,
    ) -> dict[str, torch.Tensor]:
        layout = self._build_layout_batch(
            valid_token_mask=valid_token_mask,
            anchor_token=anchor_token,
            anchor_confidence=anchor_confidence,
            allow_anchor=allow_anchor,
        )
        visible_ratio = self._visible_ratio_batch(
            valid_token_mask=valid_token_mask,
            context_visible_mask=layout["context_visible_mask"],
        )
        retry = visible_ratio < self.min_visible_ratio
        if bool(retry.any()):
            random_layout = self._build_layout_batch(
                valid_token_mask=valid_token_mask[retry],
                anchor_token=anchor_token[retry],
                anchor_confidence=anchor_confidence[retry],
                allow_anchor=False,
            )
            for key in layout:
                layout[key][retry] = random_layout[key]
        return layout

    def _build_layout_batch(
        self,
        *,
        valid_token_mask: torch.Tensor,
        anchor_token: torch.Tensor,
        anchor_confidence: torch.Tensor,
        allow_anchor: bool,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length = valid_token_mask.shape
        device = valid_token_mask.device
        target_mask = torch.zeros(batch_size, sequence_length, device=device, dtype=torch.bool)
        guard_mask = torch.zeros(batch_size, sequence_length, device=device, dtype=torch.bool)
        span_indices: list[torch.Tensor] = []
        horizon_ids: list[torch.Tensor] = []
        anchored_branch_mask = torch.zeros(
            batch_size,
            len(self.target_lengths),
            device=device,
            dtype=torch.bool,
        )

        for horizon_idx, (length, guard_width) in enumerate(zip(self.target_lengths, self.guard_tokens)):
            starts, use_anchor = self._initial_starts_batch(
                horizon_idx=horizon_idx,
                length=length,
                sequence_length=sequence_length,
                anchor_token=anchor_token,
                anchor_confidence=anchor_confidence,
                allow_anchor=allow_anchor,
            )
            placeable = self._span_is_placeable_batch(
                valid_token_mask=valid_token_mask,
                target_mask=target_mask,
                guard_mask=guard_mask,
                starts=starts,
                length=length,
                guard_width=guard_width,
            )
            unresolved = ~placeable
            if bool(unresolved.any()):
                use_anchor = use_anchor.clone()
                use_anchor[unresolved] = False
            for relaxed_guard in self._guard_schedule(guard_width):
                if not bool(unresolved.any()):
                    break
                retry_starts = self._random_starts(
                    batch_size=int(unresolved.sum().item()),
                    max_start=sequence_length - length,
                    device=device,
                )
                candidate_starts = starts.clone()
                candidate_starts[unresolved] = retry_starts
                candidate_placeable = self._span_is_placeable_batch(
                    valid_token_mask=valid_token_mask,
                    target_mask=target_mask,
                    guard_mask=guard_mask,
                    starts=candidate_starts,
                    length=length,
                    guard_width=relaxed_guard,
                )
                improved = unresolved & candidate_placeable
                starts[improved] = candidate_starts[improved]
                unresolved = unresolved & ~candidate_placeable

            if bool(unresolved.any()):
                starts = self._repair_unresolved_starts(
                    valid_token_mask=valid_token_mask,
                    target_mask=target_mask,
                    guard_mask=guard_mask,
                    starts=starts,
                    unresolved=unresolved,
                    length=length,
                    guard_width=guard_width,
                )
                use_anchor = use_anchor.clone()
                use_anchor[unresolved] = False

            span = starts[:, None] + torch.arange(length, device=device, dtype=torch.long)[None, :]
            span_indices.append(span)
            horizon_ids.append(torch.full((batch_size, length), horizon_idx, device=device, dtype=torch.long))
            anchored_branch_mask[:, horizon_idx] = use_anchor
            self._mark_span_batch(
                target_mask=target_mask,
                guard_mask=guard_mask,
                starts=starts,
                length=length,
                guard_width=guard_width,
            )

        guard_mask = guard_mask & ~target_mask
        context_visible_mask = valid_token_mask & ~(target_mask | guard_mask)
        return {
            "target_indices": torch.cat(span_indices, dim=1).unsqueeze(1),
            "target_mask": target_mask,
            "guard_mask": guard_mask,
            "context_visible_mask": context_visible_mask,
            "horizon_ids": torch.cat(horizon_ids, dim=1),
            "anchored_branch_mask": anchored_branch_mask,
        }

    def _initial_starts_batch(
        self,
        *,
        horizon_idx: int,
        length: int,
        sequence_length: int,
        anchor_token: torch.Tensor,
        anchor_confidence: torch.Tensor,
        allow_anchor: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(anchor_token.shape[0])
        device = anchor_token.device
        max_start = sequence_length - length
        if max_start < 0:
            raise ValueError("Pseudo-arrival target length exceeds token sequence length.")
        random_starts = self._random_starts(batch_size=batch_size, max_start=max_start, device=device)
        if not allow_anchor:
            return random_starts, torch.zeros(batch_size, device=device, dtype=torch.bool)

        anchor_ok = (anchor_token >= 0) & (anchor_confidence > 0.0)
        alpha = (self.alpha_max * anchor_confidence).clamp(0.0, 1.0)
        use_anchor = anchor_ok & (torch.rand(batch_size, device=device) < alpha)
        anchored_starts = self._anchored_starts_batch(
            horizon_idx=horizon_idx,
            length=length,
            sequence_length=sequence_length,
            anchor_token=anchor_token,
        )
        starts = torch.where(use_anchor, anchored_starts, random_starts)
        return starts, use_anchor

    def _anchored_starts_batch(
        self,
        *,
        horizon_idx: int,
        length: int,
        sequence_length: int,
        anchor_token: torch.Tensor,
    ) -> torch.Tensor:
        low, high = self.center_ranges[horizon_idx]
        device = anchor_token.device
        value = low + (high - low) * torch.rand(anchor_token.shape[0], device=device)
        scale = self.center_scales[horizon_idx]
        if scale == "sequence":
            offset = value * float(sequence_length)
        elif scale == "post_anchor":
            post_anchor = (sequence_length - 1 - anchor_token).clamp_min(1).to(dtype=torch.float32)
            offset = value * post_anchor
        else:
            raise ValueError(f"Unsupported pseudo-arrival horizon center_scale: {scale}")
        jitter_width = self.anchor_jitter_fraction * float(sequence_length)
        jitter = (2.0 * torch.rand(anchor_token.shape[0], device=device) - 1.0) * jitter_width
        center = anchor_token.to(dtype=torch.float32) + offset + jitter
        max_start = sequence_length - length
        return torch.round(center - (length - 1) / 2.0).long().clamp(0, max_start)

    @staticmethod
    def _random_starts(*, batch_size: int, max_start: int, device: torch.device) -> torch.Tensor:
        return torch.randint(max_start + 1, (batch_size,), device=device, dtype=torch.long)

    def _span_is_placeable_batch(
        self,
        *,
        valid_token_mask: torch.Tensor,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        starts: torch.Tensor,
        length: int,
        guard_width: int,
    ) -> torch.Tensor:
        ends = starts + length
        valid_count = self._interval_sum(valid_token_mask, starts, ends)
        blocked = target_mask | guard_mask
        blocked_count = self._interval_sum(blocked, starts, ends)
        guard_starts = (starts - guard_width).clamp_min(0)
        guard_ends = (ends + guard_width).clamp_max(valid_token_mask.shape[1])
        target_in_guard = self._interval_sum(target_mask, guard_starts, guard_ends)
        return (valid_count == length) & (blocked_count == 0) & (target_in_guard == 0)

    @staticmethod
    def _interval_sum(mask: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        prefix = torch.cat(
            [
                torch.zeros(mask.shape[0], 1, device=mask.device, dtype=torch.int32),
                mask.to(dtype=torch.int32).cumsum(dim=1),
            ],
            dim=1,
        )
        end_values = prefix.gather(1, ends[:, None]).squeeze(1)
        start_values = prefix.gather(1, starts[:, None]).squeeze(1)
        return end_values - start_values

    @staticmethod
    def _mark_span_batch(
        *,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        starts: torch.Tensor,
        length: int,
        guard_width: int,
    ) -> None:
        positions = torch.arange(target_mask.shape[1], device=target_mask.device)[None, :]
        ends = starts + length
        span = (positions >= starts[:, None]) & (positions < ends[:, None])
        guard_starts = (starts - guard_width).clamp_min(0)
        guard_ends = (ends + guard_width).clamp_max(target_mask.shape[1])
        guard = (positions >= guard_starts[:, None]) & (positions < guard_ends[:, None])
        target_mask |= span
        guard_mask |= guard

    def _repair_unresolved_starts(
        self,
        *,
        valid_token_mask: torch.Tensor,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        starts: torch.Tensor,
        unresolved: torch.Tensor,
        length: int,
        guard_width: int,
    ) -> torch.Tensor:
        repaired = starts.clone()
        for sample_idx in torch.nonzero(unresolved, as_tuple=False).flatten().tolist():
            start = self._place_random_span(
                valid_tokens=valid_token_mask[sample_idx],
                target_mask=target_mask[sample_idx],
                guard_mask=guard_mask[sample_idx],
                length=length,
                guard_width=guard_width,
            )
            if start is None:
                start = self._last_resort_span(valid_tokens=valid_token_mask[sample_idx], length=length)
            repaired[sample_idx] = int(start)
        return repaired

    @staticmethod
    def _visible_ratio_batch(
        *,
        valid_token_mask: torch.Tensor,
        context_visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_count = valid_token_mask.float().sum(dim=1).clamp_min(1.0)
        return context_visible_mask.float().sum(dim=1) / valid_count

    def _sample_single(
        self,
        *,
        valid_tokens: torch.Tensor,
        anchor_token: int,
        anchor_confidence: float,
        allow_anchor: bool,
    ) -> dict[str, torch.Tensor]:
        layout = self._build_layout(
            valid_tokens=valid_tokens,
            anchor_token=anchor_token,
            anchor_confidence=anchor_confidence,
            allow_anchor=allow_anchor,
        )
        visible_ratio = self._visible_ratio(
            valid_tokens=valid_tokens,
            context_visible_mask=layout["context_visible_mask"],
        )
        if visible_ratio >= self.min_visible_ratio:
            return layout

        for _ in range(self.max_resample_attempts):
            candidate = self._build_layout(
                valid_tokens=valid_tokens,
                anchor_token=anchor_token,
                anchor_confidence=anchor_confidence,
                allow_anchor=False,
            )
            if self._visible_ratio(valid_tokens=valid_tokens, context_visible_mask=candidate["context_visible_mask"]) >= self.min_visible_ratio:
                return candidate
        return layout

    def _build_layout(
        self,
        *,
        valid_tokens: torch.Tensor,
        anchor_token: int,
        anchor_confidence: float,
        allow_anchor: bool,
    ) -> dict[str, torch.Tensor]:
        device = valid_tokens.device
        sequence_length = int(valid_tokens.shape[0])
        target_mask = torch.zeros(sequence_length, device=device, dtype=torch.bool)
        guard_mask = torch.zeros(sequence_length, device=device, dtype=torch.bool)
        spans: list[torch.Tensor] = []
        horizon_ids: list[torch.Tensor] = []
        anchored_branch: list[bool] = []

        for horizon_idx, (length, guard_width) in enumerate(zip(self.target_lengths, self.guard_tokens)):
            use_anchor = False
            start = None
            if allow_anchor and anchor_token >= 0 and anchor_confidence > 0.0:
                alpha = min(max(self.alpha_max * anchor_confidence, 0.0), 1.0)
                use_anchor = bool(torch.rand((), device=device).item() < alpha)
                if use_anchor:
                    start = self._place_anchor_span(
                        valid_tokens=valid_tokens,
                        target_mask=target_mask,
                        guard_mask=guard_mask,
                        horizon_idx=horizon_idx,
                        length=length,
                        guard_width=guard_width,
                        anchor_token=anchor_token,
                    )
            if start is None:
                start = self._place_random_span(
                    valid_tokens=valid_tokens,
                    target_mask=target_mask,
                    guard_mask=guard_mask,
                    length=length,
                    guard_width=guard_width,
                )
                use_anchor = False
            if start is None:
                start = self._last_resort_span(valid_tokens=valid_tokens, length=length)
                use_anchor = False

            self._mark_span(
                target_mask=target_mask,
                guard_mask=guard_mask,
                start=start,
                length=length,
                guard_width=guard_width,
            )
            spans.append(torch.arange(start, start + length, device=device, dtype=torch.long))
            horizon_ids.append(torch.full((length,), horizon_idx, device=device, dtype=torch.long))
            anchored_branch.append(use_anchor)

        guard_mask = guard_mask & ~target_mask
        context_visible_mask = valid_tokens & ~(target_mask | guard_mask)
        return {
            "target_indices": torch.cat(spans, dim=0).view(1, -1),
            "target_mask": target_mask,
            "guard_mask": guard_mask,
            "context_visible_mask": context_visible_mask,
            "horizon_ids": torch.cat(horizon_ids, dim=0),
            "anchored_branch_mask": torch.tensor(anchored_branch, device=device, dtype=torch.bool),
        }

    def _place_anchor_span(
        self,
        *,
        valid_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        horizon_idx: int,
        length: int,
        guard_width: int,
        anchor_token: int,
    ) -> int | None:
        sequence_length = int(valid_tokens.shape[0])
        for relaxed_guard in self._guard_schedule(guard_width):
            for _ in range(self.max_resample_attempts):
                center = self._sample_anchor_center(
                    horizon_idx=horizon_idx,
                    anchor_token=anchor_token,
                    sequence_length=sequence_length,
                    device=valid_tokens.device,
                )
                start = int(round(center - (length - 1) / 2.0))
                start = min(max(start, 0), max(sequence_length - length, 0))
                if self._span_is_placeable(
                    valid_tokens=valid_tokens,
                    target_mask=target_mask,
                    guard_mask=guard_mask,
                    start=start,
                    length=length,
                    guard_width=relaxed_guard,
                ):
                    return start
        return None

    def _place_random_span(
        self,
        *,
        valid_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        length: int,
        guard_width: int,
    ) -> int | None:
        sequence_length = int(valid_tokens.shape[0])
        max_start = sequence_length - length
        if max_start < 0:
            raise ValueError("Pseudo-arrival target length exceeds token sequence length.")
        for relaxed_guard in self._guard_schedule(guard_width):
            candidate_starts = [
                start
                for start in range(max_start + 1)
                if self._span_is_placeable(
                    valid_tokens=valid_tokens,
                    target_mask=target_mask,
                    guard_mask=guard_mask,
                    start=start,
                    length=length,
                    guard_width=relaxed_guard,
                )
            ]
            if not candidate_starts:
                continue
            order = torch.randperm(len(candidate_starts), device=valid_tokens.device)
            return int(candidate_starts[int(order[0].item())])
        return None

    def _last_resort_span(self, *, valid_tokens: torch.Tensor, length: int) -> int:
        sequence_length = int(valid_tokens.shape[0])
        max_start = sequence_length - length
        if max_start < 0:
            raise ValueError("Pseudo-arrival target length exceeds token sequence length.")
        candidate_starts = [
            start
            for start in range(max_start + 1)
            if bool(valid_tokens[start : start + length].all())
        ]
        if not candidate_starts:
            raise ValueError("No valid pseudo-arrival target span can be sampled.")
        index = int(torch.randint(len(candidate_starts), (1,), device=valid_tokens.device).item())
        return int(candidate_starts[index])

    def _sample_anchor_center(
        self,
        *,
        horizon_idx: int,
        anchor_token: int,
        sequence_length: int,
        device: torch.device,
    ) -> float:
        low, high = self.center_ranges[horizon_idx]
        value = float((low + (high - low) * torch.rand((), device=device)).item())
        scale = self.center_scales[horizon_idx]
        if scale == "sequence":
            offset = value * float(sequence_length)
        elif scale == "post_anchor":
            offset = value * float(max(sequence_length - 1 - anchor_token, 1))
        else:
            raise ValueError(f"Unsupported pseudo-arrival horizon center_scale: {scale}")
        jitter_width = self.anchor_jitter_fraction * float(sequence_length)
        jitter = float(((2.0 * torch.rand((), device=device) - 1.0) * jitter_width).item())
        return float(anchor_token) + offset + jitter

    def _span_is_placeable(
        self,
        *,
        valid_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        start: int,
        length: int,
        guard_width: int,
    ) -> bool:
        end = start + length
        if start < 0 or end > int(valid_tokens.shape[0]):
            return False
        if not bool(valid_tokens[start:end].all()):
            return False
        blocked = target_mask | guard_mask
        if bool(blocked[start:end].any()):
            return False
        guard_start = max(start - guard_width, 0)
        guard_end = min(end + guard_width, int(valid_tokens.shape[0]))
        return not bool(target_mask[guard_start:guard_end].any())

    @staticmethod
    def _mark_span(
        *,
        target_mask: torch.Tensor,
        guard_mask: torch.Tensor,
        start: int,
        length: int,
        guard_width: int,
    ) -> None:
        end = start + length
        guard_start = max(start - guard_width, 0)
        guard_end = min(end + guard_width, int(target_mask.shape[0]))
        target_mask[start:end] = True
        guard_mask[guard_start:guard_end] = True

    @staticmethod
    def _guard_schedule(guard_width: int) -> list[int]:
        guard_width = max(int(guard_width), 0)
        schedule = [guard_width, guard_width // 2, 0]
        return list(dict.fromkeys(schedule))

    @staticmethod
    def _visible_ratio(*, valid_tokens: torch.Tensor, context_visible_mask: torch.Tensor) -> float:
        valid_count = float(valid_tokens.float().sum().item())
        if valid_count <= 0:
            return 0.0
        return float(context_visible_mask.float().sum().item()) / valid_count
