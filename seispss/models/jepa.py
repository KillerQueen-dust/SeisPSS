"""Full SeisPSS model."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from seispss.models.encoders import build_encoder
from seispss.models.masking import (
    PseudoArrivalTargetMaskGenerator1D,
    SpanMask,
    TimeSpanMaskGenerator1D,
    gather_span_tokens,
)
from seispss.models.positional_encoding import build_1d_sincos_position_embedding
from seispss.models.predictor import JEPAPredictor
from seispss.models.stems import ConvStem1D


@dataclass
class JEPAForwardOutput:
    predictions: torch.Tensor
    targets: torch.Tensor
    mask: SpanMask
    valid_token_mask: torch.Tensor


JEPA_OBJECTIVE = "jepa"
MASKED_RECONSTRUCTION_OBJECTIVE = "masked_reconstruction"


class SeisPSS(nn.Module):
    def __init__(
        self,
        *,
        model_cfg: dict,
        mask_cfg: dict,
    ) -> None:
        super().__init__()
        self.online_stem = ConvStem1D(
            in_channels=model_cfg["in_channels"],
            channels=list(model_cfg["stem_channels"]),
            kernel_sizes=list(model_cfg["stem_kernel_sizes"]),
            strides=list(model_cfg["stem_strides"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.target_stem = copy.deepcopy(self.online_stem)
        for parameter in self.target_stem.parameters():
            parameter.requires_grad = False

        embed_dim = self.online_stem.output_dim
        self.context_encoder = build_encoder(model_cfg=model_cfg, dim=embed_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False

        self.context_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.context_mask_token, mean=0.0, std=0.02)

        # Objective selector. "jepa" predicts EMA-teacher latent targets (default, unchanged);
        # "masked_reconstruction" predicts the raw waveform patches under the masked tokens.
        # Only the target and the predictor's output width differ -- stem, encoder, masking,
        # guard bands, positional encoding and predictor body are shared, so the two objectives
        # form a clean controlled ablation.
        self.objective = str(model_cfg.get("objective", JEPA_OBJECTIVE)).lower()
        if self.objective not in {JEPA_OBJECTIVE, MASKED_RECONSTRUCTION_OBJECTIVE}:
            raise ValueError(f"Unsupported model.objective: {self.objective}")
        self.sample_to_token_stride = 1
        for stride in model_cfg["stem_strides"]:
            self.sample_to_token_stride *= int(stride)
        self.reconstruction_dim = int(model_cfg["in_channels"]) * self.sample_to_token_stride
        self.reconstruction_normalize_target = str(
            model_cfg.get("reconstruction_normalize_target", "none")
        ).lower()

        self.mask_type = str(mask_cfg.get("type", "time_span")).lower()
        self.mask_generator = self._build_mask_generator(
            model_cfg=model_cfg,
            mask_cfg=mask_cfg,
        )
        self.sample_mask_on_cpu = bool(mask_cfg.get("sample_on_cpu", True))
        predictor_output_dim = (
            self.reconstruction_dim if self.objective == MASKED_RECONSTRUCTION_OBJECTIVE else embed_dim
        )
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=int(model_cfg["predictor_dim"]),
            depth=int(model_cfg["predictor_depth"]),
            num_heads=int(model_cfg["predictor_num_heads"]),
            mlp_ratio=float(model_cfg["predictor_mlp_ratio"]),
            dropout=float(model_cfg["dropout"]),
            output_dim=predictor_output_dim,
        )
        self._set_target_eval()

    def train(self, mode: bool = True) -> "SeisPSS":
        super().train(mode)
        self._set_target_eval()
        return self

    def forward(
        self,
        *,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor | None = None,
        pseudo_anchor_sample: torch.Tensor | None = None,
        pseudo_anchor_confidence: torch.Tensor | None = None,
        pseudo_anchor_valid: torch.Tensor | None = None,
    ) -> JEPAForwardOutput:
        tokens, valid_token_mask = self.online_stem(x, valid_time_mask)
        pos_emb = build_1d_sincos_position_embedding(
            length=tokens.shape[1],
            dim=tokens.shape[2],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        full_tokens = tokens + pos_emb
        full_tokens = full_tokens.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        mask_input = valid_token_mask
        anchor_sample_input = pseudo_anchor_sample
        anchor_confidence_input = pseudo_anchor_confidence
        anchor_valid_input = pseudo_anchor_valid
        if self.sample_mask_on_cpu and valid_token_mask.device.type != "cpu":
            mask_input = valid_token_mask.detach().cpu()
            anchor_sample_input = None if pseudo_anchor_sample is None else pseudo_anchor_sample.detach().cpu()
            anchor_confidence_input = (
                None if pseudo_anchor_confidence is None else pseudo_anchor_confidence.detach().cpu()
            )
            anchor_valid_input = None if pseudo_anchor_valid is None else pseudo_anchor_valid.detach().cpu()
        if self.mask_type == "pseudo_arrival":
            mask = self.mask_generator(
                valid_token_mask=mask_input,
                pseudo_anchor_sample=anchor_sample_input,
                pseudo_anchor_confidence=anchor_confidence_input,
                pseudo_anchor_valid=anchor_valid_input,
            )
        else:
            mask = self.mask_generator(valid_token_mask=mask_input)
        if mask.target_indices.device != tokens.device:
            mask = mask.to(device=tokens.device, valid_token_mask=valid_token_mask)

        if self.objective == MASKED_RECONSTRUCTION_OBJECTIVE:
            targets = self._gather_raw_targets(x=x, mask=mask, num_tokens=tokens.shape[1])
        else:
            with torch.no_grad():
                target_tokens, _ = self.target_stem(x, valid_time_mask)
                target_tokens = target_tokens + pos_emb
                target_tokens = target_tokens.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
                target_hidden = self.target_encoder(target_tokens, valid_token_mask=valid_token_mask)
                target_hidden = F.layer_norm(target_hidden, normalized_shape=(target_hidden.shape[-1],))
                targets = gather_span_tokens(target_hidden, mask.target_indices).flatten(1, 2)

        context_input = full_tokens.clone()
        context_mask_value = self.context_mask_token.to(dtype=context_input.dtype)
        target_mask_value = context_mask_value + pos_emb
        context_drop_mask = valid_token_mask & ~mask.context_visible_mask
        context_input = torch.where(context_drop_mask.unsqueeze(-1), target_mask_value, context_input)
        context_input = context_input.masked_fill(~valid_token_mask.unsqueeze(-1), 0.0)
        context_hidden = self.context_encoder(context_input, valid_token_mask=valid_token_mask)

        predictions = self.predictor(
            context_hidden=context_hidden,
            pos_emb=pos_emb.expand(x.shape[0], -1, -1),
            memory_key_padding_mask=~mask.context_visible_mask,
            target_indices=mask.target_indices,
        )
        return JEPAForwardOutput(
            predictions=predictions,
            targets=targets,
            mask=mask,
            valid_token_mask=valid_token_mask,
        )

    def _gather_raw_targets(
        self,
        *,
        x: torch.Tensor,
        mask: SpanMask,
        num_tokens: int,
    ) -> torch.Tensor:
        """Raw waveform patch under each target token: the reconstruction target.

        Token j covers input samples [stride*j, stride*(j+1)); this is a clean partition
        because T is an exact multiple of the total stem stride. The guard bands (wider than
        the stem receptive field) already remove every visible token whose receptive field
        overlaps a target span, so the encoder cannot see these samples -- the same property
        that keeps the JEPA objective leakage-free keeps reconstruction leakage-free.
        """
        batch_size, channels, length = x.shape
        stride = self.sample_to_token_stride
        usable = num_tokens * stride
        patches = (
            x[:, :, :usable]
            .reshape(batch_size, channels, num_tokens, stride)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_tokens, channels * stride)
        )
        targets = gather_span_tokens(patches, mask.target_indices).flatten(1, 2)
        if self.reconstruction_normalize_target == "per_patch":
            mean = targets.mean(dim=-1, keepdim=True)
            var = targets.var(dim=-1, unbiased=False, keepdim=True)
            targets = (targets - mean) / torch.sqrt(var + 1.0e-6)
        return targets

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        # No EMA teacher under masked reconstruction: the target is the fixed raw waveform,
        # so the target branch is never used and must not drift. Skipping keeps the training
        # loop (and thus the pretraining module) identical to the JEPA run in every other way.
        if self.objective == MASKED_RECONSTRUCTION_OBJECTIVE:
            return
        for online_param, target_param in zip(self.online_stem.parameters(), self.target_stem.parameters()):
            target_param.data.mul_(momentum).add_((1.0 - momentum) * online_param.detach().data)
        for online_param, target_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.mul_(momentum).add_((1.0 - momentum) * online_param.detach().data)

    def _set_target_eval(self) -> None:
        self.target_stem.eval()
        self.target_encoder.eval()

    def backbone_parameters(self):
        yield from self.online_stem.parameters()
        yield from self.context_encoder.parameters()
        yield self.context_mask_token

    def predictor_parameters(self):
        yield from self.predictor.parameters()

    @staticmethod
    def _build_mask_generator(*, model_cfg: dict, mask_cfg: dict) -> nn.Module:
        mask_type = str(mask_cfg.get("type", "time_span")).lower()
        if mask_type == "time_span":
            return TimeSpanMaskGenerator1D(
                num_targets=int(mask_cfg["num_targets"]),
                span_length=int(mask_cfg["span_length_tokens"]),
                min_gap_tokens=int(mask_cfg["min_gap_tokens"]),
                context_gap_tokens=int(mask_cfg.get("context_gap_tokens", 0)),
            )
        if mask_type == "pseudo_arrival":
            sample_to_token_stride = 1
            for stride in model_cfg["stem_strides"]:
                sample_to_token_stride *= int(stride)
            return PseudoArrivalTargetMaskGenerator1D(
                target_lengths_tokens=list(mask_cfg["target_lengths_tokens"]),
                guard_ratios=list(mask_cfg["guard_ratios"]),
                stem_guard_tokens=int(mask_cfg.get("stem_guard_tokens", 0)),
                horizons=dict(mask_cfg["horizons"]),
                sample_to_token_stride=sample_to_token_stride,
                min_visible_ratio=float(mask_cfg.get("min_visible_ratio", 0.55)),
                max_resample_attempts=int(mask_cfg.get("max_resample_attempts", 32)),
                alpha_max=float(mask_cfg.get("alpha_max", 0.65)),
                anchor_jitter_fraction=float(mask_cfg.get("anchor_jitter_fraction", 0.015)),
                anchor_mode=str(mask_cfg.get("anchor_mode", "full")),
            )
        raise ValueError(f"Unsupported mask.type: {mask_type}")

    def get_backbone_state_dict(self, *, variant: str) -> dict[str, torch.Tensor]:
        variant = variant.lower()
        if variant == "online":
            stem = self.online_stem
            encoder = self.context_encoder
        elif variant == "ema":
            stem = self.target_stem
            encoder = self.target_encoder
        else:
            raise ValueError(f"Unsupported backbone variant: {variant}")

        state_dict: dict[str, torch.Tensor] = {}
        for key, value in stem.state_dict().items():
            state_dict[f"stem.{key}"] = value.detach().cpu()
        for key, value in encoder.state_dict().items():
            state_dict[f"encoder.{key}"] = value.detach().cpu()
        return state_dict
