"""Backbone and dense head for phase picking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from seispss.export import load_exported_backbone
from seispss.models.backbone import SeisPSSBackbone


class ConvPhasePickingHead(nn.Module):
    """Lightweight convolutional token head."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        padding = int(kernel_size) // 2
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Conv1d(current_dim, int(hidden_dim), kernel_size=int(kernel_size), padding=padding),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            current_dim = int(hidden_dim)
        self.net = nn.Sequential(*layers)
        self.classifier = nn.Conv1d(current_dim, int(num_classes), kernel_size=1)

    def forward(self, tokens: torch.Tensor, *, output_length: int) -> torch.Tensor:
        x = tokens.transpose(1, 2)
        logits = self.classifier(self.net(x))
        if logits.shape[-1] != int(output_length):
            logits = F.interpolate(
                logits,
                size=int(output_length),
                mode="linear",
                align_corners=False,
            )
        return logits


class PhasePickingModel(nn.Module):
    """SeisPSS backbone plus dense P/S/noise head."""

    num_output_classes = 3

    def __init__(
        self,
        *,
        model_cfg: dict[str, Any],
        head_cfg: dict[str, Any],
        transfer_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.transfer_cfg = transfer_cfg
        self.backbone, self.backbone_payload = self._build_backbone(
            model_cfg=model_cfg,
            transfer_cfg=transfer_cfg,
        )
        self.head = ConvPhasePickingHead(
            input_dim=self.backbone.stem.output_dim,
            num_classes=int(head_cfg["num_classes"]),
            hidden_dim=int(head_cfg["hidden_dim"]),
            num_layers=int(head_cfg["num_layers"]),
            kernel_size=int(head_cfg["kernel_size"]),
            dropout=float(head_cfg["dropout"]),
        )
        self.transfer_protocol = str(transfer_cfg.get("protocol", "full_finetune"))
        self.backbone_frozen = False
        self._apply_transfer_freezing(transfer_cfg=transfer_cfg)

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                hidden, _ = self.backbone(x, valid_time_mask=valid_time_mask)
        else:
            hidden, _ = self.backbone(x, valid_time_mask=valid_time_mask)
        return self.head(hidden, output_length=x.shape[-1])

    def set_frozen_backbone_eval(self) -> None:
        if self.backbone_frozen:
            self.backbone.eval()

    def backbone_parameters(self):
        yield from self.backbone.parameters()

    def head_parameters(self):
        yield from self.head.parameters()

    def optimizer_parameter_groups(self, optim_cfg) -> list[dict[str, Any]]:
        weight_decay = float(optim_cfg.weight_decay)
        return [
            {
                "params": list(self.backbone_parameters()),
                "lr": float(optim_cfg.backbone_lr),
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": list(self.head_parameters()),
                "lr": float(optim_cfg.head_lr),
                "weight_decay": weight_decay,
                "name": "head",
            },
        ]

    def sliding_inference(
        self,
        x: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor | None = None,
        window_samples: int,
        stride_samples: int,
        aggregation: str = "avg_logits",
    ) -> torch.Tensor:
        total_length = int(x.shape[-1])
        window_samples = int(window_samples)
        stride_samples = int(stride_samples)
        if total_length <= window_samples:
            return self.forward(x, valid_time_mask=valid_time_mask)
        if window_samples <= 0 or stride_samples <= 0:
            raise ValueError("window_samples and stride_samples must be positive.")

        starts = list(range(0, total_length - window_samples + 1, stride_samples))
        final_start = total_length - window_samples
        if starts[-1] != final_start:
            starts.append(final_start)

        num_classes = int(self.head.classifier.out_channels)
        aggregation = str(aggregation)
        weights = x.new_zeros((x.shape[0], 1, total_length))
        if aggregation == "avg_logits":
            logits_sum = x.new_zeros((x.shape[0], num_classes, total_length))
            for start in starts:
                end = start + window_samples
                window_mask = None if valid_time_mask is None else valid_time_mask[:, start:end]
                window_logits = self.forward(x[:, :, start:end], valid_time_mask=window_mask)
                logits_sum[:, :, start:end] += window_logits
                weights[:, :, start:end] += 1.0
            return logits_sum / weights.clamp_min(1.0)

        if aggregation == "avg_probs":
            probs_sum = x.new_zeros((x.shape[0], num_classes, total_length))
            for start in starts:
                end = start + window_samples
                window_mask = None if valid_time_mask is None else valid_time_mask[:, start:end]
                window_probs = torch.softmax(
                    self.forward(x[:, :, start:end], valid_time_mask=window_mask),
                    dim=1,
                )
                probs_sum[:, :, start:end] += window_probs
                weights[:, :, start:end] += 1.0
            probs = probs_sum / weights.clamp_min(1.0)
            return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()

        raise ValueError(f"Unsupported sliding aggregation: {aggregation}")

    @staticmethod
    def _build_backbone(
        *,
        model_cfg: dict[str, Any],
        transfer_cfg: dict[str, Any],
    ) -> tuple[SeisPSSBackbone, dict[str, Any] | None]:
        init = str(transfer_cfg.get("init", "scratch")).lower()
        if init == "scratch":
            return SeisPSSBackbone(model_cfg=model_cfg), None
        if init != "pretrained":
            raise ValueError(f"Unsupported transfer init: {init}")

        export_path = transfer_cfg.get("backbone_export_path")
        if not export_path:
            raise ValueError("transfer.backbone_export_path is required when init=pretrained.")
        return load_exported_backbone(
            Path(export_path),
            map_location="cpu",
            strict=bool(transfer_cfg.get("strict", True)),
        )

    def _apply_transfer_freezing(self, *, transfer_cfg: dict[str, Any]) -> None:
        protocol = str(transfer_cfg.get("protocol", "full_finetune")).lower()
        if protocol in {"scratch", "full_finetune"}:
            self._set_backbone_requires_grad(True)
            self.backbone_frozen = False
            return
        if protocol == "frozen":
            self._set_backbone_requires_grad(False)
            self.backbone.eval()
            self.backbone_frozen = True
            return
        if protocol == "partial_finetune":
            self._set_backbone_requires_grad(True)
            self._freeze_module(self.backbone.stem)
            freeze_encoder_layers = int(transfer_cfg.get("freeze_encoder_layers", 0))
            for layer in list(self.backbone.encoder.layers)[:freeze_encoder_layers]:
                self._freeze_module(layer)
            self.backbone_frozen = not any(parameter.requires_grad for parameter in self.backbone.parameters())
            return
        raise ValueError(f"Unsupported transfer protocol: {protocol}")

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = requires_grad
