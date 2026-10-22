"""Waveform tokenization layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem1D(nn.Module):
    """A shallow convolutional tokenizer for 3-component waveforms."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes and strides must have the same length.")

        blocks = []
        current_channels = in_channels
        self.specs: list[tuple[int, int, int]] = []
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
            padding = kernel_size // 2
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.specs.append((kernel_size, stride, padding))
            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)
        self.output_dim = current_channels

    def forward(
        self,
        x: torch.Tensor,
        valid_time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = None
        if valid_time_mask is not None:
            token_mask = valid_time_mask.to(dtype=x.dtype).unsqueeze(1)

        for block, (kernel_size, stride, padding) in zip(self.blocks, self.specs):
            x = block(x)
            if token_mask is not None:
                token_mask = F.max_pool1d(
                    token_mask,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )

        x = x.transpose(1, 2)
        if token_mask is None:
            token_mask = torch.ones(
                x.shape[0],
                x.shape[1],
                device=x.device,
                dtype=torch.bool,
            )
        else:
            token_mask = token_mask.squeeze(1) > 0
        return x, token_mask
