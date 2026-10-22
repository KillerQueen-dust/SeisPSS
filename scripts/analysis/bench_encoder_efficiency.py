#!/usr/bin/env python
"""Encoder efficiency benchmark: Bi-Mamba vs parameter-matched Transformer.

Two jobs:
  1. Size the ablation pretraining budget (how much slower is the Transformer JEPA step?).
  2. Produce the raw numbers for the long-sequence efficiency figure (Q4): peak memory and
     throughput as the window grows from the 60 s training regime to minute-scale segments.

Usage:
  uv run --no-sync python scripts/analysis/bench_encoder_efficiency.py --mode step
  uv run --no-sync python scripts/analysis/bench_encoder_efficiency.py --mode sweep --out outputs/bench/encoder_efficiency.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from seispss.models.encoders import build_encoder
from seispss.models.jepa import SeisPSS

BASE_MODEL_CFG = {
    "in_channels": 3,
    "stem_mode": "conv",
    "stem_channels": [64, 128, 256],
    "stem_kernel_sizes": [7, 7, 5],
    "stem_strides": [2, 2, 2],
    "encoder_type": "bimamba",
    "encoder_depth": 8,
    "encoder_mlp_ratio": 4.0,
    "encoder_num_heads": 8,
    "mamba_d_state": 64,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "predictor_dim": 192,
    "predictor_depth": 4,
    "predictor_num_heads": 6,
    "predictor_mlp_ratio": 4.0,
    "dropout": 0.05,
}

TRANSFORMER_MODEL_CFG = {
    **BASE_MODEL_CFG,
    "encoder_type": "transformer",
    "encoder_depth": 16,
}

MASK_CFG = {
    "type": "time_span",
    "sample_on_cpu": True,
    "num_targets": 3,
    "span_length_tokens": 64,
    "min_gap_tokens": 8,
    "context_gap_tokens": 16,
}

VARIANTS = {"bimamba": BASE_MODEL_CFG, "transformer": TRANSFORMER_MODEL_CFG}


def count_params(module: torch.nn.Module, *, skip_cpu_fallback: bool = True) -> int:
    total = 0
    for name, parameter in module.named_parameters():
        if skip_cpu_fallback and "cpu_fallback" in name:
            continue
        total += parameter.numel()
    return total


def timeit(fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_step(args: argparse.Namespace) -> None:
    """Full JEPA training step (stem + masked online encoder + no-grad teacher + predictor)."""
    device = torch.device("cuda")
    for name, model_cfg in VARIANTS.items():
        model = SeisPSS(model_cfg=model_cfg, mask_cfg=MASK_CFG).to(device)
        encoder_params = count_params(model.context_encoder)
        total_params = count_params(model.online_stem) + encoder_params + count_params(model.predictor)
        x = torch.randn(args.batch_size, 3, args.samples, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def one_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x=x)
                loss = torch.nn.functional.smooth_l1_loss(out.predictions.float(), out.targets.float())
            loss.backward()
            optimizer.step()

        torch.cuda.reset_peak_memory_stats()
        seconds = timeit(one_step, warmup=args.warmup, iters=args.iters)
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        samples_per_s = args.batch_size / seconds
        print(
            f"{name:>12s}  encoder={encoder_params/1e6:6.2f}M  online-total={total_params/1e6:6.2f}M  "
            f"step={seconds*1e3:8.1f} ms  {samples_per_s:7.1f} samples/s  peak={peak_gb:5.1f} GB "
            f"(bs={args.batch_size}, {args.samples} samples)"
        )
        del model, optimizer, x
        torch.cuda.empty_cache()


def bench_sweep(args: argparse.Namespace) -> None:
    """Encoder-only forward (inference) and forward+backward across sequence lengths."""
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    stride = 8  # conv stem downsampling
    for seconds_window in args.window_seconds:
        length = int(seconds_window * 100) // stride
        for name, model_cfg in VARIANTS.items():
            encoder = build_encoder(model_cfg=model_cfg, dim=256).to(device)
            row: dict[str, object] = {
                "encoder": name,
                "window_seconds": seconds_window,
                "tokens": length,
                "batch_size": args.batch_size,
                "encoder_params_m": count_params(encoder) / 1e6,
            }
            x = torch.randn(args.batch_size, length, 256, device=device)
            try:
                encoder.eval()
                torch.cuda.reset_peak_memory_stats()

                def infer() -> None:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        encoder(x)

                seconds = timeit(infer, warmup=args.warmup, iters=args.iters)
                row["infer_ms"] = seconds * 1e3
                row["infer_samples_per_s"] = args.batch_size / seconds
                row["infer_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3

                encoder.train()
                torch.cuda.reset_peak_memory_stats()

                def train_step() -> None:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = encoder(x)
                    out.float().pow(2).mean().backward()
                    encoder.zero_grad(set_to_none=True)

                seconds = timeit(train_step, warmup=args.warmup, iters=args.iters)
                row["train_ms"] = seconds * 1e3
                row["train_samples_per_s"] = args.batch_size / seconds
                row["train_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                row["status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                row["status"] = "oom"
            print(row)
            rows.append(row)
            del encoder, x
            torch.cuda.empty_cache()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row})
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["step", "sweep"], default="step")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, nargs="+", default=[60, 120, 240, 480, 960])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if args.mode == "step":
        bench_step(args)
    else:
        bench_sweep(args)


if __name__ == "__main__":
    main()
