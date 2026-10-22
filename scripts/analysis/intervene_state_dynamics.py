#!/usr/bin/env python
"""Inference-time dt intervention for the paper's Figure 4 (mechanism -> function).

Tests whether the arrival-locked dt dynamics of the fine-tuned SeisPSS picker
are functionally load-bearing for phase picking, without any retraining.

For every BiMamba block and scan direction, a forward hook on Mamba2.in_proj
replaces the raw dt channels (the last nheads dims, which bypass conv1d) inside
a chosen token window with that trace's per-head median raw dt over its pre-P
baseline region [5, p_tok-25). Because dt = softplus(raw + dt_bias) inside the
fused kernel, this exactly substitutes "noise-typical" step sizes while leaving
the x/B/C pathways untouched -- an intervention on the dt pathway only.

Conditions (token half-width 6 = 0.48 s at stride 8 / 100 Hz):
  baseline         no intervention
  pulse_p          clamp P +- 0.48 s                  (removes the P refresh pulse)
  pulse_s          clamp S +- 0.48 s
  pulse_ps         clamp both arrival pulses
  pulse_ps_fwd     same, forward-scan modules only
  plateau          clamp [S+0.5 s, S+10 s]            (removes the retention regime)
  control_pulse    two 0.96 s windows in pre-P noise  (placement specificity control)
  control_plateau  9.5 s pre-P window ending 2 s before P (subset with P >= 12.4 s)
  post_all         clamp [P-0.5 s, end]               (aggressive upper bound)

Metrics: the paper protocol exactly -- peak decoding thresholds P=0.4/S=0.5,
min peak distance 50 samples, tolerance 20 samples (0.2 s), STEAD test split,
60 s windows, bf16 autocast. Also logged: mean max softmax probability within
the tolerance window around the true pick ("peak prob", a dose measure that
does not depend on the decoding threshold).

Usage:
  uv run --no-sync python scripts/analysis/intervene_state_dynamics.py \
      --ckpt path/to/finetuned.ckpt \
      --num-traces 3000 --out outputs/state_dynamics/intervention_stead1pct.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

WINDOW = 6000
SR = 100.0
STRIDE = 8
L_TOK = WINDOW // STRIDE
TRIM = 5          # stem edge tokens excluded from baseline
GAP = 25          # baseline ends this many tokens before P (2 s)
HW = 6            # pulse half-width in tokens (0.48 s)
PLATEAU_TOK = 125  # 10 s
MIN_P_TOK = 55    # need a usable pre-P baseline


def std_safe_normalize(raw: np.ndarray) -> np.ndarray:
    x = raw - raw.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x / (std + 1e-6)).astype(np.float32)


class DtIntervention:
    """Forward hooks that rewrite raw dt channels inside a token window."""

    def __init__(self, encoder) -> None:
        self.enabled = False
        self.win: torch.Tensor | None = None   # (B, L_tok) bool, physical order
        self.base: torch.Tensor | None = None  # (B, L_tok) bool, physical order
        self.directions = {"fwd", "bwd"}
        self.handles = []
        for layer in encoder.layers:
            for direction in ("fwd", "bwd"):
                mamba = getattr(layer, f"mamba_{direction}")
                self.handles.append(
                    mamba.in_proj.register_forward_hook(self._make_hook(mamba, direction))
                )

    def _make_hook(self, mamba, direction: str):
        def hook(module, inputs, output):
            if not self.enabled or direction not in self.directions:
                return None
            win, base = self.win, self.base
            if direction == "bwd":  # bwd module sees the flipped sequence
                win = torch.flip(win, dims=[1])
                base = torch.flip(base, dims=[1])
            nheads = mamba.nheads
            dt_raw = output[..., -nheads:]
            vals = dt_raw.float().masked_fill(~base.unsqueeze(-1), float("nan"))
            med = vals.nanmedian(dim=1, keepdim=True).values.to(dt_raw.dtype)
            new_dt = torch.where(win.unsqueeze(-1), med.expand_as(dt_raw), dt_raw)
            out = output.clone()
            out[..., -nheads:] = new_dt
            return out

        return hook

    def set(self, *, win: torch.Tensor | None, base: torch.Tensor | None,
            directions: set[str] = frozenset({"fwd", "bwd"})) -> None:
        self.enabled = win is not None
        self.win, self.base = win, base
        self.directions = set(directions)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def token_windows(condition: str, p_tok: np.ndarray, s_tok: np.ndarray) -> np.ndarray | None:
    """(B, L_tok) bool window mask in physical token order; None = no intervention."""
    B = len(p_tok)
    t = np.arange(L_TOK)[None, :]
    p = p_tok[:, None]
    s = s_tok[:, None]
    if condition == "baseline":
        return None
    if condition == "pulse_p":
        return np.abs(t - p) <= HW
    if condition == "pulse_s":
        return np.abs(t - s) <= HW
    if condition in {"pulse_ps", "pulse_ps_fwd"}:
        return (np.abs(t - p) <= HW) | (np.abs(t - s) <= HW)
    if condition == "plateau":
        return (t >= s + HW) & (t <= np.minimum(s + PLATEAU_TOK, L_TOK - TRIM))
    if condition == "control_pulse":
        c1 = np.round(0.35 * p).astype(int)
        c2 = np.round(0.70 * p).astype(int)
        return (np.abs(t - c1) <= HW) | (np.abs(t - c2) <= HW)
    if condition == "control_plateau":
        lo = p - GAP - (PLATEAU_TOK - HW)  # same length as plateau, ends 2 s before P
        return (t >= lo) & (t < p - GAP)
    if condition == "post_all":
        return (t >= p - HW) & (t <= L_TOK - TRIM)
    raise ValueError(condition)


CONDITIONS = [
    "baseline", "pulse_p", "pulse_s", "pulse_ps", "pulse_ps_fwd",
    "plateau", "control_pulse", "control_plateau", "post_all",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-traces", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=str,
                        default="outputs/state_dynamics/intervention_stead1pct.json")
    args = parser.parse_args()

    from seispss.data.seisbench import build_dataset, filter_min_trace_length
    from seispss.downstream.phase_picking.metrics import PhasePickingMetrics
    from seispss.downstream.phase_picking.model import PhasePickingModel
    from seispss.downstream.phase_picking.windowing import extract_phase_picks

    device = torch.device("cuda")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    model = PhasePickingModel(
        model_cfg=dict(cfg["model"]),
        head_cfg=dict(cfg["head"]),
        transfer_cfg={"init": "scratch", "protocol": "scratch"},
    )
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if "cpu_fallback" not in m]
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()

    eval_cfg = cfg["eval"]
    metric_kwargs = dict(
        threshold_p=float(eval_cfg["threshold_p"]),
        threshold_s=float(eval_cfg["threshold_s"]),
        tolerance_samples=int(eval_cfg["tolerance_samples"]),
        min_peak_distance_samples=int(eval_cfg["min_peak_distance_samples"]),
        sampling_rate=SR,
    )
    tol = int(eval_cfg["tolerance_samples"])

    # ---------- data: STEAD test traces with P & S inside the 60 s window ----------
    dataset = build_dataset(
        dataset_name="STEAD", sampling_rate=int(SR), component_order="ZNE",
        dimension_order="NCW", cache=None,
    )
    split = dataset.test()
    filter_min_trace_length(split, min_length_samples=WINDOW, target_sampling_rate=int(SR))
    meta = split.metadata
    p_key = "trace_p_arrival_sample" if "trace_p_arrival_sample" in meta else "trace_P_arrival_sample"
    s_key = "trace_s_arrival_sample" if "trace_s_arrival_sample" in meta else "trace_S_arrival_sample"
    split.filter(meta[p_key].notna() & meta[s_key].notna(), inplace=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(split))
    xs, p_smp, s_smp = [], [], []
    for idx in order:
        if len(xs) >= args.num_traces:
            break
        x, md = split.get_sample(int(idx))
        if x.shape[-1] < WINDOW:
            continue
        picks = extract_phase_picks(md, trace_length=x.shape[-1], target_sampling_rate=int(SR))
        p_in = [p for p in picks.get("P", []) if 0 <= p < WINDOW]
        s_in = [sv for sv in picks.get("S", []) if 0 <= sv < WINDOW]
        if not p_in or not s_in:
            continue
        p_tok = int(p_in[0]) // STRIDE
        if p_tok < MIN_P_TOK:
            continue
        xs.append(std_safe_normalize(np.ascontiguousarray(x[:, :WINDOW]).astype(np.float32)))
        p_smp.append(int(p_in[0]))
        s_smp.append(int(s_in[0]))
    X = torch.from_numpy(np.stack(xs))
    p_smp = np.asarray(p_smp)
    s_smp = np.asarray(s_smp)
    p_tok = p_smp // STRIDE
    s_tok = s_smp // STRIDE
    n = len(xs)
    print(f"eval set: {n} traces")

    # control_plateau is only geometrically possible with a long pre-P segment
    ctl_sub = p_tok - GAP - (PLATEAU_TOK - HW) >= TRIM
    print(f"control_plateau feasible subset: {int(ctl_sub.sum())} traces")

    base_np = (np.arange(L_TOK)[None, :] >= TRIM) & (np.arange(L_TOK)[None, :] < (p_tok[:, None] - GAP))

    hooks = DtIntervention(model.backbone.encoder)
    results: dict[str, dict] = {}

    for condition in CONDITIONS:
        win_np = token_windows(condition, p_tok, s_tok)
        directions = {"fwd"} if condition.endswith("_fwd") else {"fwd", "bwd"}
        keep = ctl_sub if condition == "control_plateau" else np.ones(n, dtype=bool)

        metrics_full = PhasePickingMetrics(**metric_kwargs)
        metrics_sub = PhasePickingMetrics(**metric_kwargs)
        peak_p, peak_s = [], []

        for lo in range(0, n, args.batch):
            hi = min(lo + args.batch, n)
            sel = np.arange(lo, hi)
            sel = sel[keep[sel]]
            if sel.size == 0:
                continue
            xb = X[sel].to(device)
            if win_np is not None:
                hooks.set(
                    win=torch.from_numpy(win_np[sel]).to(device),
                    base=torch.from_numpy(base_np[sel]).to(device),
                    directions=directions,
                )
            else:
                hooks.set(win=None, base=None)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
            logits = logits.float()
            batch = {
                "p_pick_sample": torch.from_numpy(p_smp[sel]),
                "s_pick_sample": torch.from_numpy(s_smp[sel]),
            }
            metrics_full.update(logits=logits, batch=batch)
            if condition != "control_plateau":
                sub_sel = ctl_sub[sel]
                if sub_sel.any():
                    metrics_sub.update(
                        logits=logits[sub_sel],
                        batch={k: v[sub_sel] for k, v in batch.items()},
                    )
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for bi, gi in enumerate(sel):
                pw = probs[bi, 0, max(p_smp[gi] - tol, 0): p_smp[gi] + tol + 1]
                sw = probs[bi, 1, max(s_smp[gi] - tol, 0): s_smp[gi] + tol + 1]
                peak_p.append(float(pw.max()))
                peak_s.append(float(sw.max()))

        row = {
            "n": int(keep.sum()),
            "metrics": metrics_full.compute(),
            "peak_prob_p_mean": float(np.mean(peak_p)),
            "peak_prob_s_mean": float(np.mean(peak_s)),
        }
        if condition == "control_plateau":
            row["metrics_subset"] = row["metrics"]  # already restricted
        else:
            row["metrics_subset"] = metrics_sub.compute()
        results[condition] = row
        m = row["metrics"]
        print(
            f"{condition:16s} n={row['n']:5d}  "
            f"P f1={m['p/f1']:.4f} rec={m['p/recall']:.4f}  "
            f"S f1={m['s/f1']:.4f} rec={m['s/recall']:.4f}  "
            f"peakP={row['peak_prob_p_mean']:.4f} peakS={row['peak_prob_s_mean']:.4f}",
            flush=True,
        )

    hooks.remove()
    out = dict(
        ckpt=args.ckpt,
        num_traces=n,
        control_plateau_subset=int(ctl_sub.sum()),
        seed=args.seed,
        window=WINDOW,
        half_width_tokens=HW,
        plateau_tokens=PLATEAU_TOK,
        eval_protocol=metric_kwargs,
        conditions=results,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
