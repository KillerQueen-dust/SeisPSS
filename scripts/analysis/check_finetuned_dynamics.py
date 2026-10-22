#!/usr/bin/env python
"""Inspect arrival-aligned Mamba-2 step-size diagnostics after fine-tuning.

Forward-scan signals are standardized per layer against a pre-P baseline and
then averaged over layers. The script reports the P-aligned peak and post-S
window statistics for eligible STEAD test traces.

Usage:
  uv run --no-sync python scripts/analysis/check_finetuned_dynamics.py \
      --ckpt path/to/finetuned.ckpt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import numpy as np
import torch

from extract_state_dynamics import DtCatcher, std_safe_normalize, WINDOW, TOKEN_STRIDE
from _state_dynamics_metrics import selection_z, window_mean, POST_S_WIN, aligned, spike_timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num-traces", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    from seispss.data.seisbench import build_dataset, filter_min_trace_length
    from seispss.downstream.phase_picking.model import PhasePickingModel
    from seispss.downstream.phase_picking.windowing import extract_phase_picks

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]["cfg"]
    model = PhasePickingModel(model_cfg=dict(cfg["model"]), head_cfg=dict(cfg["head"]),
                              transfer_cfg={"init": "scratch", "protocol": "scratch"})
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()

    ds = build_dataset(dataset_name="STEAD", sampling_rate=100, component_order="ZNE",
                       dimension_order="NCW", cache=None)
    split = ds.test()
    filter_min_trace_length(split, min_length_samples=WINDOW, target_sampling_rate=100)
    meta = split.metadata
    split.filter(meta["trace_p_arrival_sample"].notna() & meta["trace_s_arrival_sample"].notna(),
                 inplace=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(split))

    catcher = DtCatcher(model.backbone.encoder)
    n_layers = len(model.backbone.encoder.layers)
    dts, ps, ss = [], [], []
    for idx in order:
        if len(dts) >= args.num_traces:
            break
        x, md = split.get_sample(int(idx))
        if x.shape[-1] < WINDOW:
            continue
        picks = extract_phase_picks(md, trace_length=x.shape[-1], target_sampling_rate=100)
        p_in = [p for p in picks.get("P", []) if 0 <= p < WINDOW]
        s_in = [s for s in picks.get("S", []) if 0 <= s < WINDOW]
        if not p_in or not s_in:
            continue
        xb = torch.from_numpy(
            std_safe_normalize(np.ascontiguousarray(x[:, :WINDOW]).astype(np.float32))[None]
        ).cuda()
        catcher.clear()
        with torch.no_grad():
            model.backbone(xb, None)
        d = catcher.dt()
        arr = np.zeros((n_layers, 2, WINDOW // TOKEN_STRIDE), dtype=np.float16)
        for (li, dr), dt in d.items():
            arr[li, 0 if dr == "fwd" else 1] = dt[0].mean(-1).cpu().numpy()
        dts.append(arr)
        ps.append(p_in[0])
        ss.append(s_in[0])

    pop = {"dt": np.stack(dts), "p_sample": np.array(ps, np.float32),
           "s_sample": np.array(ss, np.float32)}
    z = selection_z(pop)
    p_tok = (pop["p_sample"] / TOKEN_STRIDE).astype(int)
    s_tok = (pop["s_sample"] / TOKEN_STRIDE).astype(int)
    depth, _ = window_mean(z, s_tok, *POST_S_WIN)
    mP = aligned(z, p_tok, 100, 250)
    print(f"finetuned picker backbone: N={len(depth)} "
          f"post-S depth median {np.median(depth):+.2f} sigma, "
          f"frac<-1 {np.mean(depth < -1):.1%}")
    print(f"P spike: {spike_timing(mP, 100)}")


if __name__ == "__main__":
    main()
