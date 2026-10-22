#!/usr/bin/env python
"""Extract Mamba-2 state-dynamics signals for the paper's Figure 4 (mechanism analysis).

For held-out test traces with manual P/S picks, this script records, from the
exported (EMA) Bi-Mamba backbone:

  * the input-dependent selection signal dt per token: for every BiMamba block
    and scan direction, dt = softplus(in_proj(u)[..., -nheads:] + dt_bias).
    dt is exact regardless of the fused kernel because dt bypasses conv1d;
    it is captured with a forward hook on each Mamba2.in_proj. Backward-scan
    signals are flipped back to physical time order.
  * for a small set of exemplar traces, the full SSD recurrence is replayed in
    fp32 (h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T) to obtain per-token
    hidden-state norms ||h_t||_F and state increments ||h_t - h_{t-1}||_F.
    The replay is verified against mamba_chunk_scan_combined on the same
    captured tensors.

Each trace additionally carries: manual P/S sample indices inside the window,
the label-free pseudo-arrival anchor + confidence (RobustEnergyOnsetEstimator,
pretraining settings), and per-channel SNR metadata.

Usage:
  uv run --no-sync python scripts/analysis/extract_state_dynamics.py \
      --export outputs/checkpoints/seispss/backbone_ema_step090000.pt \
      --dataset STEAD --num-traces 1000 --num-exemplars 12 \
      --out outputs/state_dynamics/stead_woaux90k
  # random-init control (same architecture, untrained weights):
  uv run --no-sync python scripts/analysis/extract_state_dynamics.py \
      --export outputs/checkpoints/seispss/backbone_ema_step090000.pt \
      --random-init --num-traces 1000 --num-exemplars 0 \
      --out outputs/state_dynamics/stead_randominit
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
import torch.nn.functional as F

WINDOW = 6000
SR = 100.0
TOKEN_STRIDE = 8

ESTIMATOR_CFG = dict(  # = configs/data/pretrain.yaml anchor_cache.estimator
    name="robust_energy_onset_v1",
    version=1,
    clip_k=8.0,
    short_windows_sec=[0.2, 0.5],
    long_window_sec=5.0,
    long_window_max_fraction=0.15,
    theta_on=2.75,
    theta_hold=1.25,
    persistence_sec=0.20,
    persistence_fraction=0.60,
    confidence_lo=2.75,
    confidence_hi=6.0,
    confidence_peak_window_sec=2.0,
    eps=1.0e-6,
)


def std_safe_normalize(raw: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean / unit-std, mirroring StdSafeNormalize (std, eps=1e-6)."""
    x = raw - raw.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (x / (std + 1e-6)).astype(np.float32)


class DtCatcher:
    """Forward hooks on every Mamba2.in_proj; recomputes exact per-token dt."""

    def __init__(self, encoder) -> None:
        self.handles = []
        self.raw: dict[tuple[int, str], torch.Tensor] = {}
        self.modules: dict[tuple[int, str], torch.nn.Module] = {}
        for li, layer in enumerate(encoder.layers):
            for direction in ("fwd", "bwd"):
                mamba = getattr(layer, f"mamba_{direction}")
                self.modules[(li, direction)] = mamba

                def hook(module, inputs, output, key=(li, direction)):
                    self.raw[key] = output.detach()

                self.handles.append(mamba.in_proj.register_forward_hook(hook))

    @torch.no_grad()
    def dt(self) -> dict[tuple[int, str], torch.Tensor]:
        """dt[(layer, dir)] -> (B, L, nheads) fp32, physical time order."""
        out = {}
        for key, zxbcdt in self.raw.items():
            mamba = self.modules[key]
            dt = F.softplus(zxbcdt[..., -mamba.nheads:].float() + mamba.dt_bias.float())
            if key[1] == "bwd":
                dt = torch.flip(dt, dims=[1])
            out[key] = dt
        return out

    def zxbcdt(self) -> dict[tuple[int, str], torch.Tensor]:
        return dict(self.raw)

    def clear(self) -> None:
        self.raw.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def write_and_tau(mamba, zxbcdt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token state-input increment norm and effective memory half-life.

    From a captured in_proj output (B, L, d_in_proj) in scan order, computes
      * write:  head-mean ||dt_t x_t B_t^T||_F = dt_t ||x_t|| ||B_t||  (B, L)
        -- the actual state-input increment magnitude of the released Mamba-2
        recurrence h_t = exp(dt_t A) h_{t-1} + dt_t x_t B_t^T, requiring no
        recurrence replay because the increment does not involve h.
      * tau:    head-median half-life ln2 / (dt_t |A|) in seconds  (B, L)
        -- the time for h to decay to half under the local transition.
    """
    d_ssm, d_state, nheads = mamba.d_ssm, mamba.d_state, mamba.nheads
    _, xBC, dt_raw = torch.split(
        zxbcdt.float(), [d_ssm, d_ssm + 2 * d_state, nheads], dim=-1
    )
    xBC = F.conv1d(
        xBC.transpose(1, 2),
        mamba.conv1d.weight.float(),
        bias=mamba.conv1d.bias.float(),
        padding=mamba.d_conv - 1,
        groups=xBC.shape[-1],
    ).transpose(1, 2)[:, : zxbcdt.shape[1]]
    xBC = F.silu(xBC)
    x, B, _ = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)
    dt = F.softplus(dt_raw + mamba.dt_bias.float())  # (B, L, H)
    xh = x.reshape(x.shape[0], x.shape[1], nheads, mamba.headdim)
    write = (dt * xh.norm(dim=-1) * B.norm(dim=-1, keepdim=True)).mean(dim=-1)
    A_abs = torch.exp(mamba.A_log.float())  # |A| per head, A = -exp(A_log)
    tau = (0.6931471805599453 / (dt * A_abs)).median(dim=-1).values * (TOKEN_STRIDE / SR)
    return write, tau


@torch.no_grad()
def replay_scan(mamba, zxbcdt: torch.Tensor, *, verify: bool = True):
    """Replay the Mamba-2 recurrence in fp32 from a captured in_proj output.

    zxbcdt: (1, L, d_in_proj) in scan order (i.e. NOT flipped back for bwd).
    Returns (state_norm, state_delta, write_norm): each (L,) fp32 numpy, scan order.
    """
    assert zxbcdt.shape[0] == 1
    d_ssm, d_state, nheads, headdim = mamba.d_ssm, mamba.d_state, mamba.nheads, mamba.headdim
    z, xBC, dt_raw = torch.split(
        zxbcdt.float(), [d_ssm, d_ssm + 2 * d_state, nheads], dim=-1
    )
    # Causal depthwise conv + SiLU, replicating the non-fused branch semantics.
    xBC = F.conv1d(
        xBC.transpose(1, 2),
        mamba.conv1d.weight.float(),
        bias=mamba.conv1d.bias.float(),
        padding=mamba.d_conv - 1,
        groups=xBC.shape[-1],
    ).transpose(1, 2)[:, : zxbcdt.shape[1]]
    xBC = F.silu(xBC)
    x, B, C = torch.split(xBC, [d_ssm, d_state, d_state], dim=-1)
    dt = F.softplus(dt_raw + mamba.dt_bias.float())  # (1, L, H)
    A = -torch.exp(mamba.A_log.float())  # (H,)

    L = zxbcdt.shape[1]
    xh = x[0].reshape(L, nheads, headdim)
    dA = torch.exp(dt[0] * A)  # (L, H)
    h = torch.zeros(nheads, headdim, d_state, device=zxbcdt.device)
    state_norm = torch.zeros(L, device=zxbcdt.device)
    state_delta = torch.zeros(L, device=zxbcdt.device)
    write_norm = torch.zeros(L, device=zxbcdt.device)
    ys = torch.zeros(L, nheads, headdim, device=zxbcdt.device)
    for t in range(L):
        write = dt[0, t, :, None, None] * torch.einsum("hp,n->hpn", xh[t], B[0, t])
        h_new = dA[t][:, None, None] * h + write
        state_norm[t] = h_new.norm()
        state_delta[t] = (h_new - h).norm()
        write_norm[t] = write.norm()
        ys[t] = torch.einsum("hpn,n->hp", h_new, C[0, t])
        h = h_new

    if verify:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        y_ref = mamba_chunk_scan_combined(
            xh[None], dt, A,
            B[:, :, None, :], C[:, :, None, :],
            chunk_size=mamba.chunk_size, D=None,
            dt_bias=None, dt_softplus=False,
        )[0]
        err = (ys - y_ref).abs().max().item()
        scale = y_ref.abs().max().item()
        if err > 1e-2 * max(scale, 1.0):
            raise RuntimeError(f"scan replay mismatch: max abs err {err:.4g} vs scale {scale:.4g}")

    return (
        state_norm.cpu().numpy(),
        state_delta.cpu().numpy(),
        write_norm.cpu().numpy(),
    )


def parse_stead_snr(md) -> float:
    try:
        arr = np.fromstring(str(md.get("trace_snr_db", "")).strip("[]"), sep=" ")
        return float(arr.mean()) if arr.size else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=str, required=True)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--noise-only", action="store_true",
                        help="use noise traces (no P/S picks) instead of event traces")
    parser.add_argument("--dataset", type=str, default="STEAD")
    parser.add_argument("--num-traces", type=int, default=1000)
    parser.add_argument("--num-exemplars", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    from seispss.data.pseudo_arrival import RobustEnergyOnsetEstimator
    from seispss.data.seisbench import build_dataset, filter_min_trace_length
    from seispss.downstream.phase_picking.windowing import extract_phase_picks
    from seispss.models.backbone import SeisPSSBackbone

    device = torch.device("cuda")
    model, payload = SeisPSSBackbone.from_export(args.export, map_location="cpu")
    if args.random_init:
        torch.manual_seed(args.seed)
        model = SeisPSSBackbone(model_cfg=payload["model_cfg"])
    model = model.to(device).eval()

    dataset = build_dataset(
        dataset_name=args.dataset,
        sampling_rate=int(SR),
        component_order="ZNE",
        dimension_order="NCW",
        cache=None,
    )
    split = dataset.test()
    filter_min_trace_length(split, min_length_samples=WINDOW, target_sampling_rate=int(SR))
    meta = split.metadata
    p_key = "trace_p_arrival_sample" if "trace_p_arrival_sample" in meta else "trace_P_arrival_sample"
    s_key = "trace_s_arrival_sample" if "trace_s_arrival_sample" in meta else "trace_S_arrival_sample"
    if args.noise_only:
        split.filter(meta[p_key].isna() & meta[s_key].isna(), inplace=True)
    else:
        split.filter(meta[p_key].notna() & meta[s_key].notna(), inplace=True)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(split))
    estimator = RobustEnergyOnsetEstimator(**ESTIMATOR_CFG)

    n_layers = len(model.encoder.layers)
    catcher = DtCatcher(model.encoder)

    records: list[dict] = []
    dt_stack: list[np.ndarray] = []  # (n_layers, 2, L_tok) head-mean dt per trace
    wr_stack: list[np.ndarray] = []  # (n_layers, 2, L_tok) head-mean ||dt x B^T||_F
    tau_stack: list[np.ndarray] = []  # (n_layers, 2, L_tok) head-median half-life (s)
    waveforms: list[np.ndarray] = []
    kept = 0
    scanned = 0

    pending_x: list[np.ndarray] = []
    pending_rec: list[dict] = []

    def flush_batch() -> None:
        nonlocal pending_x, pending_rec
        if not pending_x:
            return
        xb = torch.from_numpy(np.stack(pending_x)).to(device)
        catcher.clear()
        with torch.no_grad():
            model(xb, None)
        dts = catcher.dt()
        zx = catcher.zxbcdt()
        bsz = xb.shape[0]
        l_tok = xb.shape[-1] // TOKEN_STRIDE
        arr = np.zeros((bsz, n_layers, 2, l_tok), dtype=np.float16)
        wr_arr = np.zeros((bsz, n_layers, 2, l_tok), dtype=np.float16)
        tau_arr = np.zeros((bsz, n_layers, 2, l_tok), dtype=np.float16)
        for (li, direction), dt in dts.items():
            di = 0 if direction == "fwd" else 1
            arr[:, li, di] = dt.mean(dim=-1).cpu().numpy().astype(np.float16)
            write, tau = write_and_tau(catcher.modules[(li, direction)], zx[(li, direction)])
            if direction == "bwd":  # flip back to physical time order
                write, tau = torch.flip(write, dims=[1]), torch.flip(tau, dims=[1])
            wr_arr[:, li, di] = write.cpu().numpy().astype(np.float16)
            tau_arr[:, li, di] = tau.cpu().numpy().astype(np.float16)
        for bi, rec in enumerate(pending_rec):
            dt_stack.append(arr[bi])
            wr_stack.append(wr_arr[bi])
            tau_stack.append(tau_arr[bi])
            records.append(rec)
        pending_x, pending_rec = [], []

    for idx in order:
        if kept >= args.num_traces:
            break
        scanned += 1
        x, md = split.get_sample(int(idx))
        if x.shape[-1] < WINDOW:
            continue
        raw = np.ascontiguousarray(x[:, :WINDOW]).astype(np.float32)
        if args.noise_only:
            p_in, s_in = [float("nan")], [float("nan")]
        else:
            picks = extract_phase_picks(md, trace_length=x.shape[-1], target_sampling_rate=int(SR))
            p_in = [p for p in picks.get("P", []) if 0 <= p < WINDOW]
            s_in = [s for s in picks.get("S", []) if 0 <= s < WINDOW]
            if not p_in or not s_in:
                continue
        est = estimator.estimate(raw, sampling_rate=SR)
        rec = dict(
            trace_index=int(idx),
            p_sample=float(p_in[0]),
            s_sample=float(s_in[0]),
            anchor_sample=float(est.anchor_sample),
            anchor_confidence=float(est.confidence),
            anchor_valid=bool(est.valid),
            snr_db=parse_stead_snr(md),
        )
        pending_x.append(std_safe_normalize(raw))
        pending_rec.append(rec)
        if args.num_exemplars > 0 and len(waveforms) < args.num_traces:
            waveforms.append(raw)  # keep raw for exemplar selection/plotting
        kept += 1
        if len(pending_x) >= args.batch:
            flush_batch()
        if kept % 200 == 0:
            print(f"processed {kept}/{args.num_traces} (scanned {scanned})", flush=True)
    flush_batch()

    print(f"population extraction done: {kept} traces (scanned {scanned})")

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    pop = {
        "dt": np.stack(dt_stack),  # (K, n_layers, 2, 750) fp16, head-mean
        "wr": np.stack(wr_stack),  # (K, n_layers, 2, 750) fp16, head-mean ||dt x B^T||_F
        "tau": np.stack(tau_stack),  # (K, n_layers, 2, 750) fp16, head-median half-life (s)
        "p_sample": np.array([r["p_sample"] for r in records], dtype=np.float32),
        "s_sample": np.array([r["s_sample"] for r in records], dtype=np.float32),
        "anchor_sample": np.array([r["anchor_sample"] for r in records], dtype=np.float32),
        "anchor_confidence": np.array([r["anchor_confidence"] for r in records], dtype=np.float32),
        "anchor_valid": np.array([r["anchor_valid"] for r in records], dtype=bool),
        "snr_db": np.array([r["snr_db"] for r in records], dtype=np.float32),
        "trace_index": np.array([r["trace_index"] for r in records], dtype=np.int64),
    }
    np.savez_compressed(str(out_base) + "_population.npz", **pop)
    print(f"wrote {out_base}_population.npz")

    # ---------- exemplar deep-dive: full state replay ----------
    if args.num_exemplars > 0:
        snr = pop["snr_db"]
        conf = pop["anchor_confidence"]
        s_minus_p = (pop["s_sample"] - pop["p_sample"]) / SR
        base_ok = (
            np.isfinite(snr)
            & (pop["p_sample"] > 800)  # enough pre-P noise context to show quiescence
            & (pop["p_sample"] < 3500)
            & (s_minus_p > 2.0)
        )
        # High-SNR display exemplars additionally require a confident anchor
        # (mirrors the pretraining gate alpha = alpha_max * confidence; low-conf
        # anchors barely influence sampling, so displaying one would misrepresent
        # the mechanism). Low-SNR exemplars take no confidence gate on purpose:
        # a gated-out cue there demonstrates the designed uniform fallback.
        hi_ok = np.where(base_ok & pop["anchor_valid"] & (conf >= 0.5))[0]
        lo_ok = np.where(base_ok)[0]
        n_lo = args.num_exemplars // 3
        n_hi = args.num_exemplars - n_lo
        chosen = list(hi_ok[np.argsort(snr[hi_ok])][-n_hi:][::-1])
        chosen += [i for i in lo_ok[np.argsort(snr[lo_ok])] if i not in chosen][:n_lo]
        chosen = chosen[: args.num_exemplars]

        exemplars = []
        for rank, ki in enumerate(chosen):
            raw = waveforms[ki]
            xb = torch.from_numpy(std_safe_normalize(raw)[None]).to(device)
            catcher.clear()
            with torch.no_grad():
                model(xb, None)
            dts = catcher.dt()
            zx = catcher.zxbcdt()
            L_tok = xb.shape[-1] // TOKEN_STRIDE
            dt_arr = np.zeros((n_layers, 2, L_tok), dtype=np.float32)
            sn_arr = np.zeros((n_layers, 2, L_tok), dtype=np.float32)
            sd_arr = np.zeros((n_layers, 2, L_tok), dtype=np.float32)
            wr_arr = np.zeros((n_layers, 2, L_tok), dtype=np.float32)
            for (li, direction), dt in dts.items():
                di = 0 if direction == "fwd" else 1
                dt_arr[li, di] = dt[0].mean(dim=-1).cpu().numpy()
                mamba = catcher.modules[(li, direction)]
                s_norm, s_delta, w_norm = replay_scan(
                    mamba, zx[(li, direction)], verify=(rank == 0)
                )
                if direction == "bwd":  # flip back to physical time order
                    s_norm, s_delta, w_norm = s_norm[::-1], s_delta[::-1], w_norm[::-1]
                sn_arr[li, di] = s_norm
                sd_arr[li, di] = s_delta
                wr_arr[li, di] = w_norm
            rec = records[ki]
            exemplars.append(
                dict(
                    waveform=raw.astype(np.float32),
                    dt=dt_arr,
                    state_norm=sn_arr,
                    state_delta=sd_arr,
                    write_norm=wr_arr,
                    **rec,
                )
            )
            print(
                f"exemplar {rank}: trace_index={rec['trace_index']} snr={rec['snr_db']:.1f}dB "
                f"P={rec['p_sample']:.0f} S={rec['s_sample']:.0f} anchor={rec['anchor_sample']:.0f} "
                f"conf={rec['anchor_confidence']:.2f}",
                flush=True,
            )

        ex_out: dict[str, np.ndarray] = {}
        for field in ["waveform", "dt", "state_norm", "state_delta", "write_norm"]:
            ex_out[field] = np.stack([e[field] for e in exemplars])
        for field in ["p_sample", "s_sample", "anchor_sample", "anchor_confidence", "snr_db", "trace_index"]:
            ex_out[field] = np.array([e[field] for e in exemplars])
        np.savez_compressed(str(out_base) + "_exemplars.npz", **ex_out)
        print(f"wrote {out_base}_exemplars.npz")

    catcher.remove()
    manifest = dict(
        export=args.export,
        random_init=args.random_init,
        noise_only=args.noise_only,
        dataset=args.dataset,
        num_traces=kept,
        num_exemplars=args.num_exemplars,
        seed=args.seed,
        estimator=ESTIMATOR_CFG,
        window=WINDOW,
        token_stride=TOKEN_STRIDE,
    )
    Path(str(out_base) + "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()
