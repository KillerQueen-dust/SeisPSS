"""Numerical transforms shared by state-dynamics diagnostics."""

from __future__ import annotations

import numpy as np

STRIDE = 8
TOK_S = STRIDE / 100.0
TRIM, GAP, MINB = 5, 25, 30
POST_S_WIN = (25, 125)


def selection_z(pop: dict, key: str = "dt", p_tok: np.ndarray | None = None) -> np.ndarray:
    """Forward-scan signal, per-layer z vs the pre-P baseline, layer-averaged."""
    sig = pop[key][:, :, 0, :].astype(np.float32)
    if p_tok is None:
        p_tok = (pop["p_sample"] / STRIDE).astype(int)
    K, _, L = sig.shape
    z = np.full((K, L), np.nan, np.float32)
    for k in range(K):
        end = p_tok[k] - GAP
        if end < MINB:
            continue
        base = sig[k, :, TRIM:end]
        mu, sd = base.mean(1, keepdims=True), base.std(1, keepdims=True)
        if (sd < 1e-8).any():
            continue
        z[k] = ((sig[k] - mu) / sd).mean(0)
        z[k, :TRIM] = np.nan
        z[k, L - TRIM:] = np.nan
    return z


def aligned(z: np.ndarray, c: np.ndarray, pre: int, post: int) -> np.ndarray:
    K, L = z.shape
    g = np.full((K, pre + post), np.nan, np.float32)
    for k in range(K):
        if not np.isfinite(z[k]).any():
            continue
        lo, hi = int(c[k]) - pre, int(c[k]) + post
        sl, sh = max(lo, 0), min(hi, L)
        g[k, sl - lo: sh - lo] = z[k, sl:sh]
    return np.nanmean(g, 0)


def window_mean(z: np.ndarray, start_tok: np.ndarray, lo: int, hi: int,
                stop_tok: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-trace mean z over [start+lo, min(start+hi, stop)); returns (values, kept idx)."""
    out, idx = [], []
    for k in range(z.shape[0]):
        if not np.isfinite(z[k]).any():
            continue
        a = start_tok[k] + lo
        b = min(start_tok[k] + hi, z.shape[1])
        if stop_tok is not None:
            b = min(b, stop_tok[k])
        if b - a < 12:  # require >= ~1 s of usable window
            continue
        v = z[k, a:b]
        if not np.isfinite(v).any():
            continue
        out.append(np.nanmean(v))
        idx.append(k)
    return np.asarray(out), np.asarray(idx)


def spike_timing(m: np.ndarray, pre: int) -> dict:
    win = m[pre - 6: pre + 13]  # -0.5 s .. +1 s around the pick
    peak_idx = int(np.nanargmax(win)) - 6
    return {"peak_sigma": float(np.nanmax(win)), "peak_lag_s": peak_idx * TOK_S}
