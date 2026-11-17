"""Linear probe: ridge regression on the current row (+ 1-step deltas + book features).

Purpose: a floor for "how much signal is in a single row" and an end-to-end
check of data -> features -> metric. If the probe is far below the GRU baseline (0.617),
history matters a lot; if it is close, feature engineering is the main lever.

Usage:
    python linear_probe.py --train data/train --valid data/valid --train-seqs 300 --valid-seqs 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import SEQ_LEN, WARMUP
from eda import book_features
from metric import WEIGHT_MODES, WPAccumulator


def build(X: np.ndarray, use_delta: bool, use_book: bool) -> np.ndarray:
    """X: (S, T, 112) float32 -> (S, T, F)."""
    parts = [X]
    if use_delta:
        parts.append(np.diff(X, axis=1, prepend=X[:, :1]))
    if use_book:
        bf = book_features(X)
        parts.append(np.stack([v for k, v in bf.items() if not k.endswith("_mid")], -1))
    F = np.concatenate(parts, -1).astype(np.float32)
    return np.nan_to_num(F)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--train-seqs", type=int, default=300)
    ap.add_argument("--valid-seqs", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5, help="row thinning for fitting")
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge strength (on standardized X)")
    ap.add_argument("--no-delta", action="store_true")
    ap.add_argument("--no-book", action="store_true")
    ap.add_argument("--batch", type=int, default=50, help="sequences per chunk")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    use_delta, use_book = not args.no_delta, not args.no_book

    Xt = np.load(Path(args.train) / "features.npy", mmap_mode="r")
    Yt = np.load(Path(args.train) / "targets.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    tr = np.sort(rng.choice(Xt.shape[0], min(args.train_seqs, Xt.shape[0]), replace=False))
    rows = np.arange(WARMUP, SEQ_LEN, args.stride)

    # pass 1: accumulate X'X, X'y with running mean/std (two passes over chunks)
    n = 0
    s1 = s2 = None
    for b in range(0, len(tr), args.batch):
        idx = tr[b:b + args.batch]
        F = build(np.asarray(Xt[idx], np.float32), use_delta, use_book)[:, rows]
        F = F.reshape(-1, F.shape[-1]).astype(np.float64)
        s1 = F.sum(0) if s1 is None else s1 + F.sum(0)
        s2 = (F * F).sum(0) if s2 is None else s2 + (F * F).sum(0)
        n += len(F)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 1e-12))
    dim = len(mu) + 1
    XtX = np.zeros((dim, dim))
    Xty = np.zeros((dim, 2))
    for b in range(0, len(tr), args.batch):
        idx = tr[b:b + args.batch]
        F = build(np.asarray(Xt[idx], np.float32), use_delta, use_book)[:, rows]
        F = ((F.reshape(-1, F.shape[-1]) - mu) / sd).clip(-10, 10)
        F = np.hstack([F, np.ones((len(F), 1))])
        y = np.clip(np.asarray(Yt[idx][:, rows], np.float64).reshape(-1, 2), -2, 2)
        XtX += F.T @ F
        Xty += F.T @ y
    reg = args.alpha * n / 1e4 * np.eye(dim)
    reg[-1, -1] = 0
    W = np.linalg.solve(XtX + reg, Xty)
    print(f"fit on {n:,} rows, {dim - 1} features")

    Xv = np.load(Path(args.valid) / "features.npy", mmap_mode="r")
    Yv = np.load(Path(args.valid) / "targets.npy", mmap_mode="r")
    mpath = Path(args.valid) / "is_scored.npy"
    Mv = np.load(mpath, mmap_mode="r") if mpath.exists() else None
    nv = Xv.shape[0] if args.valid_seqs is None else min(args.valid_seqs, Xv.shape[0])
    acc_m = {m: WPAccumulator(2, m) for m in WEIGHT_MODES}
    acc_a = {m: WPAccumulator(2, m) for m in WEIGHT_MODES}
    need = np.arange(SEQ_LEN) >= WARMUP
    for b in range(0, nv, args.batch):
        idx = np.arange(b, min(b + args.batch, nv))
        F = build(np.asarray(Xv[idx], np.float32), use_delta, use_book)
        S_, T_, D_ = F.shape
        F = ((F.reshape(-1, D_) - mu) / sd).clip(-10, 10)
        p = (np.hstack([F, np.ones((len(F), 1))]) @ W).reshape(S_, T_, 2)
        y = np.asarray(Yv[idx])
        for m in WEIGHT_MODES:
            acc_a[m].update(y[:, need], p[:, need])
            if Mv is not None:
                acc_m[m].update(y, p, np.asarray(Mv[idx]) & need)
    res = {"train_seqs": len(tr), "valid_seqs": nv, "features": dim - 1,
           "delta": use_delta, "book": use_book, "alpha": args.alpha}
    for m in WEIGHT_MODES:
        res[f"WP_all_{m}"] = round(acc_a[m].score(), 5)
        if Mv is not None:
            res[f"WP_mask_{m}"] = round(acc_m[m].score(), 5)
            res[f"WP_mask_{m}_per_target"] = [round(v, 5) for v in acc_m[m].per_target()]
    print(json.dumps(res, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
