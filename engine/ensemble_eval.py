"""Which models to blend? Predict the validation set with every available model, then score
singles, equal-weight blends and cross-validated blend weights with the official metric.

    python ensemble_eval.py --valid-dir /content/data/valid --out /content/preds \
        --model ft_last=/content/drive/MyDrive/wunder/runs/ft_baseline_full/last.pt \
        --model team_best2=/content/drive/MyDrive/wunder/best2.pt \
        --model baseline=/content/drive/MyDrive/wunder/baseline.onnx \
        --model ft_swa=.../epoch_2.pt+.../epoch_3.pt+.../epoch_4.pt+.../epoch_5.pt

A model spec is NAME=PATH where PATH is
  * a checkpoint of our train.py (gru / baseline / mamba; has 'cfg'),
  * the team's checkpoint (train.py of the teammate: StatefulGRU on 112 raw + 8 engineered
    features, keys gru0/gru1/head + target_mean/std),
  * baseline.onnx of the organizers,
  * several checkpoints of ONE run joined with '+': their weights are averaged (SWA).

Blending: every model's predictions are divided by their std on the required rows (so the
models count equally), averaged with weights, and rescaled to std 0.6 before the metric
(the metric clips at +-2). Weights on a 0.1 grid are chosen on one half of the sequences
and scored on the other half (and vice versa) - that cross-validated number is the honest
one; the "fit on all" weights are what an ensemble submission would use.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import SEQ_LEN, WARMUP
from metric import WPAccumulator
from model import BaselineGRU, build_model
from train import predict_sequences

ENG = ["mean_bid", "mean_ask", "spread", "tot_bid_vol", "tot_ask_vol", "ofi", "tfi", "spread_roll_50"]


# ----------------------------------------------------------------------------- the team's model
def team_features(X: np.ndarray) -> np.ndarray:
    """(S, T, 112) -> (S, T, 120): the teammate's add_features() in numpy (pandas rolling sums
    are float64, so are these; the window is 50 with min_periods=1)."""
    X = np.asarray(X, np.float32)
    T = X.shape[1]
    bid_p = X[..., 0:11].mean(-1)
    ask_p = X[..., 11:22].mean(-1)
    spread = ask_p - bid_p
    bid_v = X[..., 22:33].sum(-1)
    ask_v = X[..., 33:44].sum(-1)
    ofi = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)
    signed = np.sign(X[..., 44]) * X[..., 48]

    def roll_sum(a, w=50):
        c = np.cumsum(a.astype(np.float64), axis=-1)
        out = c.copy()
        out[..., w:] = c[..., w:] - c[..., :-w]
        return out

    tfi = roll_sum(signed) / (roll_sum(X[..., 48]) + 1e-8)
    spread_roll = roll_sum(spread) / np.minimum(np.arange(1, T + 1), 50)
    eng = np.stack([bid_p, ask_p, spread, bid_v, ask_v, ofi, tfi, spread_roll], -1).astype(np.float32)
    return np.concatenate([X, eng], -1)


class TeamGRU(nn.Module):
    def __init__(self, in_dim=120, hidden=128):
        super().__init__()
        self.gru0 = nn.GRU(in_dim, hidden, batch_first=True)
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        o, _ = self.gru0(x)
        o, _ = self.gru1(o)
        return self.head(o)


@torch.no_grad()
def predict_team(model, Xv, n, device, batch=16):
    model.eval()
    out = np.zeros((n, SEQ_LEN, 2), np.float32)
    for b in range(0, n, batch):
        x = torch.from_numpy(team_features(Xv[b:b + batch])).to(device)
        out[b:b + batch] = (model(x) * 0.3).float().cpu().numpy()     # pred_scale of their training
    return out


# ----------------------------------------------------------------------------- loading
def load_any(spec: str, device):
    paths = spec.split("+")
    if paths[0].endswith(".onnx"):
        m = BaselineGRU()
        m.load_onnx(paths[0])
        return "ours", m.to(device), "organizers' baseline.onnx"
    sts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]   # our / team files
    st = sts[0]
    if len(sts) > 1:                                    # SWA: average weights of one run's epochs
        avg = {k: sum(s["model"][k].float() for s in sts) / len(sts) for k in st["model"]}
        st = dict(st, model=avg)
    if "cfg" in st:
        m = build_model(st["cfg"])
        m.load_state_dict(st["model"])
        how = f"{st['cfg'].get('arch', 'gru')}, epoch {st.get('epoch')}" + (f", SWA of {len(sts)}" if len(sts) > 1 else "")
        return "ours", m.to(device), how
    if {"gru0.weight_ih_l0", "head.weight"} <= set(st["model"]):
        m = TeamGRU(st["model"]["gru0.weight_ih_l0"].shape[1], st["model"]["gru0.weight_hh_l0"].shape[1])
        m.load_state_dict(st["model"])
        return "team", m.to(device), f"team GRU, in_dim {st['model']['gru0.weight_ih_l0'].shape[1]}"
    raise SystemExit(f"{spec}: unknown checkpoint format")


# ----------------------------------------------------------------------------- scoring
def wp(Y, P):
    acc = WPAccumulator(2)
    acc.update(Y, P)
    return acc.score()


def standardize(P):
    """Centre and scale each target to unit std. Centring matters: the metric clips at +-2,
    so an offset would push predictions into the clip (a constant shift is free otherwise)."""
    sd = P.std(0)
    return np.where(sd > 1e-6, (P - P.mean(0)) / np.maximum(sd, 1e-6), 0.0).astype(np.float32)


def blend(Zs, w):
    B = sum(wi * Z for wi, Z in zip(w, Zs))
    return standardize(B) * 0.6


def weight_grid(k, step=0.1):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k):
        if sum(c) == n:
            yield np.array(c, float) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH (see the docstring)")
    ap.add_argument("--out", default="preds", help="folder for the per-model predictions (.npy)")
    ap.add_argument("--valid-seqs", type=int, default=None)
    ap.add_argument("--report", default=None, help="write the results as JSON here")
    ap.add_argument("--no-cv", action="store_true", help="skip the cross-validated weights (slow with many models)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.valid_dir)
    Xv = np.load(d / "features.npy", mmap_mode="r")
    n = Xv.shape[0] if args.valid_seqs is None else min(args.valid_seqs, Xv.shape[0])
    Y = np.asarray(np.load(d / "targets.npy", mmap_mode="r")[:n])
    M = np.asarray(np.load(d / "is_scored.npy", mmap_mode="r")[:n]) & (np.arange(SEQ_LEN) >= WARMUP)
    need = np.broadcast_to(np.arange(SEQ_LEN) >= WARMUP, M.shape)
    half = np.arange(n) < n // 2
    Ym, Ya = Y[M], Y[need]
    halves = [M & half[:, None], M & ~half[:, None]]

    names, Zm, Za, Zh = [], [], [], []
    for spec in args.model:
        name, path = spec.split("=", 1)
        f = out / f"{name}.npy"
        if f.exists():
            P = np.load(f)
            print(f"{name:14s} predictions loaded from {f}")
        else:
            kind, model, how = load_any(path, device)
            P = predict_team(model, Xv, n, device) if kind == "team" else predict_sequences(model, Xv, np.arange(n), device)
            np.save(f, P)
            print(f"{name:14s} predicted ({how})")
        if (P[need].std(0) < 1e-6).any():
            print(f"   !! {name}: (almost) constant predictions on this data - the model does not work here")
        names.append(name)
        Zm.append(standardize(P[M]))
        Za.append(standardize(P[need]))
        Zh.append([standardize(P[h]) for h in halves])

    res = {"valid_seqs": int(n), "scored_rows": int(M.sum()), "singles": {}, "corr": {}, "equal_blends": {}}
    print(f"\nvalidation: {n} sequences, {int(M.sum()):,} scored rows. WP on the mask (and on all required rows):")
    for nm, zm, za in zip(names, Zm, Za):
        s = (wp(Ym, zm * 0.6), wp(Ya, za * 0.6))
        res["singles"][nm] = s
        print(f"  {nm:14s} {s[0]:.5f}   ({s[1]:.5f})")
    C = np.array([[np.mean([np.corrcoef(a[:, t], b[:, t])[0, 1] for t in range(2)]) for b in Zm] for a in Zm])
    res["corr"] = {a: {b: round(float(C[i, j]), 3) for j, b in enumerate(names)} for i, a in enumerate(names)}
    print("\ncorrelation of predictions on scored rows (lower = more to gain from blending):")
    print("  " + " " * 14 + "".join(f"{nm[:10]:>11s}" for nm in names))
    for i, nm in enumerate(names):
        print(f"  {nm:14s}" + "".join(f"{C[i, j]:11.3f}" for j in range(len(names))))

    gains = {}
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(range(len(names)), k):
            s = wp(Ym, blend([Zm[i] for i in combo], np.ones(k) / k))
            key = "+".join(names[i] for i in combo)
            res["equal_blends"][key] = s
            gains[key] = s - max(res["singles"][names[i]][0] for i in combo)
    top = sorted(res["equal_blends"].items(), key=lambda kv: -kv[1])[:12]
    print("\nbest equal-weight blends (gain vs the best single model inside the blend):")
    for key, s in top:
        print(f"  {key:60s} {s:.5f}  ({gains[key]:+.5f})")

    if args.report:                                    # write now; the CV part below adds to it
        Path(args.report).write_text(json.dumps(res, indent=1, default=float))
    if len(names) <= 7 and not args.no_cv:
        step = 0.1 if len(names) <= 5 else 0.2
        print(f"\nweights (grid step {step}) chosen on one half of the sequences, scored on the other (honest):")
        cv, fit = [], None
        grid = list(weight_grid(len(names), step))
        Yh = [Y[h] for h in halves]
        for a, b in ((0, 1), (1, 0)):
            best_w = max(grid, key=lambda w: wp(Yh[a], blend([z[a] for z in Zh], w)))
            cv.append(wp(Yh[b], blend([z[b] for z in Zh], best_w)))
        fit = max(grid, key=lambda w: wp(Ym, blend(Zm, w)))
        singles_cv = {nm: np.mean([wp(Yh[b], Zh[i][b] * 0.6) for b in (0, 1)]) for i, nm in enumerate(names)}
        res["cv_blend"] = float(np.mean(cv))
        res["cv_best_single"] = max(singles_cv.items(), key=lambda kv: kv[1])
        res["fit_weights"] = {nm: float(w) for nm, w in zip(names, fit)}
        print(f"  cross-validated blend {np.mean(cv):.5f} vs best single on the same halves "
              f"{res['cv_best_single'][1]:.5f} ({res['cv_best_single'][0]})")
        print("  weights fitted on all sequences: " + ", ".join(f"{nm} {w:.1f}" for nm, w in zip(names, fit)))
    if args.report:
        Path(args.report).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
