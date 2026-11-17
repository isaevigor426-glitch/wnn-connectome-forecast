"""Day-1 EDA on a random subset of sequences (works on prepare_data.py output).

Answers the questions that shape the model:
  1. Feature scale: are prices raw, relative to mid, normalized? Any NaN/constant columns?
     Is float16 storage safe?
  2. Book sanity: is max(bid p0..p10) < min(ask p11..p21)? (validates the bid/ask split)
  3. Targets: distribution, tails beyond +-2 (clipped by the metric), corr(t0, t1),
     autocorrelation by lag (≈ prediction horizon), dependence on step_in_seq.
  4. Signal: per-feature Pearson with t0/t1 for levels and 1-step deltas, plus a few
     hand-made book features (spread, mid change, imbalance).
  5. Validation mask: how many rows are scored and where in the sequence.

Usage:
    python eda.py --data data/train --n-seqs 100 --out reports/eda_train
    python eda.py --data data/valid --n-seqs 100 --out reports/eda_valid
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURES, IDX, SEQ_LEN, WARMUP, CLIP


def corr_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson of each column of a (N,F) with vector b (N,). NaN-safe-ish (constant -> 0)."""
    a = a - a.mean(0)
    b = b - b.mean()
    den = np.sqrt((a * a).sum(0) * (b * b).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a * b[:, None]).sum(0) / den
    return np.nan_to_num(r)


def autocorr_within(y: np.ndarray, lag: int) -> float:
    """y: (S, T). Pearson between y[:, t] and y[:, t+lag] pooled over sequences."""
    a, b = y[:, :-lag].ravel(), y[:, lag:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def book_features(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (S, T, 112) -> dict of (S, T) arrays of simple order-book features."""
    out = {}
    for inst in ("i0", "i1"):
        bp, ap_ = x[..., IDX[f"{inst}_bid_p"]], x[..., IDX[f"{inst}_ask_p"]]
        bv, av = x[..., IDX[f"{inst}_bid_v"]], x[..., IDX[f"{inst}_ask_v"]]
        # Indices are NOT depth levels -> locate best quotes by price.
        bi = bp.argmax(-1)
        ai = ap_.argmin(-1)
        best_bid = np.take_along_axis(bp, bi[..., None], -1)[..., 0]
        best_ask = np.take_along_axis(ap_, ai[..., None], -1)[..., 0]
        bbv = np.take_along_axis(bv, bi[..., None], -1)[..., 0]
        bav = np.take_along_axis(av, ai[..., None], -1)[..., 0]
        mid = (best_bid + best_ask) / 2
        out[f"{inst}_spread"] = best_ask - best_bid
        out[f"{inst}_mid"] = mid
        out[f"{inst}_dmid"] = np.diff(mid, axis=1, prepend=mid[:, :1])
        out[f"{inst}_micro_minus_mid"] = (best_bid * bav + best_ask * bbv) / (bbv + bav + 1e-12) - mid
        out[f"{inst}_imb_top"] = (bbv - bav) / (bbv + bav + 1e-12)
        out[f"{inst}_imb_all"] = (bv.sum(-1) - av.sum(-1)) / (bv.sum(-1) + av.sum(-1) + 1e-12)
        out[f"{inst}_crossed"] = (best_bid >= best_ask).astype(np.float32)
    out["dmid_diff_i0_i1"] = out["i0_dmid"] - out["i1_dmid"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder from prepare_data.py")
    ap.add_argument("--n-seqs", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Xm = np.load(d / "features.npy", mmap_mode="r")
    Ym = np.load(d / "targets.npy", mmap_mode="r")
    has_mask = (d / "is_scored.npy").exists()
    n_all = Xm.shape[0]
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(n_all, size=min(args.n_seqs, n_all), replace=False))
    X = np.asarray(Xm[sel], np.float32)       # (S, T, 112)
    Y = np.asarray(Ym[sel], np.float32)       # (S, T, 2)
    S = len(sel)
    report: dict = {"n_seqs_total": int(n_all), "n_seqs_used": int(S)}
    print(f"loaded {S} seqs: X {X.shape} ({X.nbytes / 1e9:.1f} GB)")

    # ---------- 1. feature stats ----------
    flat = X.reshape(-1, X.shape[-1])
    sub = flat[::7]  # row subsample for the heavy order statistics (RAM)
    q = np.nanpercentile(sub, [0, 1, 50, 99, 100], axis=0)
    dx = np.diff(X, axis=1)
    lag1 = np.array([np.corrcoef(X[:, :-1, j].ravel(), X[:, 1:, j].ravel())[0, 1]
                     if flat[:, j].std() > 0 else np.nan for j in range(X.shape[-1])])
    fstats = pd.DataFrame({
        "feature": FEATURES,
        "mean": np.nanmean(flat, 0), "std": np.nanstd(flat, 0),
        "min": q[0], "p01": q[1], "p50": q[2], "p99": q[3], "max": q[4],
        "frac_nan": np.isnan(flat).mean(0), "frac_zero": (flat == 0).mean(0),
        "frac_changed_vs_prev": (dx != 0).mean((0, 1)),
        "lag1_autocorr": lag1,
        "n_unique_approx": [len(np.unique(flat[::97, j])) for j in range(flat.shape[1])],
    })
    fstats.to_csv(out / "feature_stats.csv", index=False)
    report["constant_features"] = fstats.loc[fstats["std"] == 0, "feature"].tolist()
    report["features_with_nan"] = fstats.loc[fstats["frac_nan"] > 0, "feature"].tolist()
    report["max_abs_value"] = float(np.nanmax(np.abs(flat)))
    report["float16_safe_range"] = bool(report["max_abs_value"] < 6e4)
    # float16 precision loss relative to the typical step change
    f16_err = np.abs(flat[::50].astype(np.float16).astype(np.float32) - flat[::50])
    dsub = dx[:, ::7].reshape(-1, dx.shape[-1])
    typ_step = np.nanmedian(np.where(dsub != 0, np.abs(dsub), np.nan), 0)
    del dsub
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nanmax(f16_err, 0) / typ_step
    report["float16_err_over_typical_step_worst5"] = (
        pd.Series(ratio, index=FEATURES).sort_values(ascending=False).head(5).round(3).to_dict())
    del flat, dx

    # ---------- 2. book sanity ----------
    bf = book_features(X)
    for inst in ("i0", "i1"):
        report[f"{inst}_crossed_book_frac"] = float(bf[f"{inst}_crossed"].mean())
        report[f"{inst}_spread_quantiles"] = np.percentile(bf[f"{inst}_spread"], [1, 50, 99]).round(6).tolist()
        report[f"{inst}_frac_mid_changed"] = float((bf[f"{inst}_dmid"] != 0).mean())
    # are price-like columns all on one scale (raw prices) or centered (relative)?
    for inst in ("i0", "i1"):
        p = X[..., IDX[f"{inst}_bid_p"] + IDX[f"{inst}_ask_p"]]
        report[f"{inst}_price_like_mean_std"] = [float(p.mean()), float(p.std())]

    # ---------- 3. targets ----------
    yt = Y[:, WARMUP:, :]
    tstats = {}
    for j, t in enumerate(("t0", "t1")):
        v = yt[..., j].ravel()
        tstats[t] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "quantiles_0_1_50_99_100": np.percentile(v, [0, 1, 50, 99, 100]).round(4).tolist(),
            "frac_abs_gt_clip": float((np.abs(v) > CLIP).mean()),
            "frac_zero": float((v == 0).mean()),
            "autocorr": {lag: round(autocorr_within(yt[..., j], lag), 4)
                         for lag in (1, 5, 10, 50, 100, 500, 1000, 5000)},
            "mean_by_step_decile": [float(c.mean()) for c in np.array_split(yt[..., j], 10, axis=1)],
            "std_by_step_decile": [float(c.std()) for c in np.array_split(yt[..., j], 10, axis=1)],
            "std_across_seqs_of_seq_mean": float(yt[..., j].mean(1).std()),
        }
    tstats["corr_t0_t1"] = float(np.corrcoef(yt[..., 0].ravel(), yt[..., 1].ravel())[0, 1])
    report["targets"] = tstats

    # ---------- 4. signal ----------
    step_idx = np.arange(WARMUP, SEQ_LEN, 7)  # thin rows to save RAM; targets are autocorrelated anyway
    Xs = X[:, step_idx].reshape(-1, X.shape[-1])
    dXs = (X[:, step_idx] - X[:, step_idx - 1]).reshape(-1, X.shape[-1])
    ys = Y[:, step_idx].reshape(-1, 2)
    rows = []
    for j, t in enumerate(("t0", "t1")):
        rows.append(pd.Series(corr_cols(Xs, ys[:, j]), index=FEATURES, name=f"level_{t}"))
        rows.append(pd.Series(corr_cols(dXs, ys[:, j]), index=FEATURES, name=f"delta1_{t}"))
    sig = pd.concat(rows, axis=1)
    sig.to_csv(out / "feature_target_corr.csv")
    report["top_level_corr_t0"] = sig["level_t0"].abs().sort_values(ascending=False).head(10).round(4).to_dict()
    report["top_delta_corr_t0"] = sig["delta1_t0"].abs().sort_values(ascending=False).head(10).round(4).to_dict()
    bsig = {}
    for k, v in bf.items():
        vs = v[:, step_idx].ravel()
        if vs.std() == 0:
            continue
        bsig[k] = {t: round(float(np.corrcoef(vs, ys[:, j])[0, 1]), 4) for j, t in enumerate(("t0", "t1"))}
    report["book_feature_corr"] = bsig

    # ---------- 5. validation mask ----------
    if has_mask:
        M = np.asarray(np.load(d / "is_scored.npy", mmap_mode="r")[sel])
        report["mask"] = {
            "frac_scored_of_required": float(M[:, WARMUP:].mean()),
            "scored_in_warmup": int(M[:, :WARMUP].sum()),
            "frac_by_step_decile": [float(c.mean()) for c in np.array_split(M[:, WARMUP:], 10, axis=1)],
            "per_seq_scored_quantiles": np.percentile(M.sum(1), [0, 25, 50, 75, 100]).tolist(),
            "seqs_without_scored_rows": int((M.sum(1) == 0).sum()),
            "target_std_scored_vs_not": [float(Y[M].std()), float(Y[~M & (np.arange(SEQ_LEN) >= WARMUP)].std())],
        }
        runs = np.diff(M.astype(np.int8), axis=1)
        report["mask"]["mean_scored_runs_per_seq"] = float((runs == 1).sum(1).mean())

        # ---------- 6. what distinguishes scored rows? ----------
        # Standardized mean difference (scored vs required-but-unscored) for raw features,
        # |1-step change| of each feature, "feature changed" indicators and book features.
        Mr = M[:, WARMUP:].ravel()
        def smd(v):
            v = v[:, WARMUP:].ravel().astype(np.float64)
            a, b = v[Mr], v[~Mr]
            sd = v.std()
            return 0.0 if sd == 0 else float((a.mean() - b.mean()) / sd)
        dX = np.abs(np.diff(X, axis=1, prepend=X[:, :1]))
        rows = {}
        for j, f in enumerate(FEATURES):
            rows[f] = {"level": smd(X[..., j]), "abs_delta": smd(dX[..., j]),
                       "changed": smd((dX[..., j] != 0).astype(np.float32))}
        for k, v in bf.items():
            rows["book:" + k] = {"level": smd(v), "abs_delta": smd(np.abs(np.diff(v, axis=1, prepend=v[:, :1]))),
                                 "changed": smd((np.diff(v, axis=1, prepend=v[:, :1]) != 0).astype(np.float32))}
        drv = pd.DataFrame(rows).T
        drv.to_csv(out / "mask_drivers.csv")
        top = drv.abs().max(1).sort_values(ascending=False).head(15).index
        report["mask_drivers_top15"] = drv.loc[top].round(3).to_dict(orient="index")
        # P(scored) given that i0 / i1 trade or mid changed at this row
        for name, ind in (("i0_trade_vol_changed", (dX[..., IDX["i0_dv"]] != 0).any(-1)),
                          ("i1_trade_vol_changed", (dX[..., IDX["i1_dv"]] != 0).any(-1)),
                          ("i0_mid_changed", bf["i0_dmid"] != 0),
                          ("i1_mid_changed", bf["i1_dmid"] != 0)):
            ind = ind[:, WARMUP:].ravel()
            report["mask"][f"P(scored|{name})"] = [float(Mr[ind].mean()) if ind.any() else None,
                                                    float(Mr[~ind].mean()) if (~ind).any() else None]
        del dX

    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    print(f"\nsaved: {out}/report.json, feature_stats.csv, feature_target_corr.csv")


if __name__ == "__main__":
    main()
