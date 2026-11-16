"""Regenerate the six source-backed README figures.

Run from the repository root: python docs/figures.py
All scores are drawn from docs/results/*.csv; names describe model recipes,
not individual contributors. The plots do not claim controlled causal effects
where submissions also changed training data or model checkpoints.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IMG = ROOT / "img"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#9a998f", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURF,
    "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
})


def _title(fig, title, sub=None, x=0.045):
    fig.text(x, 0.963, title, fontsize=14, weight="semibold", color=INK, va="top")
    if sub:
        fig.text(x, 0.890, sub, fontsize=9.4, color=INK2, va="top")


def _finish(fig, name):
    fig.savefig(IMG / name, dpi=200)
    plt.close(fig)


def _submission(file):
    """Lookup unique submission with this exact filename in the recorded results."""
    df = pd.read_csv(ROOT / "results/submissions.csv")
    matches = df[df.file == file]
    assert len(matches) == 1, (file, len(matches))
    return matches.iloc[0]


def progress():
    """Leaderboard history; annotated snapshots, not isolated ablations."""
    df = pd.read_csv(ROOT / "results/submissions.csv", parse_dates=["time_msk"])
    df = df.dropna(subset=["public"]).sort_values("time_msk").reset_index(drop=True)
    df["best"] = df["public"].cummax()
    record = df.public >= df.best - 1e-12
    shown = df.public > 0.63
    fig, ax = plt.subplots(figsize=(10.3, 5.1), dpi=200)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.step(df.time_msk, df.best, where="post", color=BLUE, lw=2, label="best public score")
    ax.scatter(df.loc[shown & ~record, "time_msk"], df.loc[shown & ~record, "public"],
               s=34, color=MUTED, edgecolor=SURF, lw=1.5, zorder=3, label="other submissions")
    ax.scatter(df.loc[shown & record, "time_msk"], df.loc[shown & record, "public"],
               s=47, color=BLUE, edgecolor=SURF, lw=2, zorder=4, label="new public best")

    milestones = [
        ("submission.zip", 0.63988, "GRU 2 x 128\n0.6399", 0.30, -0.0043),
        ("submission_gru_6573.zip", 0.64896, "GRU trained with valid labels\n0.6490", 0.0, 0.0030),
        ("ens3_submission.zip", 0.65545, "3-GRU ensemble\n0.6555", -3.65, 0.0021),
        ("final5_mx16_submission.zip", 0.65709, "3 GRU + 2 LSTM\n0.6571", -2.50, 0.0015),
    ]
    for filename, score, label, dx, dy in milestones:
        m = df[(df.file == filename) & (np.isclose(df.public, score))]
        assert len(m) == 1, filename
        r = m.iloc[0]
        ax.annotate(label, (r.time_msk, r.public),
                    xytext=(r.time_msk + pd.Timedelta(days=dx), r.public + dy),
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.85))
    ax.set_ylim(0.635, 0.6604)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("public Weighted Pearson")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Public leaderboard: model development", 
           "Selected milestones (not controlled ablations). Baseline 0.59353 is below the axis.", x=0.075)
    fig.subplots_adjust(top=0.825, left=0.085, right=0.985, bottom=0.105)
    _finish(fig, "progress.png")


def models():
    """Recipe names are grounded in actual submission files and packed final weights."""
    selected = [
        ("gru5k_trainonly_submission.zip", "GRU 2x128 | train only", "single"),
        ("mamba_v1.zip", "Mamba | single model", "single"),
        ("submission_gru_6573.zip", "GRU 2x128 | train + valid subset", "single"),
        ("ft_baseline_full.zip", "GRU 2x128 | fine-tuned baseline", "single"),
        ("submission_lstm_float32_trial.zip", "LSTM 2x128 | single model", "single"),
        ("ens_ftswa_grufull.zip", "2 GRU | raw + differences/MLP", "ensemble"),
        ("ens3_submission.zip", "3 GRU | first combined ensemble", "ensemble"),
        ("ens5act_submission.zip", "GRU ensemble | active-sequence member", "ensemble"),
        ("final5_act3_submission.zip", "5 GRU | active-sequence variants", "ensemble"),
        ("final5_mx16_submission.zip", "3 GRU + 2 LSTM | final blend", "ensemble"),
    ]
    rows = [(label, kind, float(_submission(filename).public)) for filename, label, kind in selected]
    fig, ax = plt.subplots(figsize=(10.6, 5.4), dpi=200)
    ys = np.arange(len(rows))[::-1]
    for y, (label, kind, v) in zip(ys, rows):
        ax.hlines(y, 0.640, v, color=GRID, lw=1.2)
        ax.scatter(v, y, s=60, color=BLUE if kind == "ensemble" else MUTED,
                   edgecolor=SURF, lw=2, zorder=3)
        ax.text(v + 0.00038, y, f"{v:.5f}", va="center", fontsize=9.1, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9.2)
    ax.set_xlim(0.640, 0.6606)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("public Weighted Pearson (higher is better)")
    ax.scatter([], [], color=MUTED, s=42, label="single model")
    ax.scatter([], [], color=BLUE, s=42, label="ensemble")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Single networks vs recurrent ensembles",
           "Architecture and training recipe, not contributor names. Each dot is one scored submission.")
    fig.subplots_adjust(top=0.83, left=0.355, right=0.974, bottom=0.11)
    _finish(fig, "models.png")


def runtime():
    """Observed wall time; avoid attributing changes to weight precision alone."""
    df = pd.read_csv(ROOT / "results/submissions.csv")
    df = df[df.duration_min.notna()]
    scored = df[df.public.notna() & (df.public > 0.63)]
    fig, ax = plt.subplots(figsize=(10.2, 4.9), dpi=200)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(60, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(59, 0.6508, "60-min limit", ha="right", fontsize=9, color=INK2)
    for kind, color, legend in (("single", MUTED, "single model"), ("ensemble", BLUE, "ensemble")):
        d = scored[scored.kind == kind]
        ax.scatter(d.duration_min, d.public, s=46, color=color, edgecolor=SURF,
                   lw=1.5, zorder=3, label=legend)
    timeout = df[df.public.isna()]
    if not timeout.empty:
        ax.scatter(timeout.duration_min, np.repeat(0.6386, len(timeout)),
                   marker="x", s=50, color=ORANGE, lw=2, zorder=3,
                   label="timeout (no public score)")
    for fname, label, dx, dy in [
        ("final5_2lstm_submission.zip", "FP32 LSTM weights: 56.8 min\nWP 0.65689", -8.2, 0.0016),
        ("final5_mx16_submission.zip", "FP16 LSTM weights: 43.5 min\nWP 0.65709", -19.0, 0.0016),
    ]:
        row = _submission(fname)
        ax.annotate(label, (row.duration_min, row.public),
                    xytext=(row.duration_min + dx, row.public + dy),
                    fontsize=8.9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(15, 65)
    ax.set_ylim(0.637, 0.6596)
    ax.set_xlabel("test runtime on one CPU core (minutes)")
    ax.set_ylabel("public Weighted Pearson")
    ax.legend(loc="lower left", frameon=False, fontsize=8.8, labelcolor=INK2)
    _title(fig, "Inference time vs public score",
           "Both use 3 GRU + 2 LSTM; LSTM checkpoints also differ, so this is not a precision-only ablation.")
    fig.subplots_adjust(top=0.82, left=0.093, right=0.976, bottom=0.12)
    _finish(fig, "runtime.png")


def importance():
    """Source is permutation importance data, not an interpretation of book depth ordering."""
    df = pd.read_csv(ROOT / "results/feature_importance.csv")
    base = float(df.iloc[0].is_scored)
    d = df.iloc[1:].copy()
    d["drop"] = base - d.is_scored
    mapping = {
        "instrument 0 prices": "i0 price-like (22 cols)",
        "instrument 0 volumes": "i0 volume-like (22 cols)",
        "instrument 0 trades": "i0 trades (8 cols)",
        "instrument 1 prices": "i1 price-like (22 cols)",
        "instrument 1 volumes": "i1 volume-like (22 cols)",
        "instrument 1 trades": "i1 trades (8 cols)",
    }
    d["label"] = d.group.map(lambda x: mapping.get(x, x))
    d = d.sort_values("drop")
    fig, ax = plt.subplots(figsize=(10.2, 5.5), dpi=200)
    ax.barh(d.label, d["drop"], color=BLUE, height=0.62, edgecolor=SURF, lw=2)
    for label, v in zip(d.label, d["drop"]):
        ax.text(v + 0.0011, label, f"{v:.3f}", va="center", fontsize=8.8, color=INK2)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9.2)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("drop in local scored-row Weighted Pearson after shuffling feature group")
    _title(fig, "Permutation importance: which input groups matter?",
           "One 2-layer GRU (reference WP = 0.657). i0 volume-like and price-like inputs dominate.")
    fig.subplots_adjust(top=0.83, left=0.28, right=0.972, bottom=0.105)
    _finish(fig, "importance.png")


def shift():
    """Sequence-level mixture proportions; separate from the activity-rate finding."""
    df = pd.read_csv(ROOT / "results/family_shift.csv")
    fig, ax = plt.subplots(figsize=(10.2, 4.9), dpi=200)
    x, width = np.arange(len(df)), 0.36
    for dx, field, color, legend in [
        (-width / 2 - 0.01, "train_share", MUTED, "train: 10,607 sequences"),
        (width / 2 + 0.01, "valid_share", BLUE, "valid: 1,873 sequences"),
    ]:
        ax.bar(x + dx, df[field], width, color=color, label=legend)
        for xx, value in zip(x + dx, df[field]):
            ax.text(xx, value + 0.006, f"{value:.1%}", ha="center", fontsize=9.2, color=INK2)
    ax.set_xticks(x, [f"{fam}\nunit = {unit}" for fam, unit in zip(df.family, df.unit)])
    ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("share of sequences")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Distribution shift: train vs valid",
           "Target family mix differs. Train/valid classifier AUC = 0.995; price-event rate: 0.28 vs 0.37.")
    fig.subplots_adjust(top=0.81, left=0.10, right=0.975, bottom=0.15)
    _finish(fig, "shift.png")


def experiments():
    """Training ablations, with observed three-seed baseline spread marked."""
    df = pd.read_csv(ROOT / "results/experiments_3k.csv")
    control = df[df.experiment.str.startswith("baseline")]
    low, high, mean = control.is_scored.min(), control.is_scored.max(), control.is_scored.mean()
    d = df[~df.experiment.str.startswith("baseline")].copy()
    mapping = {
        "Pearson + weighted MSE loss": "Weighted MSE + corr term",
        "family re-weighting to valid mix": "Reweight target-family mix",
        "GRU 2x256": "GRU hidden 256 (vs 128)",
        "event features": "Add 8 event-derived inputs",
        "dropout 0.2, weight decay 1e-4": "Dropout .2 + weight decay 1e-4",
        "input minus rolling mean (500)": "Input - 500-step rolling mean",
        "train sequences most similar to valid": "Train subset closest to valid",
    }
    d["label"] = d.experiment.map(lambda s: mapping.get(s, s))
    d = d.sort_values("is_scored")
    fig, ax = plt.subplots(figsize=(10.3, 4.95), dpi=200)
    ax.axvspan(low, high, color=GRID, alpha=0.90, lw=0)
    ax.axvline(mean, color=INK2, lw=1)
    ax.scatter(d.is_scored, d.label, s=57, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
    for v, label in zip(d.is_scored, d.label):
        ax.text(v + 0.00014, label, f"{v - mean:+.4f}", va="center", fontsize=9.1, color=INK2)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9.2)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0.6637, 0.6698)
    ax.set_xlabel("local Weighted Pearson on scored validation rows")
    _title(fig, "Ablations vs random-seed variation",
           f"3,000-sequence GRU, 3 epochs. Grey band: 3 baseline seeds ({low:.4f}-{high:.4f}); each change: one run.")
    fig.subplots_adjust(top=0.81, left=0.327, right=0.97, bottom=0.12)
    _finish(fig, "experiments.png")


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    for fn in (progress, models, runtime, importance, shift, experiments):
        fn()
