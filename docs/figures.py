"""Figures for the README. Run: python docs/figures.py"""
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
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def progress():
    df = pd.read_csv(ROOT / "results/submissions.csv", parse_dates=["time_msk"]).dropna(subset=["public"])
    df = df.sort_values("time_msk").reset_index(drop=True)
    df["best"] = df["public"].cummax()
    record = df["public"] >= df["best"]
    shown = df[df["public"] > 0.63]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.step(df["time_msk"], df["best"], where="post", color=BLUE, lw=2, label="best so far")
    ax.scatter(shown.loc[~record, "time_msk"], shown.loc[~record, "public"], s=34, color=MUTED,
               edgecolor=SURF, lw=1.5, zorder=3, label="other submissions")
    ax.scatter(df.loc[record & (df.public > 0.63), "time_msk"], df.loc[record & (df.public > 0.63), "public"],
               s=46, color=BLUE, edgecolor=SURF, lw=2, zorder=4, label="new best")

    notes = {  # file -> (text, dx days, dy)
        "submission.zip": ("first GRU\n0.6399", 0.3, -0.0045),
        "submission_gru_6573.zip": ("+ validation data in training\n0.6490", -0.2, 0.0030),
        "ens3_submission.zip": ("first ensemble with Andrew's models\n0.6555", -5.6, 0.0028),
        "final5_mx16_submission.zip": ("final: 3 GRU + 2 LSTM\n0.6571", -2.6, 0.0016),
    }
    for f, (txt, dx, dy) in notes.items():
        r = df[(df.file == f) & record].iloc[0]
        ax.annotate(txt, (r.time_msk, r.public), xytext=(r.time_msk + pd.Timedelta(days=dx), r.public + dy),
                    fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

    ax.set_ylim(0.635, 0.6605)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("public score")
    fig.text(0.07, 0.95, "Public score over the competition", fontsize=13, weight="semibold", color=INK)
    fig.text(0.07, 0.905, "Organisers' baseline: 0.5935 (below the axis). Final place 471 of 5838.", fontsize=9.5, color=INK2)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.1)
    fig.savefig(IMG / "progress.png")


def _title(fig, title, sub=None):
    fig.text(0.02, 0.95, title, fontsize=13, weight="semibold", color=INK, va="top")
    if sub:
        fig.text(0.02, 0.885, sub, fontsize=9.5, color=INK2, va="top")


def models():
    """Best single models of each kind vs the ensembles (public score)."""
    rows = [
        ("GRU, train only", "single", 0.64340), ("Mamba (Andrew)", "single", 0.64346),
        ("GRU, train + valid", "single", 0.64896), ("GRU, fine-tuned baseline (Andrew)", "single", 0.65083),
        ("LSTM (Max)", "single", 0.65260),
        ("Andrew's 2 models + my GRU", "ensemble", 0.65545), ("+ GRU on active sequences", "ensemble", 0.65635),
        ("final: 3 active GRU", "ensemble", 0.65673), ("final: 3 GRU + 2 LSTM", "ensemble", 0.65709),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    y = range(len(rows))[::-1]
    for yi, (name, kind, v) in zip(y, rows):
        c = BLUE if kind == "ensemble" else MUTED
        ax.hlines(yi, 0.640, v, color=GRID, lw=1.2)
        ax.scatter(v, yi, s=60, color=c, edgecolor=SURF, lw=2, zorder=3)
        ax.text(v + 0.0004, yi, f"{v:.4f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), [r[0] for r in rows]); ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0.640, 0.6595); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.set_xlabel("public score")
    ax.scatter([], [], color=MUTED, s=40, label="single model"); ax.scatter([], [], color=BLUE, s=40, label="ensemble")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Single models vs ensembles", "The best single model is 0.6526; averaging models with different errors adds another 0.0045.")
    fig.subplots_adjust(top=0.82, left=0.3, right=0.97, bottom=0.12)
    fig.savefig(IMG / "models.png")


def runtime():
    """Score vs inference time; the limit is 60 minutes on one CPU core."""
    df = pd.read_csv(ROOT / "results/submissions.csv")
    df = df[df.duration_min.notna()]
    ok = df[df.public.notna() & (df.public > 0.63)]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axvline(60, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(59.3, 0.6582, "60-min limit", ha="right", fontsize=9, color=INK2)
    for kind, c, lab in (("single", MUTED, "single model"), ("ensemble", BLUE, "ensemble")):
        d = ok[ok.kind == kind]
        ax.scatter(d.duration_min, d.public, s=46, color=c, edgecolor=SURF, lw=1.5, zorder=3, label=lab)
    t = df[df.public.isna()].iloc[0]
    ax.scatter(t.duration_min, 0.6385, marker="x", s=50, color=ORANGE, lw=2, zorder=3, label="timed out (no score)")
    for f, txt, dx, dy in (("final5_2lstm_submission.zip", "2 LSTM, float32\n56.8 min", -9.5, 0.0003),
                           ("final5_mx16_submission.zip", "2 LSTM (one is Max's), float16\n43.5 min", -15, 0.0006)):
        r = df[df.file == f].iloc[0]
        ax.annotate(txt, (r.duration_min, r.public), xytext=(r.duration_min + dx, r.public + dy), fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xlim(15, 65); ax.set_ylim(0.637, 0.6595)
    ax.set_xlabel("inference time, minutes"); ax.set_ylabel("public score")
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Score vs inference time", "float16 LSTM weights did not change the local score and cut inference from 56.8 to 43.5 minutes.")
    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.12)
    fig.savefig(IMG / "runtime.png")


def importance():
    """Permutation importance by column group: drop of the scored-rows metric."""
    df = pd.read_csv(ROOT / "results/feature_importance.csv")
    ref = df.iloc[0].is_scored; d = df.iloc[1:].copy(); d["drop"] = ref - d.is_scored
    d = d.sort_values("drop")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    ax.barh(d.group, d["drop"], color=BLUE, height=0.62, edgecolor=SURF, lw=2)
    for g, v in zip(d.group, d["drop"]):
        ax.text(v + 0.001, g, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.tick_params(axis="y", length=0, labelcolor=INK); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("score drop when the group is shuffled")
    _title(fig, "Which inputs matter", "Permutation importance of a single GRU on scored rows (reference 0.657). Instrument 0 volumes and prices carry most of the signal.")
    fig.subplots_adjust(top=0.84, left=0.24, right=0.96, bottom=0.11)
    fig.savefig(IMG / "importance.png")


def shift():
    """Share of target families in train vs valid."""
    df = pd.read_csv(ROOT / "results/family_shift.csv")
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    x = np.arange(len(df)); w = 0.36
    for off, col, c, lab in ((-w / 2 - 0.01, "train_share", MUTED, "train"), (w / 2 + 0.01, "valid_share", BLUE, "valid")):
        ax.bar(x + off, df[col], w, color=c, label=lab)
        for xi, v in zip(x + off, df[col]):
            ax.text(xi, v + 0.006, f"{v:.0%}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(x, [f"{f}\nu = {u}" for f, u in zip(df.family, df.unit)]); ax.tick_params(axis="x", length=0, labelcolor=INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylabel("share of sequences")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    _title(fig, "Train and valid are different data",
           "Target families are mixed differently; a classifier tells train from valid with AUC 0.995. Valid has 0.37 price events per step, train 0.28.")
    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.14)
    fig.savefig(IMG / "shift.png")


def experiments():
    """Ideas tested on a 3000-sequence GRU (3 epochs) against the seed-to-seed spread."""
    df = pd.read_csv(ROOT / "results/experiments_3k.csv")
    base = df[df.experiment.str.startswith("baseline")]
    lo, hi, mid = base.is_scored.min(), base.is_scored.max(), base.is_scored.mean()
    d = df[~df.experiment.str.startswith("baseline")].sort_values("is_scored")
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    ax.axvspan(lo, hi, color=GRID, alpha=0.9, lw=0)
    ax.axvline(mid, color=INK2, lw=1)
    ax.text(hi + 0.0001, -0.45, "spread of the baseline\nover 3 seeds", fontsize=8.5, color=INK2, va="bottom")
    ax.scatter(d.is_scored, d.experiment, s=56, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
    for v, e in zip(d.is_scored, d.experiment):
        ax.text(v + 0.00015, e, f"{v - mid:+.4f}", va="center", fontsize=9, color=INK2)
    ax.tick_params(axis="y", length=0, labelcolor=INK); ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(d) - 0.4); ax.set_xlim(None, 0.6686)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlabel("local score on scored rows")
    _title(fig, "Few ideas beat seed noise", "GRU on 3000 sequences, 3 epochs; one run per idea. Grey band: the same baseline over 3 seeds, so a change inside it is noise.")
    fig.subplots_adjust(top=0.82, left=0.33, right=0.96, bottom=0.12)
    fig.savefig(IMG / "experiments.png")


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    for fn in (progress, models, runtime, importance, shift, experiments):
        fn()
