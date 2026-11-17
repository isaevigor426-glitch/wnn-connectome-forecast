"""Synthetic parquet that mimics the competition layout (for tests and dry runs).

Order books: mid is a random walk whose drift depends on top-of-book imbalance;
book indices are shuffled per sequence (index != depth level, as in the real data).
Targets: t0 = future mid move over 20 steps, t1 = over 100 steps, scaled, so they
are partly predictable from the imbalance (a linear probe must get a positive score).

    python tests/synth.py --out synthetic/valid.parquet --n-seqs 4 --with-scored
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEATURES, SEQ_LEN, WARMUP  # noqa: E402


def make_sequence(rng: np.random.Generator, seq_ix: int, T: int = SEQ_LEN):
    x = np.zeros((T, 112), np.float32)
    mids = []
    for k in range(2):
        base = 52 * k
        imb = np.zeros(T)
        e = rng.standard_normal(T)
        for t in range(1, T):
            imb[t] = 0.98 * imb[t - 1] + 0.2 * e[t]
        imb = np.tanh(imb)
        mid = 100 + np.cumsum(0.02 * imb + 0.05 * rng.standard_normal(T))
        tick = 0.01
        perm_b = rng.permutation(11)
        perm_a = rng.permutation(11)
        for lvl in range(11):
            x[:, base + perm_b[lvl]] = np.round(mid / tick - 1 - lvl) * tick
            x[:, base + 11 + perm_a[lvl]] = np.round(mid / tick + 1 + lvl) * tick
            vb = np.abs(rng.normal(10, 3, T))
            va = np.abs(rng.normal(10, 3, T))
            if lvl == 0:
                vb = vb * (1 + imb)
                va = va * (1 - imb)
            x[:, base + 22 + perm_b[lvl]] = vb
            x[:, base + 33 + perm_a[lvl]] = va
        x[:, base + 44:base + 48] = (mid[:, None] + 0.01 * rng.standard_normal((T, 4)))
        x[:, base + 48:base + 52] = np.abs(rng.normal(1, 1, (T, 4)))
        mids.append(mid)
    x[:, 104:112] = rng.standard_normal((T, 8))
    mid0 = mids[0]

    def fwd(h):
        f = np.zeros(T)
        f[:-h] = mid0[h:] - mid0[:-h]
        return f

    t0 = (fwd(20) / 0.3).astype(np.float32)
    t1 = (fwd(100) / 0.6).astype(np.float32)
    return x, t0, t1


def write(out: str, n_seqs: int, with_scored: bool, seed: int = 0, T: int = SEQ_LEN,
          shuffle_ids: bool = True) -> None:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(10_000)[:n_seqs] if shuffle_ids else np.arange(n_seqs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for s in range(n_seqs):
        x, t0, t1 = make_sequence(rng, int(ids[s]), T)
        step = np.arange(T, dtype=np.int32)
        cols = {"seq_ix": np.full(T, ids[s], np.int64), "step_in_seq": step,
                "need_prediction": step >= WARMUP}
        if with_scored:
            sc = np.zeros(T, bool)
            for st in rng.integers(WARMUP, T - 500, 8):
                sc[st:st + 400] = True
            cols["is_scored"] = sc
        for j, name in enumerate(FEATURES):
            cols[name] = x[:, j]
        cols["t0"], cols["t1"] = t0, t1
        tbl = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter(out, tbl.schema)
        writer.write_table(tbl, row_group_size=T)
    writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--with-scored", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write(a.out, a.n_seqs, a.with_scored, a.seed)
    print(f"wrote {a.out}")
