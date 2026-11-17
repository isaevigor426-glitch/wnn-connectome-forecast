"""Convert train/valid parquet into memory-mapped .npy arrays for fast random access.

Output layout (in --out):
    features.npy   (n_seq, 20000, 112)  float32|float16
    targets.npy    (n_seq, 20000, 2)    float32
    is_scored.npy  (n_seq, 20000)       bool      (only if the source has is_scored)
    seq_ix.npy     (n_seq,)             int64     original seq_ix, in row-group order
    meta.json      sizes, dtype, nan/inf counts, feature names

Sizes: full train in float32 ~= 95 GB features (+1.7 GB targets); float16 halves it.
Use --max-seqs for a quick subset (e.g. 500 seqs ~= 4.5 GB).

Every sequence is validated: exactly 20000 rows, one seq_ix, step_in_seq == 0..19999,
need_prediction == (step >= 99).

Usage:
    python prepare_data.py --src datasets/valid.parquet --out data/valid
    python prepare_data.py --src datasets/train.parquet --out data/train --max-seqs 500
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import FEATURES, SEQ_LEN, TARGETS, WARMUP


def read_sequence(pf: pq.ParquetFile, i: int, with_scored: bool, with_targets: bool = True):
    cols = ["seq_ix", "step_in_seq", "need_prediction"] + FEATURES
    if with_targets:
        cols += TARGETS
    if with_scored:
        cols.append("is_scored")
    tbl = pf.read_row_group(i, columns=cols)
    n = tbl.num_rows
    if n != SEQ_LEN:
        raise ValueError(f"row group {i}: {n} rows, expected {SEQ_LEN}")
    seq = tbl.column("seq_ix").to_numpy()
    if seq.min() != seq.max():
        raise ValueError(f"row group {i}: several seq_ix values")
    step = tbl.column("step_in_seq").to_numpy()
    if not np.array_equal(step, np.arange(SEQ_LEN)):
        raise ValueError(f"row group {i} (seq_ix {seq[0]}): step_in_seq is not 0..{SEQ_LEN - 1}")
    need = tbl.column("need_prediction").to_numpy(zero_copy_only=False).astype(bool)
    if not np.array_equal(need, step >= WARMUP):
        raise ValueError(f"row group {i}: need_prediction != (step >= {WARMUP})")
    x = np.column_stack([tbl.column(c).to_numpy(zero_copy_only=False) for c in FEATURES]).astype(np.float32)
    y = (np.column_stack([tbl.column(c).to_numpy(zero_copy_only=False) for c in TARGETS]).astype(np.float32)
         if with_targets else None)
    sc = tbl.column("is_scored").to_numpy(zero_copy_only=False).astype(bool) if with_scored else None
    return int(seq[0]), x, y, sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="feature dtype; check EDA value ranges before using float16")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.src)
    names = set(pf.schema_arrow.names)
    missing = [c for c in FEATURES + TARGETS if c not in names]
    if missing:
        raise SystemExit(f"columns missing in {args.src}: {missing[:5]}...")
    with_scored = "is_scored" in names
    n = pf.num_row_groups if args.max_seqs is None else min(args.max_seqs, pf.num_row_groups)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(args.dtype)
    X = np.lib.format.open_memmap(out / "features.npy", "w+", dt, (n, SEQ_LEN, len(FEATURES)))
    Y = np.lib.format.open_memmap(out / "targets.npy", "w+", np.float32, (n, SEQ_LEN, 2))
    S = np.lib.format.open_memmap(out / "is_scored.npy", "w+", bool, (n, SEQ_LEN)) if with_scored else None
    seq_ids = np.empty(n, np.int64)

    nan_x = inf_x = nan_y = 0
    f16_overflow = 0
    t0 = time.time()
    for i in range(n):
        sid, x, y, sc = read_sequence(pf, i, with_scored)
        seq_ids[i] = sid
        nan_x += int(np.isnan(x).sum())
        inf_x += int(np.isinf(x).sum())
        nan_y += int((~np.isfinite(y)).sum())
        if dt == np.float16:
            f16_overflow += int((np.abs(x[np.isfinite(x)]) > 65504).sum())
        X[i] = x
        Y[i] = y
        if S is not None:
            S[i] = sc
        if (i + 1) % 100 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"{i + 1}/{n} seqs  {el:.0f}s  eta {el / (i + 1) * (n - i - 1):.0f}s", flush=True)

    if len(np.unique(seq_ids)) != n:
        raise SystemExit("duplicate seq_ix across row groups")
    np.save(out / "seq_ix.npy", seq_ids)
    for a in (X, Y, S):
        if a is not None:
            a.flush()
    meta = dict(src=str(args.src), n_seqs=n, seq_len=SEQ_LEN, n_features=len(FEATURES),
                dtype=args.dtype, has_is_scored=with_scored, nan_features=nan_x,
                inf_features=inf_x, nonfinite_targets=nan_y, float16_overflow=f16_overflow,
                scored_rows=int(S.sum()) if S is not None else None, features=FEATURES)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))


if __name__ == "__main__":
    main()
