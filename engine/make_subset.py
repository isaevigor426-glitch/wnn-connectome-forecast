"""Run on the Mac: copy a random subset of whole sequences (row groups) into smaller
parquet files that fit into Google Drive (free plan = 15 GB, shared with Gmail/Photos).

Row groups are copied as-is, so the data is bit-exact; one row group = one sequence.

    python make_subset.py --src ../starter/wnn_connectome_starterpack/datasets/train.parquet \
        --out ../colab_data/train_sub.parquet --n-seqs 3500
    python make_subset.py --src ../starter/wnn_connectome_starterpack/datasets/valid.parquet \
        --out ../colab_data/valid_sub.parquet --n-seqs 600
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.src)
    n = pf.num_row_groups
    sel = np.sort(np.random.default_rng(args.seed).choice(n, min(args.n_seqs, n), replace=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    t0 = time.time()
    for k, rg in enumerate(sel):
        tbl = pf.read_row_group(int(rg))
        if writer is None:
            writer = pq.ParquetWriter(args.out, tbl.schema, compression="zstd")
        writer.write_table(tbl, row_group_size=tbl.num_rows)
        if (k + 1) % 100 == 0 or k + 1 == len(sel):
            el = time.time() - t0
            size = Path(args.out).stat().st_size / 1e9
            print(f"{k + 1}/{len(sel)} seqs, {size:.2f} GB so far, eta {el / (k + 1) * (len(sel) - k - 1) / 60:.1f} min",
                  flush=True)
    writer.close()
    print(f"done: {args.out} {Path(args.out).stat().st_size / 1e9:.2f} GB, {len(sel)} of {n} sequences")


if __name__ == "__main__":
    main()
