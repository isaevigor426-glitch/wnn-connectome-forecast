"""Compare the per-row speed of exported submissions on THIS machine (1 thread).

The platform limit is 60 min for 39.4M rows (~91 us/row including its own overhead). The only
safe reference we have is a submission that already passed: our GRU (gru128x2_v1). A new model
is safe if it is not slower than that reference on the same machine.

    python bench_speed.py submissions/gru128x2_v1 submissions/mamba_v1
Each folder must contain solution.py (+ weights). Runs are interleaved and repeated; the minimum
is reported because noise only ever adds time.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


@dataclasses.dataclass
class DP:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load(folder):
    folder = Path(folder).resolve()
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"sol_{abs(hash(folder))}", folder / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.pop(0)
    return mod.PredictionModel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    X = np.random.default_rng(0).standard_normal((args.rows, 112)).astype(np.float32)
    dps = [DP(1, i, i >= 99, X[i]) for i in range(args.rows)]
    models = {f: load(f) for f in args.folders}
    best = {f: float("inf") for f in args.folders}
    for r in range(args.repeats):
        for f, m in models.items():
            seq = 10_000 + r                                 # a new sequence id starting at step 0 -> state reset
            run = [DP(seq, dp.step_in_seq, dp.need_prediction, dp.state) for dp in dps]
            t = time.perf_counter()
            for dp in run:
                m.predict(dp)
            best[f] = min(best[f], (time.perf_counter() - t) / args.rows * 1e6)
    ref = args.folders[0]
    print(f"{'submission':45s} {'us/row':>8s} {'vs first':>9s}")
    for f in args.folders:
        print(f"{f:45s} {best[f]:8.1f} {best[f] / best[ref]:8.2f}x")
    print("rule of thumb: <= 1.00x the GRU that already passed on the platform = safe")


if __name__ == "__main__":
    main()
