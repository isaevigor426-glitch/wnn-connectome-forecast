"""Local scorer + speed meter that mimics the platform's row-by-row protocol.

Feeds every row of valid.parquet to PredictionModel.predict(DataPoint), one at a time,
with one model instance for all sequences (so state resets are exercised), then
reports:
  * WP on the validation mask (is_scored) — comparable to the leaderboard,
  * WP on all need_prediction rows — a less mask-dependent view,
  * microseconds per row and the projected time for the 39.4M-row test set,
  * protocol violations (None on required rows, wrong shape, NaN/Inf, ...).

Usage:
    python local_eval.py --solution-dir path/to/baseline --data datasets/valid.parquet --max-seqs 50
    python local_eval.py --solution-dir dummy_solution --data datasets/valid.parquet --check-determinism 3
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import SEQ_LEN, TEST_ROWS, TIME_LIMIT_S
from metric import WEIGHT_MODES, WPAccumulator
from prepare_data import read_sequence


OfficialAccumulator = None  # set by load_solution if the starter pack's utils.py is found


@dataclasses.dataclass
class _FallbackDataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load_solution(solution_dir: str):
    """Import solution.py the way the platform does: its folder first on sys.path and as cwd."""
    sol_dir = Path(solution_dir).resolve()
    if not (sol_dir / "solution.py").exists():
        raise SystemExit(f"{sol_dir}/solution.py not found")
    # utils.py may sit next to solution.py or one level up (starter pack layout)
    for d in (sol_dir.parent, sol_dir):
        sys.path.insert(0, str(d))
    os.chdir(sol_dir)
    global OfficialAccumulator
    try:
        import utils  # type: ignore  # the starter pack's utils.py
        DataPoint = utils.DataPoint
        OfficialAccumulator = getattr(utils, "GlobalAccumulator", None)
        print(f"using DataPoint/GlobalAccumulator from {utils.__file__}")
    except Exception:
        DataPoint = _FallbackDataPoint
        print("utils.py not found: using fallback DataPoint, no official scorer cross-check")
    spec = importlib.util.spec_from_file_location("solution", sol_dir / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel, DataPoint


def make_dp(DataPoint, seq_ix, step, need, state):
    try:
        return DataPoint(seq_ix=seq_ix, step_in_seq=step, need_prediction=need, state=state)
    except TypeError:
        return DataPoint(seq_ix, step, need, state)


def run(PredictionModel, DataPoint, pf, seq_order, with_scored, model=None, verbose=True):
    model = model or PredictionModel()
    preds = np.zeros((len(seq_order), SEQ_LEN, 2), np.float32)
    acc_mask = {m: WPAccumulator(2, m) for m in WEIGHT_MODES}
    acc_full = {m: WPAccumulator(2, m) for m in WEIGHT_MODES}
    official = OfficialAccumulator() if (OfficialAccumulator is not None and with_scored) else None
    errors = {"none_on_required": 0, "bad_shape": 0, "non_finite": 0, "output_on_warmup": 0}
    t_pred = 0.0
    t_start = time.perf_counter()
    rows = 0
    for k, rg in enumerate(seq_order):
        seq_ix, x, y, sc = read_sequence(pf, rg, with_scored)
        need = np.arange(SEQ_LEN) >= 99
        for i in range(SEQ_LEN):
            dp = make_dp(DataPoint, seq_ix, i, bool(need[i]), x[i].copy())
            t0 = time.perf_counter()
            out = model.predict(dp)
            t_pred += time.perf_counter() - t0
            if not need[i]:
                if out is not None:
                    errors["output_on_warmup"] += 1
                continue
            if out is None:
                errors["none_on_required"] += 1
                continue
            out = np.asarray(out, np.float32)          # the official scorer needs exactly (2,), no reshaping
            if out.shape != (2,):
                errors["bad_shape"] += 1
                continue
            if not np.all(np.isfinite(out)):
                errors["non_finite"] += 1
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            preds[k, i] = out
        rows += SEQ_LEN
        for m in WEIGHT_MODES:
            acc_full[m].update(y[need], preds[k][need])
            if sc is not None:
                acc_mask[m].update(y, preds[k], sc & need)
        if official is not None:
            official.add(y, preds[k], sc & need)
        if verbose and ((k + 1) % 10 == 0 or k + 1 == len(seq_order)):
            us = (time.perf_counter() - t_start) / rows * 1e6
            print(f"[{k + 1}/{len(seq_order)}] {us:.1f} us/row total -> test ~{us * TEST_ROWS / 1e6 / 60:.1f} min"
                  f" | WP(mask)={acc_mask['abs_target'].score() if sc is not None else float('nan'):.4f}",
                  flush=True)
    wall = time.perf_counter() - t_start
    return dict(preds=preds, acc_mask=acc_mask, acc_full=acc_full, errors=errors, official=official,
                t_pred=t_pred, wall=wall, rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution-dir", required=True)
    ap.add_argument("--data", required=True, help="valid.parquet (needs targets)")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true", help="random sequence order (seeded)")
    ap.add_argument("--check-determinism", type=int, default=0, metavar="N",
                    help="also run the first N seqs twice with fresh models and compare")
    ap.add_argument("--save-preds", default=None, help="save predictions .npy for later analysis")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args()

    data = str(Path(args.data).resolve())
    save_preds = str(Path(args.save_preds).resolve()) if args.save_preds else None
    report = str(Path(args.report).resolve()) if args.report else None
    PredictionModel, DataPoint = load_solution(args.solution_dir)
    pf = pq.ParquetFile(data)
    with_scored = "is_scored" in pf.schema_arrow.names
    n = pf.num_row_groups if args.max_seqs is None else min(args.max_seqs, pf.num_row_groups)
    order = np.arange(n)
    if args.shuffle:
        order = np.random.default_rng(0).permutation(order)

    r = run(PredictionModel, DataPoint, pf, order, with_scored)

    us_total = r["wall"] / r["rows"] * 1e6
    us_pred = r["t_pred"] / r["rows"] * 1e6
    proj_min = us_total * TEST_ROWS / 1e6 / 60
    res = {
        "seqs": int(n), "rows": r["rows"],
        "us_per_row_predict": round(us_pred, 2),
        "us_per_row_total": round(us_total, 2),
        "projected_test_minutes": round(proj_min, 1),
        "budget_minutes": TIME_LIMIT_S / 60,
        "errors": r["errors"],
    }
    for m in WEIGHT_MODES:
        if with_scored:
            res[f"WP_mask_{m}"] = {"t0": r["acc_mask"][m].per_target()[0],
                                   "t1": r["acc_mask"][m].per_target()[1],
                                   "score": r["acc_mask"][m].score()}
        res[f"WP_all_{m}"] = {"t0": r["acc_full"][m].per_target()[0],
                              "t1": r["acc_full"][m].per_target()[1],
                              "score": r["acc_full"][m].score()}

    if r["official"] is not None and r["official"].selected_rows:
        o = r["official"].result()
        res["WP_official_utils"] = {"t0": o["t0"], "t1": o["t1"], "score": o["weighted_pearson"]}
        res["metric_matches_official"] = abs(o["weighted_pearson"] - res["WP_mask_abs_target"]["score"]) < 1e-6

    if args.check_determinism:
        k = min(args.check_determinism, n)
        a = run(PredictionModel, DataPoint, pf, order[:k], with_scored, verbose=False)["preds"]
        b = run(PredictionModel, DataPoint, pf, order[:k], with_scored, verbose=False)["preds"]
        res["determinism_max_abs_diff"] = float(np.abs(a - b).max())
        # reset check: the same sequence after another one must give identical output
        if k >= 2:
            m = PredictionModel()
            c = run(PredictionModel, DataPoint, pf, [order[1], order[0]], with_scored, model=m, verbose=False)["preds"]
            res["reset_check_max_abs_diff"] = float(np.abs(c[1] - a[0]).max())

    if save_preds:
        np.save(save_preds, r["preds"])
    print(json.dumps(res, indent=1))
    if report:
        Path(report).write_text(json.dumps(res, indent=1))

    bad = sum(r["errors"].values())
    if bad:
        print(f"!! {bad} protocol violations, see 'errors'")
    if proj_min > TIME_LIMIT_S / 60 * 0.7:
        print(f"!! projected {proj_min:.0f} min is above 70% of the 60-min budget on this machine")


if __name__ == "__main__":
    main()
