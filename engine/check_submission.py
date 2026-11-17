"""Pre-submit check of a competition zip: catches what makes the platform fail.

    python check_submission.py submission.zip
    python check_submission.py submission.zip --data valid.parquet --ref-dir starter/.../baseline

What it checks
  1. Archive: opens, not corrupted, <= 20 MB, solution.py at the ROOT (not in a subfolder).
  2. Code: Python 3.11 syntax, class PredictionModel with predict(), imports outside the scorer
     image (numpy, onnxruntime, pyarrow, numba, torch, jax, tensorflow) flagged, hard-coded paths, CUDA.
  3. ONNX: weights stored in a separate file (external data, e.g. model.onnx.data) that is
     not in the zip -> the classic "works on my machine" failure. Needs `pip install onnx`.
  4. Run, exactly like the platform: the zip is unpacked into an empty folder, a fresh Python
     process with 1 thread and a DIFFERENT working directory creates PredictionModel() and
     feeds rows one by one. Checked with the official scorer's rules: None on warm-up rows,
     shape exactly (2,) and finite values on required rows, determinism (two fresh models give bit-identical output), state reset
     (a sequence gives the same output after another sequence as on a fresh model).
  5. Speed: us/row and projected minutes for the 39.4M-row test on THIS machine; with
     --ref-dir also the ratio to a reference that is known to pass (organizers' baseline or
     an earlier accepted submission) - the ratio is what transfers to the platform's CPU.

Exit code 1 if anything FAILs.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path

LIMIT_MB = 20
TEST_ROWS = 39_400_000
TIME_LIMIT_MIN = 60
# The scorer image (requirements.txt shown in the submission guide): numpy>=1.26,<3, onnxruntime>=1.18,
# numba==0.67.0 (+ its llvmlite), torch, jax, tensorflow; pyarrow comes with the starter pack.
KNOWN_PACKAGES = {"numpy", "onnxruntime", "pyarrow", "numba", "llvmlite", "torch", "jax", "jaxlib", "tensorflow"}
MAYBE_PACKAGES = {"pandas", "scipy", "sklearn", "onnx", "polars"}   # often present, not guaranteed

RESULTS: list[tuple[str, str]] = []


def report(level: str, msg: str) -> None:
    RESULTS.append((level, msg))
    print(f"[{level}] {msg}", flush=True)


# --------------------------------------------------------------------------- 1. archive
def check_archive(zpath: Path, workdir: Path) -> Path | None:
    try:
        z = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile as e:
        report("FAIL", f"not a valid zip: {e}")
        return None
    bad = z.testzip()
    if bad:
        report("FAIL", f"corrupted member in zip: {bad}")
        return None
    size_mb = zpath.stat().st_size / 1e6
    report("OK" if size_mb <= LIMIT_MB else "FAIL", f"zip size {size_mb:.2f} MB (limit {LIMIT_MB} MB)")
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    junk = [n for n in names if n.startswith("__MACOSX/") or n.endswith(".DS_Store")]
    names = [n for n in names if n not in junk]
    print("    files: " + ", ".join(f"{n} ({z.getinfo(n).file_size / 1e3:.1f} kB)" for n in names))
    if junk:
        report("WARN", f"macOS junk in the zip (harmless, but better zip without it): {len(junk)} entries")
    if "solution.py" not in names:
        nested = [n for n in names if n.endswith("/solution.py")]
        hint = f" - found {nested[0]}: zip the files themselves, not their folder (cd into it, then zip -r ../sub.zip .)" if nested else ""
        report("FAIL", "solution.py is not at the root of the zip" + hint)
        return None
    report("OK", "solution.py at the root")
    if "utils.py" in names:
        report("WARN", "the zip contains its own utils.py - it may shadow the platform's utils (DataPoint)")
    sol_dir = workdir / "submission"
    sol_dir.mkdir()
    for n in names:
        target = (sol_dir / n).resolve()
        if not str(target).startswith(str(sol_dir.resolve())):
            report("FAIL", f"unsafe path in zip: {n}")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return sol_dir


# --------------------------------------------------------------------------- 2. code
def check_code(sol_dir: Path) -> None:
    src = (sol_dir / "solution.py").read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, feature_version=(3, 11))
    except SyntaxError as e:
        report("FAIL", f"solution.py does not parse as Python 3.11: {e}")
        return
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PredictionModel"]
    if not cls:
        report("FAIL", "no top-level class PredictionModel in solution.py")
    else:
        methods = {n.name for n in cls[0].body if isinstance(n, ast.FunctionDef)}
        report("OK" if "predict" in methods else "FAIL", "PredictionModel.predict " + ("found" if "predict" in methods else "missing"))
    local = {p.stem for p in sol_dir.rglob("*.py")} | {p.name for p in sol_dir.iterdir() if p.is_dir()}
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    third = sorted(m for m in mods if m not in stdlib and m not in local and m != "utils")
    unknown = [m for m in third if m not in KNOWN_PACKAGES]
    print("    third-party imports: " + (", ".join(third) or "none"))
    for m in unknown:
        lvl = "WARN"
        note = "commonly installed, but not in the scorer requirements" if m in MAYBE_PACKAGES else "NOT in the scorer requirements"
        report(lvl, f"import {m}: {note} - make sure the scorer image has it (no internet there)")
    for pat in ("/content/", "/Users/", "C:\\", "/home/", "drive/MyDrive"):
        if pat in src:
            report("WARN", f"hard-coded path containing {pat!r} in solution.py - use Path(__file__).parent instead")
    if ".cuda(" in src or "'cuda'" in src or '"cuda"' in src:
        report("WARN", "solution.py mentions CUDA - the scorer has no GPU")
    if "torch.load(" in src and "weights_only" not in src:
        report("WARN", "torch.load without weights_only=...: torch>=2.6 defaults to weights_only=True and refuses numpy objects")


# --------------------------------------------------------------------------- 3. onnx
def check_onnx(sol_dir: Path) -> None:
    files = sorted(sol_dir.rglob("*.onnx"))
    if not files:
        return
    try:
        import onnx
    except ImportError:
        report("WARN", "pip install onnx to check ONNX files for missing external weights (the run test below still catches it)")
        return
    for f in files:
        try:
            m = onnx.load(str(f), load_external_data=False)
        except Exception as e:  # noqa: BLE001
            report("FAIL", f"{f.name}: cannot be parsed as ONNX: {e}")
            continue
        ext = {}
        for t in m.graph.initializer:
            if t.data_location == onnx.TensorProto.EXTERNAL:
                loc = {e.key: e.value for e in t.external_data}.get("location", "?")
                ext.setdefault(loc, 0)
                ext[loc] += 1
        print(f"    {f.name}: ir_version {m.ir_version}, opset {[o.version for o in m.opset_import]}, "
              f"{len(m.graph.node)} nodes, inputs {[i.name for i in m.graph.input]}")
        if not ext:
            report("OK", f"{f.name}: all weights are inside the file")
        for loc, cnt in ext.items():
            present = (f.parent / loc).exists()
            report("OK" if present else "FAIL",
                   f"{f.name}: {cnt} weight tensors live in a separate file '{loc}' - "
                   + ("included" if present else "MISSING from the zip (embed them: python embed_onnx_weights.py "
                      f"{f.name}, with '{loc}' next to it)"))


# --------------------------------------------------------------------------- 4-5. run
RUNNER = textwrap.dedent(r'''
    import dataclasses, importlib.util, json, math, sys, time, traceback
    import numpy as np

    sol_dir, data, n_seqs, n_rows, ref_dir, out_path = sys.argv[1:7]
    n_seqs, n_rows = int(n_seqs), int(n_rows)
    res = {}

    @dataclasses.dataclass
    class DataPoint:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

    def load(folder, tag):
        sys.path.insert(0, folder)
        try:
            spec = importlib.util.spec_from_file_location("solution_" + tag, folder + "/solution.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.PredictionModel
        finally:
            sys.path.remove(folder)

    def sequences():
        if data:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(data)
            names = pf.schema_arrow.names
            feats = [c for c in names if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
            out = []
            for g in range(min(n_seqs, pf.num_row_groups)):
                t = pf.read_row_group(g)
                X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in feats]).astype(np.float32)[:n_rows]
                Y = np.column_stack([t.column(c).to_numpy() for c in ("t0", "t1")]).astype(np.float32)[:n_rows] if "t0" in names else None
                M = t.column("is_scored").to_numpy(zero_copy_only=False)[:n_rows] if "is_scored" in names else None
                out.append((X, Y, M))
            return out
        rng = np.random.default_rng(0)
        return [(rng.standard_normal((n_rows, 112)).astype(np.float32), None, None) for _ in range(n_seqs)]

    def run(model, seqs, ids):
        preds, problems, t_total, rows = [], [], 0.0, 0
        for (X, _, _), sid in zip(seqs, ids):
            P = np.zeros((len(X), 2), np.float32)
            for i in range(len(X)):
                dp = DataPoint(seq_ix=sid, step_in_seq=i, need_prediction=i >= 99, state=X[i].copy())
                t = time.perf_counter()
                o = model.predict(dp)
                t_total += time.perf_counter() - t
                rows += 1
                # the same rules as the official utils.ScorerStepByStep, which raises on any of them
                if i < 99:
                    if o is not None:
                        problems.append(f"seq {sid} step {i}: returned a value on a warm-up row - the official "
                                        f"scorer raises 'return None during warm-up AFTER updating state'")
                    continue
                if o is None:
                    problems.append(f"seq {sid} step {i}: returned None on a required row"); continue
                a = np.asarray(o, dtype=np.float32)
                if a.shape != (2,):
                    problems.append(f"seq {sid} step {i}: output shape {a.shape}, the official scorer needs exactly (2,)"); continue
                if not np.all(np.isfinite(a)):
                    problems.append(f"seq {sid} step {i}: non-finite output {a.tolist()}"); continue
                P[i] = a
            preds.append(P)
        return preds, problems, t_total / max(rows, 1) * 1e6

    try:
        PM = load(sol_dir, "sub")
        t = time.perf_counter()
        m1 = PM()
        res["init_seconds"] = round(time.perf_counter() - t, 2)
    except Exception:
        res["init_error"] = traceback.format_exc(limit=3)[-1500:]
        json.dump(res, open(out_path, "w")); sys.exit(0)

    seqs = sequences()
    ids = [1000 + k for k in range(len(seqs))]
    try:
        p1, problems, us = run(m1, seqs, ids)
        res["problems"] = problems[:8]
        res["n_problems"] = len(problems)
        res["us_per_row"] = round(us, 1)
        p2, _, _ = run(PM(), seqs, ids)
        res["determinism_max_diff"] = float(max(np.abs(a - b).max() for a, b in zip(p1, p2)))
        if len(seqs) >= 2:
            # one instance: seq B then seq A under a new id; A must equal A from a fresh model
            p3, _, _ = run(PM(), [seqs[1], seqs[0]], [5000, 5001])
            res["reset_max_diff"] = float(np.abs(p3[1] - p1[0]).max())
        allp = np.concatenate([p[99:] for p in p1])
        res["pred_std"] = [round(float(v), 4) for v in allp.std(0)]
        res["frac_abs_gt_2"] = float((np.abs(allp) > 2).mean().round(4))
        Ys = [(s[1], s[2], p) for s, p in zip(seqs, p1) if s[1] is not None and s[2] is not None]
        if Ys:
            y = np.concatenate([Y[99:][M[99:]] for Y, M, _ in Ys]); pp = np.concatenate([P[99:][M[99:]] for _, M, P in Ys])
            y = np.clip(y.astype(np.float64), -2, 2); pp = np.clip(pp.astype(np.float64), -2, 2)
            sc = []
            for k in range(2):
                w = np.abs(y[:, k]); sw = w.sum()
                my, mp = (w * y[:, k]).sum() / sw, (w * pp[:, k]).sum() / sw
                cov = (w * (y[:, k] - my) * (pp[:, k] - mp)).sum() / sw
                vy, vp = (w * (y[:, k] - my) ** 2).sum() / sw, (w * (pp[:, k] - mp) ** 2).sum() / sw
                sc.append(float(cov / math.sqrt(vy * vp)) if vy > 1e-16 and vp > 1e-16 else 0.0)
            res["wp_mask_small_sample"] = [round(s, 4) for s in sc]
    except Exception:
        res["run_error"] = traceback.format_exc(limit=4)[-2000:]
    if ref_dir:
        try:
            _, _, us_ref = run(load(ref_dir, "ref")(), seqs[:1], [9000])
            _, _, us_sub = run(PM(), seqs[:1], [9001])
            res["us_ref"], res["us_sub_same_rows"] = round(us_ref, 1), round(us_sub, 1)
        except Exception:
            res["ref_error"] = traceback.format_exc(limit=2)[-800:]
    json.dump(res, open(out_path, "w"))
''')


def run_solution(sol_dir: Path, workdir: Path, args) -> None:
    runner = workdir / "runner.py"
    runner.write_text(RUNNER)
    out = workdir / "run_result.json"
    cwd = workdir / "elsewhere"          # the platform's working dir is not the solution folder
    cwd.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    data = str(Path(args.data).resolve()) if args.data else ""
    ref = str(Path(args.ref_dir).resolve()) if args.ref_dir else ""
    print(f"    running {args.seqs} sequences x {args.rows} rows "
          f"({'real data: ' + Path(data).name if data else 'random N(0,1) rows'}) ...", flush=True)
    p = subprocess.run([sys.executable, "-I", str(runner), str(sol_dir), data, str(args.seqs), str(args.rows), ref, str(out)],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=args.timeout)
    if not out.exists():
        report("FAIL", "the test process crashed:\n" + (p.stderr or p.stdout)[-2000:])
        return
    r = json.loads(out.read_text())
    if "init_error" in r:
        report("FAIL", "PredictionModel() raised on a clean machine:\n" + r["init_error"])
        return
    report("OK", f"PredictionModel() created in {r['init_seconds']} s")
    if "run_error" in r:
        report("FAIL", "predict() raised:\n" + r["run_error"])
        return
    if r["n_problems"]:
        report("FAIL", f"{r['n_problems']} protocol problems, e.g.:\n      " + "\n      ".join(r["problems"]))
    else:
        report("OK", "protocol: None on warm-up, shape (2,) and finite values on every required row")
    report("OK" if r["determinism_max_diff"] == 0 else "FAIL",
           f"determinism: max diff between two fresh models = {r['determinism_max_diff']:.3g}")
    if "reset_max_diff" in r:
        report("OK" if r["reset_max_diff"] == 0 else "FAIL",
               f"state reset between sequences: max diff = {r['reset_max_diff']:.3g}")
    print(f"    predictions: std {r['pred_std']}, share with |pred| > 2: {r['frac_abs_gt_2']:.1%}")
    if args.data and r["frac_abs_gt_2"] > 0.02:          # random inputs say nothing about the real scale
        report("WARN", f"{r['frac_abs_gt_2']:.1%} of predictions exceed +-2 and get clipped by the metric - consider scaling the output")
    if "wp_mask_small_sample" in r:
        print(f"    WP on the mask (t0, t1) for these few sequences only (noisy): {r['wp_mask_small_sample']}")
    proj = r["us_per_row"] * TEST_ROWS / 1e6 / 60
    print(f"    speed here: {r['us_per_row']} us/row -> ~{proj:.0f} min for the test set on THIS machine")
    if "us_ref" in r:
        ratio = r["us_sub_same_rows"] / r["us_ref"]
        report("OK" if ratio <= 1.05 else "WARN",
               f"speed vs reference: {ratio:.2f}x ({r['us_sub_same_rows']} vs {r['us_ref']} us/row). "
               + ("Not slower than a submission known to pass." if ratio <= 1.05 else
                  "Slower than the reference: it passes only if the reference had that much headroom."))
    elif "ref_error" in r:
        report("WARN", "reference solution could not be run:\n" + r["ref_error"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--data", default=None, help="valid.parquet (or any parquet in competition format): real rows + small-sample WP")
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3000, help="rows per sequence to feed (>= 100)")
    ap.add_argument("--ref-dir", default=None, help="folder with a solution.py known to pass, for a speed ratio")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    zpath = Path(args.zip)
    print(f"== {zpath}")
    workdir = Path(tempfile.mkdtemp(prefix="subcheck_"))
    try:
        sol_dir = check_archive(zpath, workdir)
        if sol_dir is not None:
            check_code(sol_dir)
            check_onnx(sol_dir)
            run_solution(sol_dir, workdir, args)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    fails = [m for lvl, m in RESULTS if lvl == "FAIL"]
    warns = [m for lvl, m in RESULTS if lvl == "WARN"]
    print("\n" + ("RESULT: DO NOT SUBMIT - " + f"{len(fails)} problem(s) will break the run" if fails else
                  "RESULT: ready to submit" + (f" ({len(warns)} warning(s) above)" if warns else "")))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
