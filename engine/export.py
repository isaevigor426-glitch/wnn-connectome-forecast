"""Checkpoint -> submission folder + zip (solution.py + weights.npz), with self-checks.

1. Writes weights.npz (float32) and copies solution_template.py as solution.py.
2. Equivalence: runs torch (whole sequence) and the numpy solution (row by row) on
   --check-steps rows of a validation sequence; max abs diff must be < 1e-4.
3. Speed: microseconds per predict() call on 1 thread.
4. Zips to <out>.zip and checks the 20 MB limit.

    python export.py --ckpt runs/gru/best.pt --valid-dir /content/data/valid --out submissions/gru_v1
"""
from __future__ import annotations

import argparse
import json
import dataclasses
import importlib.util
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from model import build_model

HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class DP:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def load_model(ckpt: str):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_model(st["cfg"])
    m.load_state_dict(st["model"])
    m.eval()
    return m, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="submission folder to create")
    ap.add_argument("--valid-dir", default=None, help="for the equivalence check (else random input)")
    ap.add_argument("--check-steps", type=int, default=3000)
    ap.add_argument("--scale", type=float, nargs=2, default=None,
                    help="override output scale for t0 t1 (default: best scale from validation)")
    args = ap.parse_args()

    model, st = load_model(args.ckpt)
    scale = np.array(args.scale if args.scale else st.get("output_scale", [1.0, 1.0]), np.float32)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    files = ["solution.py", "weights.npz"]
    if model.cfg.get("arch") == "baseline":
        w["target_mean"] = np.zeros(2, np.float32)
        w["target_std"] = np.ones(2, np.float32)
        (out / "config.json").write_text(json.dumps({"output_scale": scale.tolist()}, indent=1))
        shutil.copy(HERE / "solution_template_baseline.py", out / "solution.py")
        files.append("config.json")
    elif model.cfg.get("arch") == "mamba":
        c = model.cfg
        d_inner = c["expand"] * c["d_model"]
        w.update(n_layers=np.array(c["n_layers"]), d_model=np.array(c["d_model"]), d_inner=np.array(d_inner),
                 d_state=np.array(c["d_state"]), n_heads=np.array(d_inner // c["headdim"]),
                 headdim=np.array(c["headdim"]), d_conv=np.array(c["d_conv"]),
                 out_norm=np.array(int(c.get("out_norm", False))), output_scale=scale)
        shutil.copy(HERE / "solution_template_mamba.py", out / "solution.py")
    else:
        w["layers"] = np.array(model.cfg["layers"])
        w["hidden"] = np.array(model.cfg["hidden"])
        w["output_scale"] = scale
        shutil.copy(HERE / "solution_template.py", out / "solution.py")
    np.savez(out / "weights.npz", **w)
    print(f"epoch {st.get('epoch')}, best valid {st.get('best_score')}, output_scale {scale.tolist()}")

    # --- equivalence check
    if args.valid_dir:
        X = np.asarray(np.load(Path(args.valid_dir) / "features.npy", mmap_mode="r")[0, :args.check_steps], np.float32)
    else:
        X = np.random.default_rng(0).standard_normal((args.check_steps, 112)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(X.copy())[None])[0][0].numpy() * scale
    spec = importlib.util.spec_from_file_location("sub_solution", out / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pm = mod.PredictionModel()
    got = np.zeros_like(ref)
    t = time.perf_counter()
    for i in range(len(X)):
        r = pm.predict(DP(7, i, i >= 99, X[i]))
        if r is not None:
            got[i] = r
    us = (time.perf_counter() - t) / len(X) * 1e6
    diff = float(np.abs(got[99:] - ref[99:]).max())
    print(f"numpy vs torch max |diff| = {diff:.2e}  ({'OK' if diff < 1e-4 else 'MISMATCH!'})")
    print(f"speed: {us:.1f} us/row -> ~{us * 39.4e6 / 1e6 / 60:.0f} min for the test set on this CPU")

    # --- zip
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(out / f, f)
    mb = zp.stat().st_size / 1e6
    print(f"wrote {zp} ({mb:.2f} MB, limit 20 MB)")
    if diff >= 1e-4:
        raise SystemExit("numpy inference does not match torch — do not submit")


if __name__ == "__main__":
    main()
