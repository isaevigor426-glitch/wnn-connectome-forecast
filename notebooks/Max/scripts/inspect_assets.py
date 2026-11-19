#!/usr/bin/env python3
"""Lightweight asset inventory. Does NOT print actual tensor values."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

if not ASSETS.exists():
    raise SystemExit("No assets/ directory. Run scripts/extract_assets.py with the original ZIP.")
for p in sorted(ASSETS.rglob("*")):
    if not p.is_file():
        continue
    if p.suffix == ".npy":
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        print(f"{p.relative_to(ROOT)}: shape={a.shape}, dtype={a.dtype}")
    elif p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            summary = ", ".join(f"{k}:{z[k].shape}/{z[k].dtype}" for k in z.files)
        print(f"{p.relative_to(ROOT)}: {summary}")
