# wnn-connectome-forecast

Forecasting two-sided execution prices from 20,000-step sequences
(112 features, targets `t0, t1 ∈ [-2, 2]`). Metric: `|y|`-weighted
Pearson correlation.

> **Status:** work in progress. See [Status](#status) below.
> Team project, 4 people. Started 2025-11.

## Problem

- **Input:** sequences of 20,000 steps × 112 features (order book snapshots,
  trades, aggregates for two instruments).
- **Targets:** `t0`, `t1` — execution prices of two sides, clipped to `[-2, 2]`.
- **Metric:** **`|y|`-weighted Pearson correlation**, not MSE.

## Key findings (EDA)

- Targets live on a **discrete grid** with 4 possible units:
  `0.1254, 0.2413, 0.4685, 0.6569`. Each sequence belongs to one "family".
- **Family is predictable** from the first 99 steps with **99.9% accuracy**.
- **`k0 + k1 ∈ {-1, -2}`** dominates (90%+ rows) → `t0 ≈ -t1 - u`.
- **`is_scored` is predictable** from causal features: **AUC 0.862**.
- **92%** of weighted variance comes from rows with `|y| > 1`.

## Approach

- 2-layer GRU (h=128), 192k params, weighted MSE loss.
- Trained on 3,000 sequences, 8 epochs, OneCycle LR.
- ONNX export with hidden-state interface, **bit-equivalent** to PyTorch.
- Validated with family-mix reweighting + bootstrap over sequences.

## Results

| Model | all rows | is_scored |
|---|---|---|
| baseline ONNX | 0.404 | 0.572 |
| GRU 2×128, 600 seq | 0.467 | 0.653 |
| **GRU 2×128, 3000 seq, λ=2** | **0.492** | **0.654** |
| h=192 | 0.484 | 0.661 |

Per-family: `0.56 / 0.48 / 0.34 / 0.24` — family 4 is the bottleneck.

## Status

**Working (validated):**
- [x] EDA: families, grid, `k0+k1`, `is_scored` AUC — `notebooks/01`
- [x] Baseline GRU 2×128, 3000 seq — `notebooks/03`
- [x] ONNX export + equivalence check — `notebooks/03`
- [x] Family-mix validation + bootstrap — `notebooks/04`

**In progress (assigned):**
- [ ] Restore `t1` from `t0` via `k0+k1 ∈ {-1,-2}` — @vera
- [ ] Multi-task: event/calm head — @you
- [ ] `is_scored` auxiliary head — @dima

**Explored, negative result (documented, not pursued):**
- Dropout, channel-dropout, diff-features, family oracle,
  calibration, ensembles, α>0.5 weighted sampling
  → all within noise or worse. See `REPORT.md` §4.

**Open questions (not started):**
- TCN / Transformer / Mamba vs GRU
- 10k+ sequences
- Focus loss for rare pairs in family 4

## Team

- **@you** — GRU training, EMA/SWA, ONNX export, diagnostics
- **@vera** — EDA, family analysis, `k0+k1` structure
- **@dima** — Data pipeline, speed benchmarks, CI
- **@boris** — Metric, bootstrap CI, evaluation harness

## Reproduce

```bash
pip install -r requirements.txt
python data/download.py          # ~30 GB, ~15 min
python src/train.py --seq 3000 --lam 2
python src/export_onnx.py --check
