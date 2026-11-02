# Authors

Team project, started 2025-11.

## Core Contributors

### [@isaevigor426-glitch] — Lead ML Engineer / Researcher
*Role: End-to-end model development, target structure analysis, and inference optimization.*

- **Target Structure & EDA:**
  - Discovered that `t0` and `t1` live on a discrete grid with 4 families (`0.1254, 0.2413, 0.4685, 0.6569`).
  - Proved that sequence family is predictable from the first 99 steps with 99.9% accuracy.
  - Found the structural constraint `k0 + k1 ∈ {-1, -2}` (targets are near-mirrored), dominating 90%+ of rows.
  - Demonstrated that `is_scored` is highly predictable from causal features (AUC 0.862).
  - Showed that ~92% of the weighted variance comes from rows with `|y| > 1`.
- **Modeling & Training:**
  - Implemented 2-layer GRU (h=128, 192k params) with weighted MSE loss to directly optimize the `|y|`-weighted Pearson correlation metric.
  - Experimented with EMA, SWA, ensembles, and correlation-penalty loss (λ=2).
  - Achieved the best validation score: **0.492 / 0.654** (all rows / is_scored) vs baseline 0.404 / 0.572.
- **Diagnostics & Analysis:**
  - Performed deep family-level error analysis, identifying "Family 4" as the main bottleneck (0.235 / 0.396).
  - Proved that the model fails on rare events (recall ~0.01) and struggles with side prediction (t0 vs t1, accuracy 0.775 vs oracle 0.875).
- **Engineering & Optimization:**
  - Exported the PyTorch GRU to ONNX with a custom hidden-state wrapper (`Step`), verified bit-equivalence, and ensured deterministic inference.
  - Conducted rigorous speed benchmarks (baseline vs ONNX vs `nan_to_num` vs threshold) to find the optimal balance for the time limit (h=128 is the best speed/quality tradeoff).
  - Established the evaluation harness using sufficient statistics and bootstrap over sequences.

### [@teammate-2-username] — 

### [@teammate-3-username] — 

### [@teammate-4-username] —

---

## Report Structure (`REPORT.md`)

The technical report is divided into the following sections:

1. **Problem statement** — task definition, metrics, and constraints.
2. **Why weighted corr ≠ MSE** — math explanation and motivation for the loss function.
3. **Target structure (EDA)** — grid, families, `k0+k1` constraint, `is_scored` predictability.
4. **What didn't work** — negative results (dropout, diff-features, calibration, etc.).
5. **What worked** — final model, data scaling (600 → 3000), SWA, and λ=2.
6. **Diagnostics** — family 4 bottleneck, rare-event recall, side accuracy.
7. **Engineering (ONNX, speed)** — bit-equivalence, determinism, inference time.
8. **Limitations & next steps** — what was not tried (TCN, focus loss, multi-task).
