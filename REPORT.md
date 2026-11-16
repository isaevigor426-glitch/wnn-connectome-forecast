# Technical report

Wunder Fund **Alpha Connectome** challenge, November 2026. Team of four; final public score **0.6571**
(baseline 0.5935, first place ≈ 0.676).

## 1. Task

- **Data.** `train` 10 607 and `valid` 1 873 sequences of 20 000 steps; 112 anonymised features per step:
  22 "price", 22 volume and 8 trade columns for each of two instruments, plus 8 hidden columns `a0–a7`.
  All columns are quantile-transformed to a normal shape (|x| ≤ 5.1993 = Φ⁻¹(10⁻⁷)).
- **Targets.** `t0`, `t1`, indicators of the future price move of instrument 0, clipped to [−2, 2].
- **Metric.** |y|-weighted Pearson correlation over the scored rows (`is_scored`, ≈ 9 %) of all sequences
  at once; mean over the two targets.
- **Inference.** One Python call per row, state reset on a new sequence, 1 CPU core, 60 minutes for
  ≈ 39 M rows, deterministic.

## 2. Loss that matches the metric

For a fixed input, the minimiser of `E[|y|·(p − y)²]` is `E[|y|·y | x] / E[|y| | x]`, which maximises the
|y|-weighted correlation. Switching from MSE to this loss was the largest single gain (+0.046 over the
baseline). The organisers' baseline also produced outputs that were too large and were cut by the ±2 clip;
scaling it by 0.3 alone moved it from 0.594 to 0.650 public — a reminder that the clip, not the scale, is
what a correlation metric sees.

## 3. Structure of the target

- **Families.** Every value is `k·u`, `u ∈ {0.1254, 0.2413, 0.4685, 0.6569}` (F0–F3), one unit per sequence;
  k ranges are cut by the ±2 clip, giving 63 distinct values. The family is predictable from the first 99
  steps with 99.9 % accuracy, so the network learns it implicitly (an oracle one-hot input added nothing).
- **Sum and difference.** At least one of `t0`, `t1` is always negative; `S = t0 + t1` is almost always −u or
  −2u in F2/F3. A natural reading (a hypothesis): `t0`/`t1` are outcomes of buying/selling now and unwinding
  later; `S` is the spread cost, `D = t0 − t1` twice the move.
- **Event time.** The target changes only on steps where the price columns of instrument 0 change, and all 22
  change together. The autocorrelation of increments of `D` in event time has a −0.5 peak at lag 2–3 events,
  as expected for "sum of the next N moves". These increments are visible in the features (corr 0.94).
- **Where the model loses.** F3 (largest unit, ≈ half of the metric weight) scores ≈ 0.40 against 0.72–0.78
  for F0–F2.

## 4. Validation

- A 300-sequence hold-out from `valid` has a seed-to-seed spread of ≈ 0.002–0.003. Family-mix re-weighting
  and a bootstrap over sequences made comparisons tighter, but many 0.001–0.002 "gains" did not survive on
  the leaderboard.
- **Distribution shift.** A gradient-boosting classifier tells `train` from `valid` sequences with AUC 0.995
  inside every family, mostly by volume levels. A model trained without `valid` was 0.0056 worse publicly,
  while the hold-out (taken from the same slice as the training part of `valid`) showed only 0.001.
  Conclusion: under a shift, test recipes as **train → valid**, then retrain the winner on everything.

## 5. What worked late

All measured as train → valid (3 000 train sequences, 3 seeds of the baseline for the noise level):

| Recipe | is_scored | all rows |
|---|---|---|
| baseline (mean of 3 seeds; spread 0.0017 / 0.0013) | 0.6666 | 0.4916 |
| family re-weighting to the `valid` mix | 0.6681 | 0.4937 |
| 3 000 **most active** sequences (two seeds) | 0.6687 / 0.6703 | 0.4983 / 0.4985 |
| most active + family re-weighting | 0.6707 | 0.4998 |
| sequences matched to the activity distribution of `valid` | 0.6663 | 0.4958 |

`valid` is more active than `train` (share of steps with a price event 0.37 vs 0.28, strongest in F3).
Selecting the active half of `train` helped more than matching `valid`'s distribution: quiet sequences sit
near their mean target and carry little of the metric's variance.

**GRU vs LSTM** (same recipe): LSTM 0.6720 / 0.4948, GRU 0.6707 / 0.4998; GRU + 2 LSTM 0.6748 / 0.5026 was
the best mix. A teammate's LSTM trained on all data scored 0.6526 public alone — the best single model.

## 6. What did not work

Each was measured against a matched control; all stayed within noise or hurt.

- Architecture: GRU width 192/256 (+0.003 locally, nothing publicly, starts to memorise), Mamba, MLP before
  the GRU, two branches, longer BPTT, a 63-class head (−0.035), longer training (overfits after ~8 epochs).
- Inputs: differences, minus-EMA, replacing volumes by deviations (−0.014 in F3: volume level is signal),
  spread/OFI from the columns, activity windows, counters "steps since change", event "tape" of the last 8
  events, tick-space re-encoding.
- Training: correlation term in the loss (helped on 600 sequences, vanished on 3 000), |y|^β weights,
  is_scored as weight, dropout, EMA/SWA, auxiliary heads, distillation of the ensemble, fine-tuning on `valid`.
- Post-processing and stacking: per-family calibration, fitted blend weights (always worse than an equal
  average), power transforms, per-sequence adaptation (looked like +0.05 in-sample, −0.05 out of sample).
- Using the past target: a decoder recovers past targets from prices (corr 0.86 in F3) but adds nothing —
  everything it recovers is a function of prices the network already sees. An "ideal decoder" ceiling of
  +0.10 turned out to be leakage of the future through still-open target windows.

## 7. Engineering

- **Inference engine.** All models run inside one numba-compiled call per row: transposed weight matrices
  in float16 (fits in L2 cache), float32 arithmetic, vectorised gates with polynomial exp/tanh, a self-test
  on load. GRU 2×128 costs ≈ 33 µs per row.
- **LSTM support** was added as new model types: float32 weights (exact to 1e−7 on 20 000 steps) and a
  float16 variant. Locally float32 looked cheaper; on the scoring machine it took 57 of 60 minutes, while
  float16 gave identical scores (Δ < 1e−5) and ran in 43.5 minutes — a lesson in measuring on the target
  hardware.
- **Ensembling tooling.** Extracting members from a packed submission, adding GRU/LSTM members with equal
  weights after standardisation, and checking every build against PyTorch on a full 20 000-step sequence.
- **ONNX path** for single models: a step wrapper with explicit states, checked against PyTorch.

## 8. What I would do differently

- Build a trustworthy validation (train → valid, several seeds) before searching for improvements.
- Try other model classes early — LSTM turned out stronger than GRU, and this was found two days before
  the deadline.
- Measure inference time on the target hardware, not only locally.
