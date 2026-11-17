# engine/ — numba inference engine, submission tooling and the GRU pipeline

All final submissions of the team ran inside this inference engine. This folder also holds the
pipeline that trained two members of the ensembles — GRU A, the organizers' baseline GRU fine-tuned on
all data, and GRU B, a GRU with difference inputs and an MLP head (see the main README) — and the tools
used to build, check and time every submission. The scripts import each other, so run them from this folder or as
`python engine/<script>.py`.

## Inference engine

`solution_template_ensemble_numba.py` is the `solution.py` of an ensemble submission;
`build_ensemble.py` writes it next to `weights.npz`.

- **One compiled call per row** runs every model and the blend `out = const + Σ coef_k · p_k`
  (per-model centring, blend weights and the final scale are folded into `coef` / `const`).
- **Matrix-vector products** are "axpy" loops over transposed weights, 8 input rows per pass: the inner
  loop runs over contiguous memory and is vectorised (AVX2 / AVX-512), no BLAS calls.
- **Gates without libm.** sigmoid = 1 / (1 + exp(−x)) with a Cephes-style exp built from the float's
  exponent bits (a small LLVM intrinsic), tanh = rational approximation, `error_model="numpy"` (Python's
  zero-division checks block vectorisation), fast-math limited to FMA contraction. The sigmoid must
  saturate to exactly 1.0 like float32 torch: an approximation that stopped at 1 − 1e-7 made long-memory
  gates leak (drift 3e-3 after 3 000 steps); this one reproduces torch to ~1e-5.
- **Float16 weight matrices**, float32 vectors and arithmetic. Two 128-unit models take 0.8 MB and stay in
  the 1 MB L2 cache of the scoring CPU (AMD EPYC 9224). Measured: once the weights do not fit in L2, the
  time per model grows ~1.5×. Rounding to float16 changes the validation score by < 1e-5.
- **Self-test on load:** a few rows with outputs computed by torch at build time must be reproduced;
  otherwise the matrices are widened to float32 storage (same values) and the test is repeated; if that
  fails too, `__init__` raises — the submission fails at once instead of after an hour.

Speed, relative to our first accepted submission (numpy GRU 2×128), same machine:

| submission | µs/row | relative |
|---|---|---|
| numpy GRU 2×128 (accepted, reference) | ~61 | 1.00 |
| organizers' baseline (ONNX Runtime) | 33 | 0.53 |
| 2 models in one ONNX graph | 92 | 1.50 — too slow for the limit |
| 2 models, numba, float32 weights | 38–43 | 0.62–0.69 |
| 2 models, numba, float16 weights | 32 | 0.52 |
| 3 models of 128 units, numba, float16 | 48 | 0.80 (passed on the platform) |
| GRU 192 alone, numba, float16 | 32 | 0.51 |
| baseline + GRU 256, numba, float16 | 70 | 1.13 |

## Building and checking a submission

```bash
python prepare_data.py --src valid_sub.parquet --out data/valid           # .npy for the validation set
python build_ensemble.py --valid-dir data/valid --out submissions/ens \
    --model "ft=runs/ft2/epoch_0.pt@0.5" \
    --model "gru=runs/gru_full/best.pt@0.2" \
    --model "reg=runs/reg128m/best.pt@0.3"
python check_submission.py submissions/ens.zip --data valid_sub.parquet --seqs 3 --rows 20000 \
    --ref-dir submissions/accepted_gru
python bench_speed.py submissions/accepted_gru submissions/ens --rows 5000 --repeats 5
```

- Model spec `NAME=PATH[+PATH...][@WEIGHT]`: checkpoints of this folder's `train.py` (arch `baseline` or
  `gru`) or the organizers' `baseline.onnx`; several checkpoints of one run joined with `+` are
  weight-averaged. The LSTM (and `wnn` GRU) members of the final ensemble were added to the engine as new
  model types in the final submission; that version of `solution.py` is the reference for them.
- The blend is defined as in `ensemble_eval.py`: each model centred and scaled on the scored validation
  rows, weighted, rescaled to std 0.6 (the metric clips at ±2), so the validation score carries over.
- `build_ensemble.py` scores the blend with float16-rounded matrices, writes `solution.py` + `weights.npz`,
  runs the written solution row by row against torch (refuses to write the zip above 1e-3) and zips it.
  `--engine onnx` builds the slower one-graph ONNX variant (`solution_template_ensemble.py`).
- `check_submission.py`: archive layout and size, imports against the scorer image, a run in a clean
  process with one thread, the official protocol (None during warm-up, finite `(2,)` arrays), determinism,
  state reset between sequences, speed against a submission known to pass.
- `embed_onnx_weights.py` moves `model.onnx.data` into the `.onnx` file: the new PyTorch exporter
  (`dynamo=True`) writes weights to a separate file by default, and a submission without it fails.

## Training pipeline

- `prepare_data.py` — parquet → `.npy` with checks (20 000 rows per sequence, steps in order, finite values);
  `make_subset.py` — random whole sequences into a smaller parquet (e.g. `valid_sub`, 600 sequences).
- `train.py` — one model:
  - `--arch baseline` — the organizers' architecture (GRU 112→128→128 → Linear), `--init-onnx baseline.onnx`
    to fine-tune their weights; `--arch gru` — 112 features and their one-step differences, standardised
    and clipped, Linear+SiLU → GRU 2×128 → MLP head on [GRU output, input projection];
    `--arch mamba` — Mamba-2 (chunked SSD scan in plain PyTorch);
  - loss 1 − |y|-weighted Pearson per chunk + 0.1 · weighted MSE; truncated BPTT over the full 20 000 steps
    in chunks of 1 000 with the state carried over;
  - reads `train.parquet` on the fly with several threads (`--train-parquet`); `--extra-parquet valid.parquet
    --exclude-seqs-from valid_sub.parquet` adds the validation sequences that are not used for evaluation;
  - `--init-from a.pt+b.pt` continues from (averaged) checkpoints; `--dropout`, `--input-noise`, `--wd`
    (on matrices only); `--keep-epochs`, `--resume`.
- `ensemble_eval.py` — all models on one validation set: single scores, correlations, every equal-weight
  blend, weights chosen on one half of the sequences and scored on the other (`--no-cv` skips that part).
- `export.py` + `solution_template*.py` — single-model numpy submissions (our GRU, baseline, Mamba).
- `eda.py`, `linear_probe.py` — first look at the data; `local_eval.py` replays a parquet through a
  solution as the scorer does and cross-checks the official metric; `docker/` — a scorer-like image
  (Python 3.11, 1 CPU, 16 GB RAM, no network).

## Results

Public scores of the submissions built with this pipeline (the team's later ensembles are in the main README):

| submission | public score |
|---|---|
| organizers' baseline × 0.3 | 0.65045 |
| our GRU, 3 500 sequences | 0.64368 |
| Mamba-2 | 0.643 |
| baseline fine-tuned on all of train | 0.65083 |
| fine-tuned baseline (weight average of epochs 2–5) + our GRU on all of train — numba engine | 0.65426 |
| + our GRU with dropout 0.2, input noise 0.05, weight decay 1e-2 (3 models) | 0.65512 |
| members trained also on the rest of `valid`, blend weights 0.5 / 0.2 / 0.3 | 0.65612 |

Validation on 600 sequences of `valid` (`valid_sub`, |y|-weighted Pearson on the scored rows):

| model | valid_sub |
|---|---|
| fine-tuned baseline, average of epochs 2–5 (`ft_swa`) | 0.6668 |
| … + 1 epoch on train + rest of valid (`ft2`) | 0.6674 |
| our GRU on all of train (`gru_full`) | 0.6637 |
| … with dropout / noise / weight decay (`reg128`) | 0.6654 |
| … also trained on the rest of valid (`reg128m`) | 0.6672 |
| blend of `ft2`, `gru_full`, `reg128m` (0.5 / 0.2 / 0.3) | 0.6715 |

The public score was consistently ~0.015 below `valid_sub`, and gains carried over about one to one.

## What we learned

- Dropout, input noise and weight decay on matrices helped GRUs trained from scratch (+0.0017 on
  validation, no drop at the end of training any more) but hurt fine-tuning the organizers' weights (−0.002).
- A wider GRU (192 units) was not better than 128, with or without regularisation.
- The 1 273 validation sequences not used for evaluation (+12 % data) gave +0.0017 for a GRU trained from
  scratch and +0.0006 for the continued fine-tune; the gain held on the leaderboard.
- Weight averaging of epochs gave the best single model for the fine-tune, nothing for GRUs from scratch.
- Blends gain from different families (fine-tuned baseline vs our GRU, correlation 0.96); more GRUs of the
  same kind added ~0.0001.
- In Colab, `!cmd $OUT.zip` is not expanded (IPython reads `OUT.zip` as an attribute) — write `{OUT}.zip`.

## Tests

```bash
pip install torch numba==0.67.0 onnx onnxruntime pyarrow pandas pytest
BASELINE_ONNX=path/to/baseline.onnx pytest -q engine/tests    # 20 tests, ~6 min on a CPU
```

Without `BASELINE_ONNX` the two fine-tuning tests are skipped. Covered: loss = metric; chunked = full
sequence (GRU, Mamba); scan = step recurrence; train → resume → export → harness for every architecture;
parquet streaming; ensemble builds (numba float16 / float32, ONNX) reproduce torch and pass the harness;
regularisation acts only in training; extra training sequences and `--init-from`.

## Notebooks

The Colab notebooks of this pipeline are in [`notebooks/Andrew/`](../notebooks/Andrew/) (comments in
Russian). They copy this folder from Google Drive (`wunder/day2`) to `/content/day2`. `day9`–`day11` are
saved with their outputs (training logs, blend selection, checks of the built submissions).
