# Max — analysis and LSTM experiments

These are the saved Colab notebooks and the supporting source files used for Max's
experiments. Notebook outputs are preserved as research history; the notebooks
contain diagnostic cells and alternative attempts and should not be run blindly
from top to bottom against an existing training run.

| Notebook | Contents |
|---|---|
| [analysis.ipynb](analysis.ipynb) | Exploratory analysis: target distributions, event timing, feature relationships, PCA and sequence families |
| [LSTM_Colab_T4.ipynb](LSTM_Colab_T4.ipynb) | LSTM 2×128 training on Colab T4, Drive checkpoints/resume, valid300 comparisons, float32 export diagnostics and trial ensembles |

Only these two distinct notebooks were found in the supplied files. The duplicated
Drive-download folder is not a third experiment.

## Supporting files

- `src/`: PyTorch models, TBPTT trainer, checkpoint handling, offline metric,
  event features, weight exporter and a derived type-3 LSTM Numba engine.
- `scripts/`: resumable data acquisition, preparation, validation, export,
  export verification, benchmarking and guarded submission packaging.
- `tests/`: synthetic correctness tests and historical development reports.
- `reference/`: original team GRU engines and metric code, retained unchanged
  for comparison; these are team references, not newly authored models.
- `ASSET_MANIFEST.json`: metadata for external weights and validation arrays.

The current checkout includes the LSTM export correction: default LSTM matrices
remain float32 rather than being rounded to float16. Existing checkpoints remain
compatible. GRU defaults and explicit half-precision exports remain supported.

## Recorded local validation

The saved LSTM notebook reports the following **offline valid300 family-mix
scores**, not public leaderboard scores:

| Candidate | Score |
|---|---:|
| honest3: mean of baseline × 0.3, gru6573 and corr3k | 0.660993 |
| LSTM | 0.661758 |
| honest3 + LSTM, equal model weights | 0.663036 |
| Gain over honest3 | +0.002043 |

The saved quality report has `quality_pass: true` and `candidate_exact: false`.
This measures the benefit of adding LSTM to saved honest predictions; it does not
certify a particular exported ensemble. Gains below about 0.003 on this holdout
are treated cautiously. The historical float16 export verification failed
(`max_error = 0.015285670757293701`, threshold < 0.001). The notebook retains the
subsequent float32 diagnostics. Its trial packed public-ensemble + LSTM benchmark
also failed the runtime gate; it is not the team's final five-model submission.
Refer to individual saved outputs instead of applying a pass flag to all exports.

## Use in Colab

1. Copy this directory's contents to `MyDrive/alfa_codex_handoff` (so `src/` and
   `scripts/` are directly inside it), then open the desired notebook in Colab.
2. Set `PROJECT`, `RAW`, `DATA`, `OUT` and `ASSETS` to your actual paths. For a new
   session, prepared data must either persist on Drive or be prepared again.
3. Supply the official dataset and external validation arrays separately.
   Do not put datasets, checkpoints, predictions or packed weights in Git.
4. For an already trained model, skip data acquisition/training and use the
   required export/verification cells only. Preserve old artifacts with new names.

Detailed CLI instructions are in [scripts/README.md](scripts/README.md). That file
and the local test reports record the initial development stage; the saved Colab
notebook includes later training and diagnostics. `make_notebook.py` generates the
initial template, not the later saved experiment history. Paths in those tools
assume the original package layout; do not regenerate over the saved notebook.

A separate reproducible source-only check can be run from this directory:

```bash
python -m pytest tests/test_training.py tests/test_metrics.py tests/test_acquire.py -q
```

These tests need PyTorch, NumPy, Numba, PyArrow, tqdm and pytest. They do not train
on competition data. Full training and data acquisition are manual Colab tasks.
