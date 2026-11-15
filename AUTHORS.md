# Authors

Team of four, Wunder Fund Alpha Connectome challenge, November 2026.

### Igor Isaev ([@isaevigor426-glitch](https://github.com/isaevigor426-glitch)) — modelling, analysis, ensembling
- Analysis of the target: discrete families, event-time structure, sum/difference of `t0` and `t1`.
- The metric-matched loss `|y|·(p − y)²`, training pipeline and validation methodology.
- Detection of the train/valid shift and the active-sequence training recipe.
- GRU/LSTM training, LSTM support (float32 and float16) in the inference engine, ensemble assembly and checks.

### Andrew — inference engine, models
- The numba inference engine (one compiled call per row, float16 weights) used by all final submissions.
- Two ensemble members: a fine-tuned baseline GRU and a GRU with difference inputs and an MLP head.

### Max — LSTM
- The strongest single model of the team (LSTM, 0.6526 public) and its training recipe.

### Liza — experiments
- Systematic comparison of losses, output scaling and hand-crafted features.
