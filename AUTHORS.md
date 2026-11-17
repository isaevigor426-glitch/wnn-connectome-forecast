# Contributors

**Wunder Fund Alpha Connectome · November 2026 · 4-person team**

This project combines data analysis, recurrent modelling, validation research and single-core inference engineering. The research and final submission were collaborative; the sections below describe the areas of work recorded by the team rather than claiming that a leaderboard result belongs to a single person.

## Contributions by workstream

| Workstream | Contributions | Main contributor(s) |
|---|---|---|
| **Target analysis & diagnostics** | Discrete target families; target sum/difference; event-time structure; model-error breakdown | Igor Isaev |
| **Training & validation** | Weighted-MSE objective; sequence-aware PyTorch training; train/valid shift analysis; active-sequence selection | Igor Isaev |
| **Recurrent architecture experiments** | Standard and fine-tuned GRUs; difference-feature/MLP GRU; LSTM experiments | Igor Isaev, Andrew, Max |
| **Inference optimisation** | Custom Numba engine for recurrent steps; packed model parameters; LSTM and FP16 integration; numerical checks | Andrew, Igor Isaev |
| **Ensembling** | Exporting/reusing packed models; per-target normalisation; assembling and testing GRU/LSTM blends | Igor Isaev |
| **Loss, scaling & feature ablations** | Alternative objectives, output calibration and hand-crafted-feature comparisons | Liza |

## Individual contributions

### Igor Isaev · [@isaevigor426-glitch](https://github.com/isaevigor426-glitch)

- Investigated the anonymous targets: four discrete value families, event-clock behaviour, and the relationship between `t0` and `t1`.
- Developed sequence-aware GRU/LSTM training and validation experiments, including metric-aligned weighted loss, train/valid distribution-shift diagnostics and active-sequence selection.
- Trained recurrent models, added LSTM support to the packed inference path, assembled and numerically checked multiple ensembles.
- Maintained the experiment analysis and reproducible report/figures in the repository.

### Andrew

- Designed and implemented the high-performance **Numba CPU inference engine**, with packed weights and stateful recurrent evaluation, and the submission tooling around it: ensemble builder (float16 weights, self-test on load), pre-submit checker reproducing the scorer protocol, speed benchmark ([`engine/`](engine/)).
- Developed the fine-tuned baseline-shaped GRU and the GRU using input differences, a projection and an MLP output head, including regularised versions and versions trained also on the unused part of `valid`.
- Built the Colab training pipeline (parquet streaming, truncated BPTT over full sequences) and ran the Mamba-2 and wider-GRU experiments ([`notebooks/Andrew/`](notebooks/Andrew/)).
- Contributed models and runtime engineering used in the final ensemble.

### Max

- Developed and trained an LSTM approach, including the team's strongest recorded standalone public submission (**0.65260 WP**).
- Contributed LSTM modelling to the final ensemble experiments.

### Liza

- Explored alternative loss functions, prediction scaling and hand-crafted input features.
- Helped compare modelling ideas against the reference approach.

---

**About attribution.** The figures and technical report refer to members by *architecture and experimental recipe* (for example, “GRU with first differences + MLP”), not by contributor name. This makes the modelling choices independently understandable. See [technical report](REPORT.md) for the results and [README](README.md) for the final overview.
