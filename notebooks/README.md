# Notebooks

Colab notebooks as they were run, one folder per contributor.

## [Isaev_Igor](Isaev_Igor/)

| notebook | topic |
|---|---|
| `00_setup_data_feature_importance` | data setup, feature-group importance |
| `01_eda_target_structure` | target structure: discrete families, event time |
| `02_first_gru` | first GRU |
| `03_loss_swa_ema` | loss variants, SWA and EMA |
| `04_features_losses_inference` | features, losses, inference |
| `05_postprocessing_event_detector` | post-processing, event detector |
| `06_F3_analysis_event_detector` | F3 analysis, event detector |
| `06_f3_oracle_distillation_ensembles` | F3 oracle, distillation, ensembles |
| `07_shift_and_active_selection` | train/valid shift, active-sequence selection |
| `08_final_models_lstm_ensemble` | final models, LSTM ensemble |

## [Andrew](Andrew/)

Runs of the [`engine/`](../engine/README.md) pipeline (comments in Russian); `day9`–`day11` are saved with outputs.

| notebook | topic |
|---|---|
| `day2_colab` | first GRU with difference inputs on a subset, export to a numpy submission |
| `day3_finetune`, `day3_finetune_v2` | fine-tuning the organizers' baseline on all of train (parquet streaming) |
| `day4_mamba` | Mamba-2 |
| `day5_ensemble_eval` | all models on one validation set, blend weights |
| `day6_ensemble_submit` | first ensemble submission with the numba engine |
| `day7_gru192` | wider GRU (192), weight averaging of epochs |
| `day8_regularization` | dropout, input noise, weight decay for the GRU |
| `day9_ft_reg` | regularised fine-tune of the baseline; stronger regularisation |
| `day10_more_data` | training also on the unused part of `valid` |
| `day11_final` | blend weights, final three-model submission |
| `day12_all_data` | the same members trained on all labeled data |

## [Max](Max/README.md)

Saved Colab research notebooks with their supporting LSTM pipeline and tests.

| notebook | topic |
|---|---|
| [`analysis`](Max/analysis.ipynb) | exploratory analysis of targets, event timing, features and sequence families |
| [`LSTM_Colab_T4`](Max/LSTM_Colab_T4.ipynb) | LSTM 2×128 training/resume, honest valid300 comparisons, float32 export diagnostics and trial ensembles |

See [Max's README](Max/README.md) for recorded scores, export limitations and
instructions for using the supporting `src/`, `scripts/`, `tests/` and team references.
