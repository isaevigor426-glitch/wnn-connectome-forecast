# WNN33 Competition Report: Predicting Future Price Movements

## 1. Problem Statement

The task was to predict two target variables, `t0` and `t1`, representing future price movements of a trading instrument. 

- **Data:** 20,000-step sequences with 112 anonymized features (order book, trades, and additional hidden features).
- **Targets:** `t0` and `t1`, clipped to the range `[-2.0, 2.0]`.
- **Prediction Requirement:** Values must be predicted starting from step 99 (warm-up is 0-98).
- **Constraints:** Strict inference time limit (60 minutes on a single CPU core for 39.4 million rows). This necessitated an optimized ONNX export.

## 2. Why Weighted Correlation ≠ MSE

The evaluation metric is a **Global Weighted Pearson Correlation**, where each row is weighted by `w = |y|`. 

**Key Insight from EDA:**
- 92% of the weighted variance comes from rows where `|y| > 1`. 
- Rows with `|y| ≈ 0` contribute almost nothing to the score but significantly distort standard MSE (Mean Squared Error).

**Solution:** 
Instead of standard MSE, I designed a custom loss function: `mean(|y| * (p - y)^2)`. This forces the model to focus on "events" (large values) rather than noise, directly optimizing for the metric.

## 3. Target Structure & EDA (Exploratory Data Analysis)

- **Discrete Grid:** Targets lie on a discrete grid with four possible step sizes (0.1254, 0.2413, 0.4685, 0.6569). Each sequence belongs to exactly one "family" (step size).
- **Predictability:** The family can be inferred from the first 99 steps with 99.9% accuracy (the model learns this automatically).
- **t0 / t1 Relationship:** There is a strict dominance rule: `k0 + k1 ∈ {-1, -2}`. Targets are nearly mirror images (e.g., `t0 ≈ -t1 - u`).
- **`is_scored` Nature:** The scoring mask is predictable from causal features with an AUC of 0.862, indicating it selects specific market events rather than random rows.

*(Note: Histograms and distribution plots can be found in the `results/` folder).*

## 4. What Didn't Work (Negative Results)

*This section demonstrates rigorous A/B testing and the ability to stop unproductive paths.*

- **Dropout & Channel Dropout:** No gain (within noise).
- **Diff-features (deltas):** Decreased score (0.454 vs 0.488).
- **Family Oracle (one-hot):** No help, as the model already infers the family from early steps.
- **Calibration:** Score unchanged.
- **Ensembles & EMA/SWA:** ±0.001 (noise).
- **Pair Classification (k0, k1) with weights:** Failed on rare classes (event recall ~0.01).

## 5. What Worked

- **More Data:** Scaling from 600 to 3000 sequences yielded +0.02.
- **Correlation Loss:** Adding `- λ * corr(y, p)` with λ=2 gave +0.004.
- **Optimal Architecture:** A 2-layer GRU (h=128) provided the best balance between quality and speed. Larger models (h=192) gave no gain but were 1.4x slower.

## 6. Diagnostics: Where the Model Fails

- **Family 4 (most frequent):** Score drops to 0.235 / 0.396.
- **Rare Events:** The model fails to catch classes `(-3, 1)` and `(1, -3)` (recall 0.011 and 0.006).
- **Side Prediction:** The model predicts which target is "lower" with 0.775 accuracy (ceiling 0.875).
- **Conclusion:** The model learns "calm" rows well but lacks features to predict rare events.

## 7. Engineering

- **ONNX Export:** A custom `Step` module was written to export with hidden states. Bitwise equivalence with PyTorch was verified.
- **Speed Optimization:** Profiling showed `np.nan_to_num` slowed inference by 47%. Replaced with `abs(x) < 1e6` check (+14% speed).
- **Determinism:** Verified that `predict()` produces identical results across repeated runs.

## 8. Limitations & Next Steps

- **Reconstruct t1 from t0** using the formula `k0 + k1 ∈ {-1, -2}`.
- **Multi-task Model:** Separate heads for "event/calm" classification and `is_scored`.
- **Focus loss** for rare classes in family 4.
- **More Data:** Scaling to 10k+ sequences.

---
*Report generated based on the WNN33 competition results. Final validation score: 0.492 / 0.654.*
