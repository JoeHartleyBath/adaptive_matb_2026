# Design choice: Warm-start with weak L2 is the best LogReg personalisation strategy

**Decision**  
Warm-start with weak L2 regularisation (C = 0.1) is the recommended personalisation strategy for the frozen LogReg group model (K = 30, C = 0.001, L2, StandardScaler). It achieves the best balanced accuracy across all calibration durations, with well-balanced sensitivity and specificity. As little as 60 s of calibration data per label lifts mean balanced accuracy from 0.60 → 0.70, and 180 s reaches 0.76.

**Date:** 2026-03-13

---

## Context

The LogReg hyperparameter plateau document (`dc_logreg_hyperparameter_plateau.md`) froze the cross-subject LogReg pipeline at K = 30, C = 0.001, L2, StandardScaler. The group-level model achieves mean AUC ≈ 0.65 under LOSO evaluation with calibration normalisation (ADR-0004), but individual participant variance is high (std ≈ 0.12). This investigation asks: **how much can a short calibration session improve per-participant accuracy, and which personalisation strategy is best?**

**Fixed settings (shared across all strategies):**

| Setting | Value |
|---------|-------|
| Group classifier | LogisticRegression(K = 30, C = 0.001, L2, StandardScaler) |
| Normalisation | Calibration norm (ADR-0004): z-score using fixation + Forest_0 baseline |
| Evaluation | LOSO (41 folds) |
| Participants | 41 (6 QC-excluded from 47) |
| Features | 54 (13 bandpower + 12 Hjorth + 4 spectral entropy + 4 permutation entropy + 12 stats + 4 aperiodic + 5 wPLI) |
| Epoch structure | 2 s windows, 0.5 s step, 128 Hz, 128 EEG channels |
| Calibration sampling | Random within blocks (balanced labels) |
| Calibration durations | 30, 60, 90, 120, 180 s per label |
| Gap buffer | ±3 epochs around calibration chunks (prevents temporal leakage) |
| Threshold tuning | Youden's J on calibration data (test uses personalised threshold) |
| Seed | 42 |

Script: `scripts/logreg_personalisation_comparison.py`  
Results: `results/test_pretrain/logreg_personalisation_comparison.json`

---

## Strategies evaluated

| Code | Strategy | Description |
|------|----------|-------------|
| A | Group only | Frozen group LogReg, no personalisation (baseline). Threshold set on cal data. |
| B | Warm-start strong L2 | Warm-start from group weights, retrain with C = 0.01 on cal data. |
| C | Warm-start weak L2 | Warm-start from group weights, retrain with C = 0.1 on cal data. |
| D | Scratch LogReg | Discard group model, train fresh LogReg on cal data only. |
| E | Scratch RF | Train RandomForest(100 trees) from scratch on cal data only. |
| F | Incremental SGD | SGDClassifier with log_loss, warm-started from group weights, partial_fit on cal data. |

---

## Results

### Mean AUC (n = 41)

| Cal (s) | Group (A) | WS-strong (B) | WS-weak (C) | Scratch LR (D) | Scratch RF (E) | SGD-inc (F) |
|---------|-----------|----------------|--------------|-----------------|----------------|-------------|
| 30 | 0.655 | 0.708 | 0.742 | 0.708 | 0.762 | 0.731 |
| 60 | 0.637 | 0.723 | 0.755 | 0.704 | 0.765 | 0.719 |
| 90 | 0.643 | 0.731 | 0.760 | 0.707 | 0.763 | 0.733 |
| 120 | 0.641 | 0.751 | 0.785 | 0.722 | 0.795 | 0.752 |
| 180 | 0.631 | 0.802 | 0.836 | 0.771 | 0.851 | 0.792 |

### Delta AUC vs Group (mean per-participant)

| Cal (s) | WS-strong (B) | WS-weak (C) | Scratch LR (D) | Scratch RF (E) | SGD-inc (F) |
|---------|----------------|--------------|-----------------|----------------|-------------|
| 30 | +0.053 | +0.087 | +0.053 | +0.106 | +0.076 |
| 60 | +0.086 | +0.118 | +0.066 | +0.128 | +0.082 |
| 90 | +0.088 | +0.117 | +0.064 | +0.120 | +0.090 |
| 120 | +0.110 | +0.145 | +0.081 | +0.154 | +0.111 |
| 180 | +0.171 | +0.205 | +0.140 | +0.220 | +0.161 |

### Balanced accuracy at personalised threshold (mean, n = 41)

| Cal (s) | Group (A) | WS-strong (B) | WS-weak (C) | Scratch LR (D) | Scratch RF (E) | SGD-inc (F) |
|---------|-----------|----------------|--------------|-----------------|----------------|-------------|
| 30 | 0.607 | 0.661 | **0.682** | 0.658 | 0.626 | 0.671 |
| 60 | 0.605 | 0.674 | **0.700** | 0.659 | 0.655 | 0.669 |
| 90 | 0.603 | 0.676 | **0.693** | 0.660 | 0.653 | 0.680 |
| 120 | 0.602 | 0.693 | **0.715** | 0.666 | 0.679 | 0.689 |
| 180 | 0.602 | 0.735 | **0.762** | 0.703 | 0.742 | 0.727 |

---

## Key observations

1. **All personalisation strategies improve over the group model.** Even 30 s of calibration data per label provides a meaningful AUC lift (+0.05 to +0.11 depending on strategy).

2. **WS-weak (C = 0.1) wins on balanced accuracy at every calibration duration.** This is the metric that matters for online deployment, where both high and low MWL states must be detected. BalAcc ranges from 0.68 at 30 s to 0.76 at 180 s.

3. **Scratch RF wins on AUC but has biased thresholds.** RF achieves the highest raw AUC (0.76–0.85) but its Youden threshold produces high specificity (0.89–0.95) with poor sensitivity (0.36–0.63), making it unsuitable for online MWL detection where missing high-load events is costly.

4. **WS-weak produces the most balanced sens/spec trade-off.** At 120 s calibration: sens = 0.69, spec = 0.74. At 180 s: sens = 0.73, spec = 0.79. Neither class is systematically over- or under-detected.

5. **More calibration data consistently helps.** All strategies show monotonic or near-monotonic improvement from 30 → 180 s. The practical recommendation depends on how much calibration time is acceptable in the experimental protocol.

6. **Incremental SGD offers no advantage over warm-start.** SGD-inc is conceptually appealing for online updates but is consistently outperformed by WS-weak across all durations and metrics.

---

## Recommended configuration for online deployment

| Parameter | Value |
|-----------|-------|
| Strategy | Warm-start weak L2 (C = 0.1) |
| Group model | K = 30, C = 0.001, L2, StandardScaler (frozen) |
| Calibration duration | 120 s per label (practical) or 180 s per label (best) |
| Expected BalAcc | 0.72 (120 s) to 0.76 (180 s) |
| Expected mean AUC | 0.79 (120 s) to 0.84 (180 s) |

---

## Note on metric aggregation

The group model AUC reported here (≈ 0.65) differs from the plateau document (≈ 0.60) because of two methodological differences:

1. **Evaluation scheme:** This comparison uses LOSO (leave-one-subject-out), while the plateau doc used 70/30 random train/test splits.
2. **AUC aggregation:** This comparison reports the *mean of per-participant AUCs*, while the plateau doc reported *pooled AUC* (concatenate all test epochs, compute one ROC). Per-participant mean AUC is typically higher when inter-participant variance is large (std ≈ 0.12), because participants with extreme class imbalance or near-chance performance pull down pooled AUC disproportionately.

Both numbers are correct for their respective evaluation designs.

---

## Implications

- **Use WS-weak (C = 0.1) for the adaptive MATB experiment.** It is the best-performing strategy on balanced accuracy with a well-balanced sensitivity/specificity trade-off.
- **Plan for ≥ 120 s of calibration per label** in the experimental protocol. This gives BalAcc ≈ 0.72, a meaningful improvement over the 0.60 group baseline.
- **Scratch RF should not be used online** despite its high AUC — its biased threshold makes it unreliable for detecting high-load epochs.
- **The warm-start approach is also attractive for its simplicity:** it requires only `clf.set_params(C=0.1); clf.fit(X_cal, y_cal)` after initialising from the group model weights, with no additional hyperparameters.
