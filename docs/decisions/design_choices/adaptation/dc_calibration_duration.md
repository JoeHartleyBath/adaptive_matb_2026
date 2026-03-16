# Design choice: 120 s per label is the best calibration duration in offline TSST controller simulation

**Decision**  
In the full-cohort LOSO controller sweep on VR-TSST data (41 participants), 120 s of calibration data per label yields better end-to-end controller balanced accuracy than 60 s. This establishes 120 s as the reference calibration window for offline simulation and comparison purposes.

**Date:** 2026-03-16 (corrected; original 2026-03-13 was based on P05-only data)

> **Correction notice (2026-03-16):** The original version of this document (dated 2026-03-13) concluded that 60 s outperformed 120 s with a best controller BA of 94.4%. That analysis was inadvertently based on a single-participant (P05) test run — a subsequent `--only P05` invocation of the sweep script overwrote the full results. The document has been rewritten from a verified full-cohort sweep (41 participants × 5 seeds × 218,120 runs, completed 2026-03-16).

---

## Scope and limitations

**This finding applies to the offline TSST simulation only.** The calibration and test data come from the same VR-TSST session (4 condition blocks, ~297 epochs each), with calibration chunks randomly sampled from within those blocks. This is fundamentally different from the planned MATB online study, where:

- Calibration will be a **dedicated phase** with its own task structure (DC-07).
- The adaptive task session will be temporally separate from calibration.
- Task demands, environment, and epoch statistics will differ between calibration and deployment.

The MATB study's calibration duration should be determined by piloting, not by transplanting this TSST-specific finding. What this document does establish is that **more calibration data does not automatically improve controller-level performance** — there is a budget-vs-accuracy tradeoff that must be evaluated at the controller level, not just via classifier AUC.

---

## Context

The LogReg WS-weak personalisation pipeline (group: K = 30, C = 0.001; personalised: warm-start C = 0.1) was validated in `dc_logreg_personalisation_comparison.md`. That document showed that classifier AUC improves with more calibration data (60 s → 120 s → 180 s per label), but the gains diminish rapidly after 60 s.

This follow-up asks: **when the personalised classifier feeds into a smoother + hysteresis controller, which calibration duration yields the best end-to-end controller performance in the offline TSST simulation?** In this simulation, calibration data is drawn from the same session as test data, so longer calibration shrinks the available test set.

**Investigation:**  
A full LOSO sweep (41 participants × 5 seeds × 2 calibration durations × 7 threshold strategies × 19 smoother configs × 4 hysteresis margins = 218,120 simulation runs) evaluated 60 s and 120 s per label across the complete controller parameter space.

Script: `scripts/sweep_mwl_smoothers_logreg.py`  
Results: `results/test_pretrain/smoother_sweep_logreg.json`

---

## Fixed settings

| Setting | Value |
|---------|-------|
| Group classifier | LogisticRegression(K = 30, C = 0.001, L2, StandardScaler) |
| Personalisation | Warm-start LogReg, C = 0.1 (WS-weak) |
| Normalisation | Calibration norm (ADR-0004): fixation + Forest_0 baseline |
| Evaluation | LOSO (41 folds) |
| Smoother types | EMA, SMA, AdaptiveEMA, FixedLag (19 configs) |
| Threshold strategies | Youden ± offset, fixed (0.50, 0.60), cost-weighted (w = 0.6, 0.7) |
| Hysteresis margins | 0.00, 0.02, 0.05, 0.08 |
| Seeds | 5 (random calibration chunk placement) |
| Gap buffer | ±3 epochs around calibration chunks |

---

## Results: 60 s vs 120 s per label

### Aggregate performance (mean across all configs, seeds, and 41 participants)

| Cal duration | Mean bal_acc (across all configs) | n configs |
|-------------|----------------------------------|----------|
| 60 s/label | 76.3% | 532 |
| **120 s/label** | **77.8%** | 532 |

### Top-5 controller configs at each calibration duration

**60 s/label:**

| Threshold | Smoother | Hyst | Mean BA | ±SD |
|-----------|----------|------|---------|-----|
| fixed_0.50 | fl_l4_p0.001_m0.2 | 0.08 | 79.5% | 15.6 |
| fixed_0.50 | ema_a0.03 | 0.00 | 79.5% | 14.9 |
| fixed_0.50 | fl_l8_p0.001_m0.2 | 0.08 | 79.5% | 15.3 |
| fixed_0.50 | ema_a0.03 | 0.02 | 79.4% | 14.3 |
| fixed_0.50 | ema_a0.05 | 0.08 | 79.4% | 15.3 |

**120 s/label:**

| Threshold | Smoother | Hyst | Mean BA | ±SD |
|-----------|----------|------|---------|-----|
| fixed_0.50 | ema_a0.05 | 0.02 | 80.3% | 15.6 |
| fixed_0.50 | ema_a0.05 | 0.00 | 80.3% | 15.6 |
| fixed_0.50 | aema_0.05_0.15 | 0.08 | 80.1% | 16.0 |
| fixed_0.50 | ema_a0.1 | 0.08 | 80.1% | 15.9 |
| fixed_0.50 | fl_l4_p0.001_m0.2 | 0.05 | 80.1% | 15.8 |

### Calibration budget impact

| Cal duration | Epochs/label | Total cal epochs | % of session consumed |
|-------------|-------------|-----------------|----------------------|
| **60 s** | 120 | 240 | **~20%** |
| 120 s | 240 | 480 | ~40% |

---

## Rationale

- **120 s outperforms 60 s on controller metrics** (+1.5% aggregate mean, +0.8% top-5 peak). The additional calibration data improves classifier personalisation sufficiently to overcome the smaller test set.
- **fixed_0.50 dominates all top configs.** Neither Youden thresholds nor cost-weighted thresholds appear in the top ranks — the simple fixed 0.50 decision boundary is universally best across both calibration durations. This simplifies deployment (no need to compute per-participant Youden thresholds).
- **Inter-participant variance is high** (SD ≈ 15%). The 0.8% BA gap between durations is small relative to this variance, so the choice is not strongly constrained. Both durations are defensible.
- **Classifier AUC gains at 120 s now survive the controller stack.** Unlike the previous (incorrect, P05-only) analysis, the full-cohort sweep shows the AUC improvement at 120 s translates to better end-to-end controller BA.
- **Test-set budget still matters** but does not reverse the conclusion in the full cohort. The ~40% session consumption at 120 s is a practical concern but does not negate the BA advantage.

---

## Alternatives considered

| Duration | Assessment |
|----------|-----------|
| 30 s/label | Pruned in the RBF sweep as "never competitive". Not re-tested here. |
| 60 s/label | Tested. Top-5 peak 79.5% vs 80.3% at 120 s. Uses only ~20% of session (budget advantage) but slightly lower BA. |
| 90 s/label | Not swept (interpolation between 60 and 120 — unlikely to reverse the trend). |
| 180 s/label | Validated in personalisation comparison (BalAcc = 0.76). Not practical online — would consume ~60% of session. |

---

## Implications

- **For offline TSST simulation:** use 120 s per label as the default calibration window when comparing controller configurations.
- **Best controller config:** EMA α = 0.05, fixed threshold 0.50, hysteresis 0.02, 120 s/label — mean BA 80.3% ± 15.6% across 41 participants.
- **Threshold strategy simplification:** fixed_0.50 beats all Youden and cost-weighted variants. Per-participant Youden threshold computation is not needed for the controller.
- **For the MATB online study:** this result does *not* directly prescribe the calibration duration. The MATB calibration phase is a separate, dedicated block (DC-07) with different task structure and epoch characteristics. The MATB calibration duration should be determined during piloting.
- **Transferable insight:** with the full cohort, more calibration data (120 s) does translate to better controller BA. The earlier (incorrect) finding that 60 s was superior was an artifact of single-participant analysis.
- The `sweep_mwl_smoothers_logreg.py` script retains both durations for reproducibility.

---

## Status

Final (scoped to offline TSST simulation). Corrected 2026-03-16 from full-cohort sweep.
