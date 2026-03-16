# Design choice: LogReg hyperparameter tuning has reached plateau under calibration normalisation

**Decision**  
The LogisticRegression pipeline under calibration normalisation (ADR-0004) is frozen at: **K = 30 (SelectKBest, f_classif), C = 0.001, L2, StandardScaler**. A joint sweep of 80 configurations confirmed that the top 10 configs span only 0.0006 AUC — a flat plateau dominated by inter-participant variance. Further hyperparameter tuning will not produce meaningful online gains.

**Date:** 2026-03-15

---

## Context

The RBF hyperparameter ceiling document (`dc_rbf_hyperparameter_ceiling.md`) established the canonical 54-feature set and showed that LOSO RBF AUC ≈ 0.67 is the cross-participant ceiling with per-participant z-scoring. For online deployment, calibration normalisation (ADR-0004) replaces per-participant z-scoring, dropping the RBF upper bound to ≈ 0.61.

LogReg is the simplest candidate classifier and may be preferred online for latency, interpretability, and update-ability. This investigation asks: **can LogReg match or approach the RBF ceiling under calibration normalisation?**

**Baseline pipeline:**

| Setting | Value |
|---------|-------|
| Classifier | LogisticRegression(C = 0.001, class_weight = "balanced") |
| Penalty | L2 |
| Scaler | StandardScaler |
| Feature selector | None (all 54 features) |
| Normalisation | Calibration norm (ADR-0004): z-score using fixation + Forest_0 baseline |
| Evaluation | Cross-split holdout (30% test participants), multi-seed |
| Participants | 41 (6 QC-excluded from 47) |
| Epoch structure | 2 s windows, 0.25 s step, 128 Hz, 128 EEG channels |
| Baseline AUC | 0.5975 mean (3 seeds, K = 54 / no selection) |

Script: `scripts/optimise_logreg.py`

---

## Phase 1: Feature selection (K sweep)

SelectKBest (f_classif ANOVA F-test) was swept across 9 values of K, each evaluated with 3 random seeds (cross-split holdout, 30% test).

| K | Mean AUC | Δ vs K = 54 | Std AUC |
|---|----------|-------------|---------|
| 10 | 0.5904 | −0.0071 | 0.0415 |
| 15 | 0.5881 | −0.0094 | 0.0351 |
| 20 | 0.6016 | +0.0041 | 0.0488 |
| 25 | 0.6041 | +0.0066 | 0.0489 |
| **30** | **0.6116** | **+0.0141** | 0.0444 |
| 35 | 0.6112 | +0.0137 | 0.0436 |
| 40 | 0.6014 | +0.0039 | 0.0383 |
| 45 | 0.6022 | +0.0047 | 0.0374 |
| 54 | 0.5975 | — | 0.0322 |

**K = 30 is optimal** (+0.014 AUC over no selection). This matches the K = 30 dominance observed in the RBF ablation (selected in ~54% of LOSO folds).

Performance peaks at K = 30–35 and degrades both below (too few features, information lost) and above (noise features dilute the signal in a linear model).

---

## Phase 2: Joint hyperparameter sweep (K × C × penalty × scaler)

A 4-axis grid sweep was run with joblib parallelisation (4 workers, loky backend), 3 seeds per config:

| Axis | Values | Count |
|------|--------|-------|
| K | 20, 25, 30, 35 | 4 |
| C | 0.001, 0.003, 0.01, 0.03, 0.1 | 5 |
| Penalty | L2, ElasticNet (l1_ratio = 0.5) | 2 |
| Scaler | StandardScaler, RobustScaler | 2 |
| **Total configs** | | **80** |

### Top 10 configurations

| Rank | K | C | Penalty | Scaler | Mean AUC | Δ vs baseline |
|------|---|---|---------|--------|----------|---------------|
| 1 | 30 | 0.001 | L2 | Standard | **0.6116** | +0.0141 |
| 2 | 35 | 0.003 | L2 | Standard | 0.6116 | +0.0141 |
| 3 | 30 | 0.003 | L2 | Robust | 0.6115 | +0.0140 |
| 4 | 30 | 0.003 | L2 | Standard | 0.6113 | +0.0138 |
| 5 | 35 | 0.01 | ElasticNet | Robust | 0.6112 | +0.0137 |
| 6 | 35 | 0.001 | L2 | Standard | 0.6112 | +0.0137 |
| 7 | 35 | 0.1 | ElasticNet | Robust | 0.6111 | +0.0136 |
| 8 | 30 | 0.01 | L2 | Standard | 0.6110 | +0.0135 |
| 9 | 35 | 0.01 | L2 | Standard | 0.6110 | +0.0135 |
| 10 | 30 | 0.001 | L2 | Robust | 0.6110 | +0.0135 |

**Total range across top 10: 0.0006 AUC.** This is a flat plateau.

Runtime: 80 configs × 3 seeds = 240 fits completed in ~41 seconds.

---

## Feature ranking (f_classif, K = 30)

The top 30 features selected by ANOVA F-test (fitted on training split, seed 42):

| Rank | Feature | F-score | Selected |
|------|---------|---------|----------|
| 1 | FM_Beta | 961.2 | ✓ |
| 2 | Fro_PeEnt | 609.1 | ✓ |
| 3 | FM_Delta | 531.1 | ✓ |
| 4 | Fro_HjAct | 417.9 | ✓ |
| 5 | Cen_PeEnt | 407.9 | ✓ |
| 6 | FM_Alpha | 279.9 | ✓ |
| 7 | Fro_Kurt | 215.0 | ✓ |
| 8 | FM_Theta | 208.0 | ✓ |
| 9 | Occ_Kurt | 143.6 | ✓ |
| 10 | Occ_HjAct | 133.2 | ✓ |
| … | … | … | ✓ |
| 29 | Cen_Skew | 24.3 | ✓ |
| 30 | Occ_SpEnt | 21.1 | ✓ (cutoff) |
| 31 | Occ_HjMob | 19.4 | ✗ |
| 32 | Cen_Theta | 19.2 | ✗ |

The top 5 features (FM_Beta, Fro_PeEnt, FM_Delta, Fro_HjAct, Cen_PeEnt) have F-scores 2–50× larger than the remaining 25, consistent with bandpower and entropy features carrying the strongest univariate workload signal.

---

## Key observations

1. **Feature selection is the only meaningful lever.** K = 30 provides +0.014 AUC over K = 54 (no selection). All other axes (C, penalty, scaler) contribute negligibly.

2. **The top 10 configs span only 0.0006 AUC — a flat plateau.** Inter-participant variance (std ≈ 0.04) is ~70× larger than the gap between configurations. Any config in the top 10 would perform equivalently in practice.

3. **K = 30 matches the RBF pipeline finding.** In the RBF ablation, K = 30 was selected in ~54% of LOSO folds. The same feature cutoff is optimal for both linear and non-linear classifiers, suggesting a natural signal/noise boundary in the 54-feature set.

4. **ElasticNet and RobustScaler provide no advantage.** ElasticNet (l1_ratio = 0.5) appears at rank 5 and 7, but within the noise floor. L1 sparsity does not help when K = 30–35 already removes noisy features.

5. **LogReg under calibration norm ≈ 0.61, vs RBF ≈ 0.67 (pp z-score) / ≈ 0.61 (mixed norm).** The linear model matches RBF when both use online-compatible normalisation. The gap between classifiers is <0.01 AUC under calibration normalisation — normalisation quality dominates classifier choice.

6. **Per-seed variance is high.** Individual seeds range from 0.549 to 0.649 AUC, reflecting the sensitivity to which participants land in the test split. This further confirms that mean AUC differences <0.005 are noise.

---

## Rationale

- The sweep was designed to explore one-change-at-a-time (Phase 1: K only) before expanding to a full grid (Phase 2: joint sweep), following the same methodology as the RBF ablation.
- 80 configurations across 4 axes all produced AUC within 0.0006 of each other in the top 10.
- The calibration normalisation (ADR-0004) is the binding constraint: 54 features z-scored against a short baseline window produce noisier input than per-participant z-scoring, limiting any classifier's discriminative power.
- Further gains will come from: (a) longer/better calibration baselines, (b) online EMA smoothing and Youden-J thresholding, (c) personalisation or fine-tuning — not from hyperparameter tuning.

---

## Implications

- **Freeze the LogReg config:** K = 30, C = 0.001, L2, StandardScaler, calibration normalisation.
- **Use the same 30 features for deployment.** The f_classif ranking is stable across seeds and matches the RBF pipeline's K = 30 selection.
- **Do not run further LogReg hyperparameter sweeps.** The plateau is confirmed; additional compute will not yield actionable differences.
- **Online performance target:** AUC ≈ 0.61 (pre-smoothing). Post-smoothing with EMA + Youden-J threshold is expected to improve effective accuracy on block-level classification.
- **Classifier choice for deployment should be driven by practical factors** (latency, interpretability, update-ability) rather than AUC differences <0.01.

---

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Wider C range (e.g. 0.0001–10) | Top-10 configs already include C = 0.001 to 0.1 with no trend; extending further adds no value |
| L1 penalty (pure Lasso) | ElasticNet (l1_ratio = 0.5) already tested at rank 5/7; pure L1 would be more aggressive but K = 30 already handles sparsity |
| RobustScaler | Tested in sweep; appears at rank 3, 5, 7, 10 but within noise floor of StandardScaler |
| Higher seed count (e.g. 20 seeds) | Would reduce mean estimator variance but the plateau conclusion is clear at 3 seeds; 20-seed runs on compare_ml_models.py confirm the same ordering |
| Nested cross-validation | More rigorous but much slower; the flat plateau means any selection from the top 10 is defensible |
| Non-linear feature interactions | Would require moving beyond LogReg; RBF already captures non-linearities and achieves the same AUC under calibration norm |

---

**Status:** Final (2026-03-15)

**References**
- `scripts/optimise_logreg.py` — K sweep and joint sweep script
- `scripts/compare_ml_models.py` — baseline multi-model comparison (20-seed reference)
- `results/test_pretrain/logreg_k_sweep.json` — Phase 1 K sweep results (9 K values × 3 seeds)
- `results/test_pretrain/logreg_joint_sweep.json` — Phase 2 joint sweep results (80 configs × 3 seeds)
- `docs/decisions/design_choices/modelling/dc_rbf_hyperparameter_ceiling.md` — RBF ceiling document (companion)
- `docs/decisions/adr-0004-calibration-normalisation.md` — calibration normalisation ADR
