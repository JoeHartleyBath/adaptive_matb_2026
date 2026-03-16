# Design choice: RBF hyperparameter tuning has reached ceiling with current features

**Decision**  
The Nystroem-approximated RBF SVM pipeline is already near-optimal for its hyperparameters. Adding aperiodic 1/f slope (4 features) and wPLI connectivity (5 features) lifted performance from 0.6615 → 0.6725 mean AUC (+0.011), the largest single-change improvement observed. The canonical feature set is now **54 features**.

**Date:** 2026-03-12

---

## Context

After establishing that Nystroem-RBF was the strongest base learner in the ensemble LOSO evaluation (mean AUC 0.662, median 0.680, n = 28 after Scenario C QC exclusion), a systematic one-change-at-a-time ablation was run overnight to isolate which hyperparameter lever — if any — could push AUC higher.

**Baseline pipeline (Experiment A):**

| Setting | Value |
|---------|-------|
| Kernel approximation | Nystroem (n_components = 300) |
| Gamma × C grid | 6 configs: γ ∈ {0.01, 0.05, 0.1}, C = 1.0 |
| Feature selector | f_classif via SelectKBest |
| K candidates | {15, 20, 25, 30, 45} |
| K probe model | LogisticRegression |
| Inner CV | StratifiedGroupKFold(5) |
| Outer CV | LOSO (28 folds) |
| Per-participant z-scoring | Yes |
| Final classifier | LogisticRegression(C = 1.0) on Nystroem-transformed features |

Script: `scripts/rbf_ablation_loso.py`

---

## Ablation results (experiments A–G)

| Exp | Label | What changed | Mean AUC | Δ vs A | Median AUC | BalAcc | F1 | Time |
|-----|-------|--------------|----------|--------|------------|--------|----|------|
| **A** | baseline | — | 0.6615 | — | 0.6796 | 0.6191 | 0.6180 | 6 min |
| B | wide_gamma | 10 γ × 6 C = 60 configs | 0.6572 | −0.0043 | 0.6716 | 0.6159 | 0.6148 | 17 min |
| C | rbf_k_probe | RBF (not LogReg) selects K | 0.6587 | −0.0028 | 0.6728 | 0.6172 | 0.6161 | 11 min |
| D | nys_500 | n_components = 500 | 0.6622 | +0.0007 | 0.6804 | 0.6204 | 0.6194 | 9 min |
| **E** | nys_800 | n_components = 800 | **0.6632** | **+0.0017** | 0.6780 | 0.6206 | 0.6195 | 16 min |
| F | fine_k | K ∈ {10…45}, 11 values | 0.6628 | +0.0013 | 0.6716 | **0.6218** | **0.6207** | 7 min |
| G | mi_select | mutual_info_classif | 0.6616 | +0.0001 | 0.6688 | 0.6204 | 0.6194 | 93 min |

Experiments H (combined) and I (combined + MI) were not completed; H crashed with NaN from extreme γ values in Nystroem. Given the negligible single-lever gains, running H/I was deemed unnecessary.

Experiment B2 (wide C only) was not run for the same reason.

---

## Key observations

1. **No single lever produced a meaningful improvement.** The largest gain (E, nys_800) was +0.0017 mean AUC — well within fold-level noise (std ≈ 0.145).

2. **Expanding the gamma grid hurt performance.** Experiment B (60-config grid) scored *lower* than baseline, suggesting over-search introduces noise in inner-CV selection with small training folds.

3. **γ = 0.01 dominates.** In 6 out of 7 experiments, γ = 0.01 was selected for nearly every outer fold. The signal lives in a narrow bandwidth of the RBF kernel.

4. **K = 30 dominates.** Across all experiments, K = 30 was selected in ~54% of folds. The fine-K experiment (F) spread selections more evenly (K = 22–35) but gained only +0.0013 AUC.

5. **Mutual information selection was 15× slower with zero gain.** mi_select took 93 minutes vs 6 minutes for baseline, for +0.0001 AUC. f_classif is sufficient.

6. **Nystroem components beyond 300 provide diminishing returns.** nys_500 and nys_800 gave +0.0007 and +0.0017 respectively — negligible relative to the 2–3× compute cost.

---

## Rationale

- The ablation was designed to isolate individual levers so gains (or losses) can be attributed cleanly.
- Seven experiments spanning five distinct hyperparameter axes all produced Δ AUC < 0.002.
- The 45 hand-crafted features (13 bandpower + 12 Hjorth + 4 spectral entropy + 4 permutation entropy + 12 stats) set an information-theoretic ceiling that no amount of kernel tuning can surpass.
- The LOSO std ≈ 0.145 means the true per-fold AUC is highly variable; marginal hyperparameter changes are dominated by participant-level variance.

---

## Implications

- **Freeze the RBF baseline config** (γ = 0.01, C = 1.0, nys = 300, K = 30, f_classif) for all future comparisons.
- **Canonical feature set is now 54 features** (45 original + 4 aperiodic slope + 5 wPLI). The fine_k grid no longer helps and should not be used.
- Accept LOSO AUC ≈ 0.67 as the cross-participant ceiling for this dataset + pipeline.
- Earlier Phase B attempts (B1 sub-bands, B1b temporal) confirmed that not all features help — only 1/f and wPLI provided genuine lift.

---

## Phase B feature expansion attempts (2026-03-12)

Two feature expansion experiments were run and both failed to improve AUC:

### B1: Sub-band power + asymmetry (62 features)

Added 17 features: LowAlpha/HighAlpha/LowBeta/HighBeta power (10), parietal + temporal alpha asymmetry (2), sub-band ratios HA/LA and LB/HB (5).

| Exp | Mean AUC | Δ vs 45-feat A | Median AUC |
|-----|----------|----------------|------------|
| A baseline (62 feat) | 0.6589 | −0.0026 | 0.6632 |
| F fine_k (62 feat) | 0.6623 | +0.0008 | 0.6709 |

K=10 dominated (11/28 folds in F) — model actively discarded the new features. Sub-bands are collinear with full-band power.

### B1b: Temporal region time-domain features (53 features)

Added 8 features: Hjorth (3), SpEnt (1), PeEnt (1), stats (3) for the Temporal region (20 channels, previously unused).

| Exp | Mean AUC | Δ vs 45-feat A | Median AUC |
|-----|----------|----------------|------------|
| A baseline (53 feat) | 0.6548 | −0.0067 | 0.6651 |
| F fine_k (53 feat) | 0.6557 | −0.0058 | 0.6674 |

K=45 dominated (19/28 folds in A) — model kept all features but AUC still dropped. K=40 dominated in F (11/28). Temporal channels add noise, not signal for MWL discrimination.

**Both sets reverted.**

### B2: Aperiodic 1/f slope + wPLI connectivity (54 features) — SUCCESS

Added 9 features:
- **1/f aperiodic slope** (4): OLS fit in log10-log10 PSD space (1–40 Hz, excluding 7–14 Hz alpha peak) for FrontalMidline, Parietal, Central, Occipital. More negative slope = steeper 1/f.
- **wPLI connectivity** (5): Weighted phase-lag index for 5 region-pair × band combinations: FM↔Par theta, FM↔Par alpha, FM↔Cen theta, Cen↔Par alpha, FM↔Occ alpha. Bandpass (4th-order Butterworth) → Hilbert → cross-spectral → wPLI.

| Config | Mean AUC | Δ vs 45-feat A | Median AUC | BalAcc | F1 |
|--------|----------|----------------|------------|--------|----|
| RBF A baseline (54 feat) | **0.6725** | **+0.011** | **0.7000** | 0.6313 | 0.6306 |
| RBF F fine_k (54 feat) | 0.6687 | +0.007 | 0.6842 | 0.6271 | 0.6264 |
| LogReg LOSO (54 feat) | 0.6547 | — | 0.6845 | 0.6173 | 0.6164 |

Key observations:
- **+0.011 mean AUC / +0.020 median AUC** — the largest improvement from any single change in this project.
- Median AUC crossed **0.70** for the first time.
- Fine_k (F) now *hurts* (−0.004 vs A), suggesting the coarser k grid generalises better with the expanded feature set.
- K distribution in A: k=30 dominated (10/28 folds), with k=25 (6) and k=45 (6) also common. The model uses more features than the 45-feature baseline.
- LogReg also improved (+0.009 mean AUC), confirming the lift is feature-driven, not model-specific.

**54 features retained as the canonical feature set.**

---

## Alternatives considered

| Alternative | Why not |
|-------------|---------|
| Run H/I anyway | H crashed with NaN; single-lever gains were all < 0.002, so combining them is unlikely to yield meaningful lift |
| Polynomial kernel | Less interpretable, rarely outperforms RBF on EEG features in the literature |
| Deep kernel learning | Requires substantially more data; 28 participants is too few |
| Sub-band decomposition (B1) | Tested: 62 features, AUC −0.003. Collinear with full-band power |
| Temporal region features (B1b) | Tested: 53 features, AUC −0.007. Temporal channels add noise |
| Aperiodic 1/f slope | **Implemented and retained.** 4 features, contributed to +0.011 AUC lift |
| wPLI connectivity | **Implemented and retained.** 5 features, contributed to +0.011 AUC lift |

---

## Per-participant z-scoring: ablation and online implications (2026-03-12)

The pipeline applies per-participant z-scoring (`StandardScaler().fit_transform()` on all epochs for each participant) **before** any train/test split. This uses the full session's mean/std — including future epochs — which is incompatible with online deployment where only past data is available.

### Ablation result

| Condition | Mean AUC | Median AUC | BalAcc | F1 |
|-----------|----------|------------|--------|----|
| With pp z-score (full session) | **0.6725** | **0.7000** | 0.6313 | 0.6306 |
| Mixed norm (pp z-score train + calibration test) | 0.6082 | 0.6049 | 0.5638 | 0.5194 |
| Without pp z-score | 0.6116 | 0.6168 | 0.5582 | 0.5119 |

Removing per-participant z-scoring drops AUC by 6 points. Without it, the pipeline-level `StandardScaler` (fitted on training participants only) cannot compensate for large between-participant feature scale differences, and the classifier partially learns participant identity rather than workload state.

### Why this is not a LOSO leakage problem

The z-scoring is within-participant: each participant's scaler uses only their own data. In LOSO, the test participant is a different person — no cross-participant information leaks. The z-scoring removes within-participant offset/scale, which is exactly what a calibration baseline would do in deployment.

### Mixed normalisation: LOSO result (2026-03-12)

The mixed normalisation strategy (`prepare_mixed_norm()`) was tested on the canonical 54-feature RBF config (Exp A):

- **Training participants**: per-participant z-score (full session — appropriate because training data is always offline)
- **Held-out test participant**: calibration z-score using fixation + Forest_0 baseline only (temporally causal)

Result: **mean AUC 0.6082, median 0.6049** — a drop of −0.0643 mean / −0.0951 median from the full-session optimistic figure.

Notably, the mixed norm result is marginally *lower* than the no-zscore baseline (0.6082 vs 0.6116 mean). This indicates the short Forest_0 calibration window does not provide reliable enough statistics to normalise test epochs: when the calibration estimate is noisy, it degrades the test signal more than leaving it un-normalised.

The K and γ distributions are identical across pp z-score and mixed norm runs — the training-side normalisation is the same; only the test representation changes.

**Implication for deployment:** A longer or repeated calibration baseline (multiple forest blocks rather than Forest_0 only) would likely be needed to close the gap.

### Online deployment strategy

The VR-TSST session structure provides a natural solution. Each task block is immediately preceded by a 180 s Forest relaxation scene (Forest_N → Task_N). Additionally, 390 s of pre-task baseline precedes the first task (Movement_Baseline 90 s + 2 × Fixation 60 s + Forest1 180 s).

**Implemented and tested:** `prepare_mixed_norm()` in `src/ml/pretrain_loader.py` exports Forest epochs alongside task epochs in the norm cache, computes per-participant z-score statistics from baseline (Forest) data only, and applies those statistics to normalise task epochs. For the test participant in LOSO, only temporally-preceding baseline data is used. Mixed norm result: 0.6082 mean AUC (−0.064 vs upper bound).

For online MATB deployment, a calibration block at session start is required. Performance target should be set against the mixed-norm figure (0.6082), not the optimistic pp z-score figure (0.6725).

See `results/test_pretrain/rbf_ablation_no_pp_zscore.json` and `results/test_pretrain/rbf_ablation_mixed_norm.json` for full per-fold results.

---

**Status:** Final (updated 2026-03-12 with B2 1/f + wPLI success, pp z-score ablation, mixed norm evaluation)

**References**
- `scripts/rbf_ablation_loso.py` — ablation script
- `scripts/personalised_logreg.py` — feature extraction (contains `_aperiodic_slope_batch()` and `_wpli_batch()`)
- `results/test_pretrain/rbf_ablation.json` — full per-fold results (54 features, exp A + F)
- `results/test_pretrain/rbf_ablation_45feat.json` — original 45-feature ablation (exp A–G)
- `results/test_pretrain/rbf_ablation_phaseB1_62feat.json` — 62-feature experiment
- `results/test_pretrain/rbf_ablation_53feat_temporal.json` — 53-feature experiment
- `results/test_pretrain/loso_54feat_logreg.json` — LogReg LOSO with 54 features
- `results/test_pretrain/rbf_ablation_log.txt` — runtime log
- `results/test_pretrain/rbf_ablation_no_pp_zscore.json` — no-pp-zscore ablation (exp A)
- `results/test_pretrain/rbf_ablation_mixed_norm.json` — mixed norm ablation (exp A, 2026-03-12)
- `results/test_pretrain/rbf_ablation_pp_zscore_verify.json` — pp z-score reproducibility check (exp A, 2026-03-12)
- `scripts/ensemble_loso.py` — prior stacking evaluation that identified RBF as best base
