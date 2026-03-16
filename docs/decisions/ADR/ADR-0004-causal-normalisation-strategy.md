# ADR-0004: Calibration-based causal normalisation for online EEG classification

## Status

Accepted — 2026-03-13

## Context

The adaptive MATB system requires online-compatible (causal) per-participant
z-scoring of EEG features.  The pre-training dataset from the VR-TSST study
provides a structured session:

    Fix0 (60 s) → Fix1 (60 s) → Forest0 (180 s) → Task0 (180 s)
    → Forest1 → Task1 → Forest2 → Task2 → Forest3 → Task3

Seven normalisation strategies were evaluated via 41-fold LOSO
LogisticRegression (C=0.001, balanced, StandardScaler in pipeline) on
41 participants (6 excluded per `config/pretrain_qc.yaml`):

| Strategy      | Mean AUC | Median | Std    | BalAcc | F1     | Online |
|---------------|----------|--------|--------|--------|--------|--------|
| none          | 0.6133   | 0.6301 | 0.1291 | 0.5597 | 0.5193 | yes    |
| pp_zscore     | 0.6357   | 0.6407 | 0.1330 | 0.6049 | 0.6038 | no     |
| forest_only   | 0.5995   | 0.6120 | 0.1537 | 0.5611 | 0.5350 | yes    |
| cumulative    | 0.6309   | 0.6362 | 0.1419 | 0.5869 | 0.5630 | yes    |
| fixation_only | 0.6236   | 0.6332 | 0.1255 | 0.5710 | 0.5289 | yes    |
| calibration   | 0.6321   | 0.6507 | 0.1236 | 0.5779 | 0.5503 | yes    |
| expanding     | 0.6484   | 0.6646 | 0.1480 | 0.6049 | 0.5907 | yes    |

Full per-participant results: `results/test_pretrain/causal_norm_comparison.json`
Script: `scripts/causal_norm_comparison.py`

## Decision

Use **calibration** (fixation + Forest0, ~300 s) as the primary causal
normalisation strategy for all downstream work.

## Rationale

1. **Standard BCI practice.**  A fixed calibration-then-deploy paradigm is
   the established approach in online neurofeedback and adaptive systems.
   Reviewers will recognise it immediately and it requires no novel
   justification.

2. **Lowest cross-participant variance** (std 0.1236) among all online
   strategies, indicating the most consistent performance across
   participants — critical for a deployed system.

3. **Conceptually clean baseline.**  The normalisation reference consists
   entirely of resting-state data (fixation crosses + passive forest
   viewing).  There is no task contamination, so the baseline cannot
   absorb the workload-related variance we are trying to detect.

4. **Simple to deploy.**  Compute feature mean/std from the calibration
   phase, freeze them, and classify indefinitely.  No bookkeeping of
   expanding windows or prior-block buffers.

5. **Marginal AUC gap to expanding.**  The difference (0.6321 vs 0.6484,
   Δ = 0.016) is well within noise (std ≈ 0.13) and does not justify the
   expanding strategy's conceptual and practical costs.

### Why not expanding?

Expanding includes prior task epochs — i.e. data containing the very
workload variations being classified — in the normalisation baseline.
This creates two problems:

- The baseline drifts toward the task distribution, potentially
  attenuating the signal of interest.
- A reviewer could reasonably argue the normalisation reference is
  contaminated by the dependent variable.

The AUC advantage likely reflects the larger sample size producing more
stable z-score estimates, not a genuinely better normalisation strategy.

### Why not pp_zscore?

pp_zscore is non-causal (uses future data) and therefore cannot be
deployed online.  It serves as an upper-bound reference only.

## Consequences

- Positive: clean, defensible methods section ("participants completed a
  ~5 min calibration phase; feature statistics were frozen and applied to
  all subsequent task epochs").
- Positive: simplest online implementation.
- Positive: most stable performance across participants.

- Negative: small AUC concession vs expanding (~0.016), unlikely to be
  practically significant.
- Negative: calibration quality depends on the first ~300 s of data;
  poor early data quality could affect an entire session.

Mitigation: report expanding (and all other strategies) as comparisons
in the thesis/paper to demonstrate the decision was empirically informed.
