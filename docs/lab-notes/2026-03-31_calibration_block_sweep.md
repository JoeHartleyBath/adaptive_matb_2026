# Lab notes — 2026-03-31 — Calibration block count & model sweep

## Goals

- Determine the minimum number of 3-min calibration triplets needed to train a
  reliable personalised MWL model
- Compare logistic regression, SVM-linear, SVM-RBF, LDA, and RF across a range
  of k (SelectKBest) and C values
- Add Youden's J as an additional metric to assess where an optimal HIGH vs
  NOT-HIGH decision threshold lies for each model

## Work done

- Created `scripts/calibration_block_count_sweep.py`
  - Sweeps every combination of n triplets (n=1…6) × a configurable set of
    model configs
  - Verification gate: each triplet must contain exactly {LOW, MODERATE, HIGH}
    or the script aborts with a diagnostic message
  - Norm stats derived from resting-baseline XDF (not LOW-block fallback) —
    matching the real session pipeline
  - Metrics per fit: AUC (binary HIGH vs NOT-HIGH), accuracy @0.5, Youden's J
    with its optimal threshold, P(HIGH) at each level (LOW / MODERATE / HIGH)
  - Outputs: raw CSV (one row per fit) + summary CSV (mean±std per model config
    per n_blocks)
- Three sweeps run on PSELF pilot data:
  - v1: LogReg only (63 fits)
  - v2: LogReg + SVM-linear × k × C (1,260 fits)
  - v3: all models — LogReg, SVM-linear, SVM-RBF, LDA, RF × k × C/gamma (3,780 fits)
- Data: `sub-PSELF ses-S001`, calibration XDFs `cal_c1` + `cal_c2`,
  rest XDF `acq-rest`, adaptation XDF `acq-adaptation_physio_old1`

## Results

### Model ranking at n=6 (all 6 triplets)

| Rank | Model   |  k |    C | AUC   |   J   | J-thr | P(H)\|L | P(H)\|M | P(H)\|H |
|------|---------|---:|-----:|------:|------:|------:|--------:|--------:|--------:|
|  1   | svm_lin | 40 |  1.0 | 0.744 | 0.381 | 0.926 |   0.612 |   0.801 |   0.934 |
|  2   | svm_lin | 54 |  1.0 | 0.738 | 0.357 | 0.895 |   0.667 |   0.797 |   0.949 |
|  3   | svm_lin | 54 |  0.1 | 0.731 | 0.374 | 0.982 |   0.758 |   0.882 |   0.977 |
| 10   | lda     | 54 |   —  | 0.660 | 0.306 | 0.957 |   0.796 |   0.935 |   0.967 |
| 12   | logreg  | 30 |0.003 | 0.658 | 0.284 | 0.680 |   0.596 |   0.733 |   0.788 | ← current deployed |

SVM-RBF and RF did not appear in the top 15. SVM-RBF collapsed (flat kernel in
z-score space); RF underfits the small training sets. LDA matches LogReg on AUC
but shows poor LOW/MODERATE separation (gap only 0.14) which would be a
real-time problem.

### Why svm_lin k=40 C=1.0 is preferred

- Best AUC (+0.086 over current LogReg)
- Best ordinal separation: 0.612 → 0.801 → 0.934 — clean stepwise increase
  across all three levels (not just the binary split the AUC measures)
- Youden's J = 0.381, optimal threshold ≈ 0.93 — scheduler trigger must be
  set high, not at 0.5 (running SVM-lin with threshold=0.50 would fire
  assistance for ~80% of windows)

### Calibration block count — plateau analysis

| Step | svm_lin k=40 C=1.0 ΔAUC | Combo std |
|------|-------------------------:|----------:|
| n=1  | (baseline 0.546)         |   ±0.041  |
| n=2  | +0.033                   |   ±0.061  |
| n=3  | +0.048                   |   ±0.072  |
| n=4  | **+0.061**               | **±0.027** ← reliability plateau |
| n=5  | +0.030                   |   ±0.012  |
| n=6  | +0.025                   |   ±0.000  |

Combo-to-combo std halves sharply at **n=4**. Marginal AUC gain n=4→6 = +0.055.
**Conclusion: 4 triplets (12 min of calibration) is the defensible minimum.**

### Caveats

- Single participant (PSELF), one adaptation recording
- Earlier `sweep_scratch_models.py` flagged SVM-lin P(H)|L=0.68 saturation risk;
  with resting-baseline norm it is 0.612 here — improvement noted
- J-threshold ≈ 0.93 vs legacy scheduler default 0.50 — handled by J-threshold
  export (see implementation below)

---

## Plan: replication after model switch

The model change has been implemented (2026-03-31). Replication plan:

### Runs needed

1. **PSELF, more adaptation recordings** — re-run sweep on
   `acq-adaptation_physio_old2` through `old10` and current `acq-adaptation_physio.xdf`
   to confirm ranking stability across recording sessions.

2. **New participant** — as soon as cal + adaptation recordings are available:
   ```powershell
   $PHYSIO = "C:\data\adaptive_matb\physiology\sub-P001\ses-S001\physio"
   C:\adaptive_matb_2026\.venv\Scripts\python.exe scripts/calibration_block_count_sweep.py `
     --xdf-cal1  "$PHYSIO\sub-P001_ses-S001_task-matb_acq-cal_c1_physio.xdf" `
     --xdf-cal2  "$PHYSIO\sub-P001_ses-S001_task-matb_acq-cal_c2_physio.xdf" `
     --xdf-rest  "$PHYSIO\sub-P001_ses-S001_task-matb_acq-rest_physio.xdf" `
     --xdf-adapt "$PHYSIO\sub-P001_ses-S001_task-matb_acq-adaptation_physio.xdf" `
     --scenario-adapt experiment/scenarios/adaptive_automation_P001_c1_8min.txt `
     --out-csv results/cal_block_sweep_P001_v1.csv
   ```

**Reversion gate**: if SVM-lin k=40 C=1.0 fails to outperform LogReg on ≥2
additional participants, revert `calibrate_participant_logreg.py` and update
the model comment in `run_full_study_session.py`.

---

## Implementation: what changed (2026-03-31)

### B — SVM-linear replaces LogReg in the calibration pipeline

**`scripts/calibrate_participant_logreg.py`:**
- `_CAL_MODEL = "svm_lin"`, `_CAL_K = 40`, `_CAL_C = 1.0` replace the old
  LogReg constants (`_LOGREG_K=30`, `_LOGREG_C=0.003`)
- Classifier is now `SVC(kernel='linear', C=1.0, class_weight='balanced',
  probability=True, random_state=SEED)`
- After fitting, Youden's J threshold is computed from the calibration data
  and written to `model_config.json` alongside the model artefact

**`src/run_full_study_session.py`:**
- Docstring updated to reflect SVM-lin k=40 C=1.0
- Phase 5 reads `model_config.json` and passes `--threshold` to adaptation phase

### C — Participant-adaptive threshold in the scheduler

**`src/adaptation/mwl_adaptation_scheduler.py` / `run_full_study_session.py`:**
- `MwlAdaptationConfig.threshold` is now set per-participant from the
  Youden's J threshold stored in `model_config.json` at calibration time
- The legacy default of 0.50 is no longer used in live runs

### A — 12-min calibration (2 × 6-min) — NOT YET IMPLEMENTED

Requires scenario generator changes first. Tracked separately.

---

## Decisions

- Model switched to SVM-lin k=40 C=1.0 (2026-03-31, PSELF pilot evidence)
- Threshold is now participant-adaptive via Youden's J from calibration data
- Calibration duration remains 2 × 9-min until scenario generator updated

## Issues / risks

- Single-participant evidence — monitor carefully on first real participant
- `SVC(probability=True)` fit is slower (Platt scaling) — check lab fit time
- Hysteresis = 0.02 was tuned at threshold 0.50; at 0.93 it may be too tight —
  review if adaptation toggles unexpectedly rapidly in first participant

## Next actions

- [ ] Re-run sweep on PSELF `old2`–`old10` adaptation recordings
- [ ] Run sweep on first non-PSELF participant when data available
- [ ] Review hysteresis behaviour in first live session
- [ ] Scenario generator: 6-min cal scenario feasibility check
- [ ] If replication fails: open ADR to document decision and revert

## Time spent

- Total: ~3 hours
- Breakdown:
  - Script development + verification gate: ~1.5 h
  - Three sweep runs + result extraction: ~1 h
  - Analysis, documentation, and implementation: ~0.5 h
