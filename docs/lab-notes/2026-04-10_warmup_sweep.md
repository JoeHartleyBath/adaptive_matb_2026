# Lab notes — 2026-04-10 — Training block warm-up sweep (PSELF S005)

## Goals

- Evaluate whether discarding more of the early transient in each calibration block
  improves model quality (AUC, CV balanced accuracy).
- Understand the effect on adaptation behaviour (assist ON%, transitions).
- Inform the choice of `WARMUP_S` for the next participant run.

---

## Context

`slice_block()` in `src/eeg/eeg_windower.py` unconditionally discards the first `WARMUP_S`
seconds of every calibration block before windowing. The rationale is two-fold: IIR filter
transient settling (~5 s) and a habituation / task-engagement period.

This sweep re-examines that choice by treating `WARMUP_S` as a free parameter from 0 → 30 s.

Data foundation: `cal_c1` + `cal_c2`, with `block_01` recovered (start inferred from END marker
− 59 s, per `_tmp_retrain_with_block01.py`).

---

## Methods

Script: `scripts/_tmp_warmup_sweep.py`

Key design decisions:

1. **Binary classifier (LOW vs HIGH)** — S005's MODERATE and HIGH blocks had identical task
   difficulty due to a staircase calibration error. A 3-class SVM conflates the indistinguishable
   MODERATE/HIGH pair with the real LOW/HIGH signal. MODERATE windows were discarded for this
   analysis. The production pipeline remains 3-class.

2. **AUC is in-sample** — Youden-J threshold and AUC are derived from the full training set.
   CV balanced accuracy (`cross_val_score`, 5-fold stratified) is the only genuinely held-out
   metric. In-sample AUC values are upper bounds.

3. **`slice_block` bypassed in sweep script** — Direct array slicing used so that the sweep's
   `warmup_s` is the *sole* warmup applied (normal production code adds `WARMUP_S` via
   `slice_block`; failing to account for this caused a misinterpretation in the first sweep run).

---

## Results

| warmup_s | n/class (LOW) | CV BA | AUC (in-sample) | Youden thr |
|---|---|---|---|---|
| 0  | 1315 | 0.694 | 0.782 | 0.494 |
| 5  | 1196 | 0.728 | 0.803 | 0.506 |
| 10 | 1081 | 0.742 | 0.802 | 0.515 |
| 15 |  962 | 0.754 | 0.825 | 0.548 |
| 20 |  847 | 0.763 | 0.831 | 0.603 |
| 25 |  728 | 0.768 | 0.850 | 0.512 |
| 30 |  613 | 0.782 | 0.855 | 0.509 |

Figure: `results/figures/warmup_sweep_pself_s005_binary.png`

Key observations:

- CV BA and AUC improve monotonically. The gain is gradual (+8.8 pp CV BA over 0→30 s),
  not a sharp step-change. Likely reflects a mix of filter transient (~5 s) and slower
  habituation throughout the first ~30 s of each block.
- The MODERATE ≈ HIGH confound means the improvement may be larger here than in a clean session.
- Adaptation behaviour was insensitive to warmup: all models predicted the adaptation XDF as
  predominantly HIGH (mean P(HIGH) ≈ 0.70–0.83), causing the scheduler to latch ON immediately.
  This is a session-level finding (genuinely high-workload adaptation task) not a warmup artefact.

---

## Decisions

- **`WARMUP_S` changed from 30 → 15 s** (`src/eeg/eeg_windower.py`).
  Rationale: 15 s retains most of the quality gain while roughly doubling usable training
  windows per block (44 s usable vs 29 s at 30 s warmup; ~176 vs ~116 windows/block at 128 Hz).
  The remaining CV BA gap (0.754 vs 0.782) is modest and based on a confounded session.
  To be re-evaluated after the next clean calibration run.
- No change to the production 3-class pipeline.
- Warmup > 30 s not worth testing: blocks are 59 s; window counts collapse below 30 windows/block.

---

## Issues / risks

- All AUC and threshold values are in-sample. For a defensible threshold, compute AUC on
  out-of-fold predictions (`cross_val_predict` with `method='predict_proba'`) in future sweeps.
- Conclusions are based on a single confounded participant; confirm 15 s choice on next run.

## Next actions

- Run next calibration session with `WARMUP_S = 15`.
- Repeat sweep (or simplified version) on that participant's data against a clean 3-class dataset.
