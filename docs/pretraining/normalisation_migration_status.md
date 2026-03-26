# Normalisation migration status

**Date:** 2026-03-17  
**Status:** Done

---

## The problem

The offline modelling pipeline applies **per-participant z-scoring** to the full feature matrix (all epochs from the entire session) before any train/test split. This uses the participant's global mean and standard deviation — including future epochs — to normalise each epoch.

In online deployment (live MATB adaptive automation), future data does not exist. A classifier trained offline with look-ahead normalisation will see differently-scaled inputs at inference time, making offline AUC an unreliable proxy for real-world performance.

### Quantified impact

| Condition | Mean AUC | Median AUC | BalAcc | F1 |
|-----------|----------|------------|--------|----|
| With pp z-score (current) | **0.6725** | **0.7000** | 0.6313 | 0.6306 |
| Without pp z-score | 0.6116 | 0.6168 | 0.5582 | 0.5119 |
| **Delta** | **−0.061** | **−0.083** | −0.073 | −0.119 |

Removing z-scoring entirely drops AUC by 6 points. Normalisation matters — the question is *which* normalisation method can be replicated online.

### Why pp z-scoring is not a LOSO leakage problem

The z-scoring is within-participant: each participant's scaler uses only their own epochs. The held-out participant in LOSO is a different person, so no cross-participant information leaks. The issue is purely temporal: the offline scaler sees the participant's entire session, which an online system cannot.

---

## What we have done so far

### 1. Identified the problem (audit)

Audited the preprocessing and feature extraction pipeline. Found that:

- **Pipeline-level `StandardScaler`** (inside the sklearn `Pipeline`) is fitted only on training participants — this is correct and online-compatible.
- **Per-participant z-scoring** in `personalised_logreg.py` (line ~845) uses `StandardScaler().fit_transform()` across all epochs for each participant — this uses future data and is not online-compatible.

### 2. Ablation: removed pp z-scoring entirely

Added `--no-pp-zscore` flag to `scripts/rbf_ablation_loso.py` and re-ran the full 28-fold LOSO (original 28-participant dataset, old QC exclusion set). Results above. Documented in `docs/decisions/design_choices/modelling/dc_rbf_hyperparameter_ceiling.md`.

### 3. Redesigned the dataset export format

The old monolithic `dataset.h5` stored pre-windowed epochs only (task blocks). This made it impossible to experiment with baseline-derived normalisation because the baseline data (forest relaxation, fixation cross) was discarded at export time.

**Old format:** Single `dataset.h5` (~14 GB) with `participants/{pid}/epochs` and `participants/{pid}/labels`. Pre-windowed 2 s epochs, gzip-compressed.

**New format:** Per-participant continuous HDF5 files in `output/matb_pretrain/continuous/{pid}.h5` (~230–265 MB each, 28 files). Each file contains:

| Dataset | Shape | Content |
|---------|-------|---------|
| `/eeg` | `(n_samples, 128)` | Continuous preprocessed EEG (0.5–40 Hz causal BP, 50 Hz notch, CAR, 128 Hz) |
| `/task_onsets`, `/task_offsets` | `(4,)` | Sample boundaries for 4 task blocks |
| `/task_labels` | `(4,)` | 0=LOW, 2=HIGH per block |
| `/task_block_order` | `(4,)` | Temporal order 0–3 |
| `/forest_onsets`, `/forest_offsets` | `(4,)` | Sample boundaries for Forest1–4 |
| `/fixation_onsets`, `/fixation_offsets` | `(2,)` | Sample boundaries for fixation cross blocks |

Windowing is now deferred to analysis time, meaning any downstream script can choose its own window size, step, warmup, and — critically — its own normalisation strategy using whatever baseline data it needs.

**Script:** `pipelines/14_matb_pretraining_export/export_matb_pretrain_dataset.py`

### 4. Updated QC exclusions

Synchronised `config/pretrain_qc.yaml` with the 19-element exclusion set used across all scripts:

- 9 × EEG_QUALITY failures
- 10 × NO_SIGNAL failures
- Effective included: 28 participants at this stage (P14 also fails at load → 28 files written)

> **Note:** QC exclusion list was subsequently revised (2026-03-13). Under the updated criteria, **40 participants** are included. See `config/pretrain_qc.yaml` for the current set.

### 5. Built all continuous files

Full rebuild completed. P14 skipped (load failure). Manifest generated at `output/matb_pretrain/continuous/manifest.json`. Current count: 40 included participants.

### 6. Created shared windowing utility

Created `src/ml/pretrain_loader.py` — a `PretrainDataDir` class that:

- Discovers available PIDs from the continuous directory
- Opens per-participant `.h5`, reads segment boundaries, uses existing `slice_block()` + `extract_windows()` to produce epoch arrays
- Provides `load_task_epochs(pid)` → `(epochs, labels, block_idx)`, plus `load_forest_epochs()` and `load_fixation_epochs()`
- Uses `step_s=0.5` default (matching VR-TSST analysis convention)
- Validated: P01 → 1188 task epochs `(1188, 128, 256)`, labels `{0, 2}` ✓

This is the single adapter layer between the new format and all downstream scripts.

---

## Completed (steps 7–11)

All remaining steps were completed. The final decision is documented in
`docs/decisions/ADR/ADR-0004-causal-normalisation-strategy.md`:

- Pipeline updated to read from the continuous per-participant HDF5 format.
- QC exclusion list revised: current set has 6 excluded participants
  (see `config/pretrain_qc.yaml`) plus P14 (load failure) → **40 included**.
- Seven causal normalisation strategies compared via 40-fold LOSO
  (`scripts/causal_norm_comparison.py`). Results in
  `results/test_pretrain/causal_norm_comparison.json`.
- **Decision: calibration-based normalisation** (fixation + Forest0, ~300 s).
  Lowest cross-participant AUC variance (std 0.1236) among online-compatible
  strategies; clean resting-state baseline with no task contamination.

---

## File reference

| File | Role |
|------|------|
| `pipelines/14_matb_pretraining_export/export_matb_pretrain_dataset.py` | Continuous per-participant HDF5 exporter |
| `src/ml/pretrain_loader.py` | Shared loader: continuous HDF5 → windowed epochs |
| `src/eeg/eeg_windower.py` | Low-level windowing primitives (`extract_windows`, `slice_block`) |
| `scripts/personalised_logreg.py` | Feature extraction + cache builder + evaluation |
| `scripts/rbf_ablation_loso.py` | RBF SVM ablation (pp z-score flag added) |
| `config/pretrain_qc.yaml` | QC exclusion list (single source of truth) |
| `docs/decisions/design_choices/modelling/dc_rbf_hyperparameter_ceiling.md` | Full ablation results + pp z-score section |
| `results/test_pretrain/rbf_ablation_no_pp_zscore.json` | Per-fold results without pp z-scoring |
| `output/matb_pretrain/continuous/manifest.json` | Build manifest (40 included participants) |
