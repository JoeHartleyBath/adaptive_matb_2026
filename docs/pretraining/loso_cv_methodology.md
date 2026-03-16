# LOO-CV Pretraining Validation — Methodology & Known Issues

**Date:** March 2026  
**Status:** First full run in progress (44 folds)

---

## What this experiment measures

A leave-one-subject-out (LOO) cross-validation of an EEGNet pretraining pipeline using VR-TSST EEG data as a proxy dataset, prior to collecting real MATB data.

Each fold:
1. **Pretrain** EEGNet on 43 participants (LOW=0, HIGH=2, binary MWL classification)
2. **Fine-tune** the pretrained model on a subset of the held-out participant's data
3. **Evaluate** on a held-out test split of the same participant

The goal is not to maximise AUC — it is to confirm the pipeline works end-to-end and that pretrained features transfer across participants at above-chance levels.

---

## Dataset

- **Source:** VR-TSST EEG, exported to `dataset.h5`
- **Participants:** 44 (P02, P08, P46 excluded — QC failures)
- **Windows per participant:** 2372 (128Hz, 2s, 128 channels)
- **Labels:** LOW=0, HIGH=2 (remapped to 0/1 inside pipeline)
- **Design:** Block-structured — 4 × 3-minute continuous conditions

---

## Pipeline components

| Component | Detail |
|---|---|
| Model | EEGNet (F1=8, D=2, 128ch, 256 samples) |
| Normalisation | Per-epoch z-score in `MwlDataset`; `InstanceNorm2d` inside EEGNet |
| Loss | `CrossEntropyLoss` (unweighted — dataset is balanced) |
| Optimiser | Adam, lr=1e-3 |
| Scheduler | `ReduceLROnPlateau`, patience=10, factor=0.5 |
| Early stopping | patience=20 epochs, monitor val_loss |
| Val split (pretrain) | Last 15% of participant list (alphabetical) |
| Fine-tune modes | `none`, `head_only`, `late_layers`, `full` |

---

## Known methodological issues (first run)

These issues were identified during the first run. Results from this run should be interpreted with caution. A corrected re-run is planned.

### 1. Temporal test split — **major**
The fine-tuning test set was the **last 20% of epochs per class (temporal tail-slice)**.  
For block-structured data, the end of each condition block is systematically different from the middle (fatigue, habituation, drift). This biases the test set and inflates between-subject AUC variance.

**Fix implemented (not yet run):** replaced with stratified random split per class, seeded for reproducibility (`_stratified_random_split` in `test_pretrain_pipeline.py`).

### 2. Double normalisation — **minor**
`MwlDataset` applies per-epoch z-score AND `InstanceNorm2d` normalises again inside EEGNet. This is redundant but not harmful — it reduces the model's ability to use amplitude as a feature.

**Current state:** double normalisation retained for now. To be revisited after corrected LOO run.

### 3. Val participant split is alphabetical — **minor**
Validation participants during pretraining are the last ~15% alphabetically (P40–P44). If these participants are systematically different (later recording sessions, equipment drift), the model's early stopping signal is biased.

**Planned fix:** switch to seeded random participant-level val split.

---

## How to interpret first-run results

| Metric | What it means |
|---|---|
| `none` AUC | Zero-shot transfer — pretrained features without any subject-specific adaptation |
| `head_only` AUC | Classifier retrained, encoder frozen |
| `late_layers` AUC | Last EEGNet block + classifier retrained |
| `full` AUC | Full fine-tune on subject's data |
| Mean AUC ≈ 0.50 | At chance — expected given temporal split issue |
| High between-subject variance | Partly real (EEG is noisy), partly artefact of biased test split |

**Do not draw conclusions about participant-level decodability from this run.**  
The corrected run (stratified random split) is required before interpreting per-participant AUC.

---

## Planned corrected run

1. Fix temporal split → stratified random split ✅ (code written, not yet run)
2. Fix val participant split → seeded random split (one line change, not yet made)
3. Re-run fine-tuning only for all 44 folds (pretraining weights are unaffected)
4. Re-run `check_mwl_dataset.py` and cross-reference AUC with data quality checks