# Design decision index

This index lists all design choice documents for the project.
Use the **Status** column to distinguish stable vs pilot-tuned/proposed choices.

Open or unresolved questions must live in `open_decisions.md`.

---


## Design choices

| ID | Title | Category | Status | Link |
|----|-------|----------|--------|------|
| DC-01 | Multitask MATB(-II)-style paradigm | Study design | Final (foundational) | [dc_study_paradigm_multitask_matb.md](study_design/dc_study_paradigm_multitask_matb.md) |
| DC-02 | Full multitask MATB(-II) task set | Study design | Final | [dc_task_set.md](study_design/dc_task_set.md) |
| DC-03 | Workload manipulation via difficulty and event rate | Study design | Final (foundational) | [dc_workload_manipulation.md](study_design/dc_workload_manipulation.md) |
| DC-04 | Use three workload levels (low/moderate/high) | Study design | Final | [dc_three_workload_levels.md](study_design/dc_three_workload_levels.md) |
| DC-05 | Include a practice phase before calibration blocks | Study design | Final | [dc_include_practice_phase.md](study_design/dc_include_practice_phase.md) |
| DC-06 | Practice structure prior to calibration blocks | Study design | Proposed (pilot-tuned) | [dc_training_structure.md](study_design/dc_training_structure.md) |
| DC-07 | Full-study calibration structure (2×9 min; 1-min blocks) | Study design | Final (locked; implement after Pilot 1) | [dc_full_study_calibration_structure.md](study_design/dc_full_study_calibration_structure.md) |
| DC-08 | RBF hyperparameter tuning has reached ceiling with current features | Modelling | Final | [dc_rbf_hyperparameter_ceiling.md](modelling/dc_rbf_hyperparameter_ceiling.md) |
| DC-09 | LogReg hyperparameter plateau (K=30, C=0.001) | Modelling | Final | [dc_logreg_hyperparameter_plateau.md](modelling/dc_logreg_hyperparameter_plateau.md) |
| DC-10 | Warm-start weak L2 (C=0.1) is the best LogReg personalisation | Modelling | Final | [dc_logreg_personalisation_comparison.md](modelling/dc_logreg_personalisation_comparison.md) |
| DC-11 | 120 s per label is best for offline TSST controller simulation | Adaptation | Final (corrected 2026-03-16) | [dc_calibration_duration.md](adaptation/dc_calibration_duration.md) |
| DC-12 | Resting EEG baseline before familiarisation | Study design | Final | [dc_resting_baseline_placement.md](study_design/dc_resting_baseline_placement.md) |
