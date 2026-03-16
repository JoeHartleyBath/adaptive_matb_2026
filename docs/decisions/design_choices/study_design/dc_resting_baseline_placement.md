# Design choice: Resting EEG baseline before familiarisation

**Decision**  
Add a 2–3 minute resting fixation-cross phase at the very start of the session (after EEG setup/impedance, before familiarisation or any task exposure). This resting EEG recording provides the per-participant normalisation baseline (μ/σ) used by the real-time MWL estimator.

**Session order (updated)**  
```
Setup + impedance → Resting fixation (2–3 min) → Familiarisation → Practice (3×5 min)
→ Staircase → Calibration (2×9 min) → Fine-tune → Adaptation run
```

**Rationale**  
- ADR-0004 (calibration normalisation) was validated using VR-TSST resting data (~300 s of fixation crosses + passive forest viewing) collected *before* any task blocks. To match, the MATB baseline must also precede all task exposure.
- A post-practice or post-staircase baseline would be contaminated by prior task engagement — it would not be genuine resting-state EEG.
- Standard BCI practice: compute normalisation statistics from a clean resting period, freeze them, then classify indefinitely (ADR-0004 §Rationale point 1).
- Minimal session cost: adds only 2–3 minutes to total session time.
- Doubles as a "confirm EEG signal quality" checkpoint before committing to the full session.

**Alternatives considered**  
- **Resting baseline after practice/staircase:** participant has already done 15+ minutes of effortful multitasking — not genuinely resting; violates ADR-0004 rationale.
- **Two baselines (before + after staircase), averaged:** not how ADR-0004 was validated; averaging resting and post-task baselines creates a hybrid with unknown properties; adds session time and analysis complexity.
- **Use staircase EEG data as baseline:** contaminated by task activity; violates the "no task contamination" principle of ADR-0004 (§Rationale point 3).
- **Use group-level norm stats from VR-TSST:** ignores individual neural characteristics; cross-study transfer introduces unknown bias.
- **Use first LOW calibration block as baseline:** still task data (tracking, monitoring, etc. are active even at LOW difficulty).

**Implications**  
- A fixation-cross scenario must be created for OpenMATB (or run outside OpenMATB as a simple screen + EEG recording).
- The `norm_stats.json` artefact consumed by `mwl_estimator.py` will be computed from this resting recording's features (μ/σ of the 54-feature vector).
- A script or pipeline step is needed to extract features from the resting XDF and save `norm_stats.json`.
- This phase must be included in the operator checklist and session documentation.

**Status**  
Final

**References**  
- Internal: `docs/decisions/ADR/ADR-0004-causal-normalisation-strategy.md`
- Internal: `docs/pilot/pilot_study_spec.md` (VR-TSST session structure)
- Internal: `src/mwl_estimator.py` (consumes `norm_stats.json`)
- Internal: `src/ml/pretrain_loader.py` → `calibration_norm_features()` (offline equivalent)
