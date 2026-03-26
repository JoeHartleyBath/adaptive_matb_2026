# Pilot Verification Plan

Status: draft
Last updated: 2026-03-17

Purpose: defines how each phase of the full study session is piloted with real participants — what to test, how to verify it, how many participants are needed, and the gate criterion before proceeding to the full study.

Related documents:
- [pilot_study_spec.md](pilot_study_spec.md) — session structure and protocol
- [pilot_build_plan.md](pilot_build_plan.md) — build roadmap
- [pilot_session_checklist.md](pilot_session_checklist.md) — physical lab setup

---

## Participant IDs

Pilot participants use the namespace `PPILOT01`–`PPILOT06`. These are pre-registered in `config/participant_assignments.yaml` with counterbalanced sequences and condition orders, separate from the study namespace (`P001` onward).

This keeps the study participant slots clean — `P001` is always the first study participant regardless of how many pilots were run. Do not use `P001`–`P009` for pilots.

Assignments:

| ID | Sequence | Condition order |
|----|----------|-----------------|
| PPILOT01 | SEQ1 | adaptation → control |
| PPILOT02 | SEQ2 | control → adaptation |
| PPILOT03 | SEQ3 | adaptation → control |
| PPILOT04 | SEQ1 | control → adaptation |
| PPILOT05 | SEQ2 | adaptation → control |
| PPILOT06 | SEQ3 | control → adaptation |

To add further pilot IDs if needed:

```powershell
python scripts/generate_scenarios/generate_participant_assignments.py \
    --participant-ids PPILOT07 PPILOT08 --sequences SEQ1 SEQ2
```

---

## Orchestration

Run the existing orchestrator for all pilot sessions:

```powershell
# Stage A (no physio)
python src/run_full_study_session.py --participant PPILOT01 --group-model-dir <path>

# Stage B (full physio)
python src/run_full_study_session.py --participant PPILOT03 --group-model-dir <path> --pilot1 --labrecorder-rcs
```

Do **not** use separate per-phase scripts. Running the full orchestrator validates the inter-phase plumbing (d_final handoff, path derivation, artefact discovery) as well as each phase in isolation.

---

## Staging

Two stages, ordered cheapest-to-most-complex:

| Stage | Pilots | Physio | Phases covered |
|-------|--------|--------|----------------|
| A | 1–2 | None | 1, 2, 3, 4 (behavioural only), 7 |
| B | 3–6 | EEG + Shimmer + Polar | 4 (with XDFs), 5, 6 |

Stage A flags task/scenario problems before the physio setup adds complexity.
Stage B adds `--pilot1 --labrecorder-rcs` to the orchestrator call.

---

## Phase 1 — Practice (4 familiarisation scenarios)

### How to test
Observe the participant live throughout. At the end of T3 (high workload), ask them to name all four tasks without prompting. Verbal think-aloud during T3 is useful if it doesn't disrupt performance.

### How to verify
- All 4 scenario CSVs and manifests written without error.
- T3 tracking performance (from CSV) — eyeball for a continued learning curve vs. stabilisation. If RMSE is still declining sharply at the end of T3, the practice block may be too short.
- Qualitative: participant can describe all four tasks unprompted after T3.

### N needed
2–3 pilots. This is a qualitative saturation question (clarity of instructions, ramp steepness), not a statistical one.

### Gate
No task confusion in debrief. T3 performance looks stable at block end. All 4 CSVs written cleanly. No scenario crashes.

---

## Phase 2 — Staircase calibration

### How to test
Runs automatically via `adaptation_skeleton.txt` with `--adaptation`. d_final is extracted by Phase 3 via `extract_d_final()`.

### How to verify
- `tests/verification/verify_adaptation_session.py` exits 0.
  - Key internal checks: ≥5 adaptation rows logged, staircase triggered, d_final extractable.
- d_final distribution across pilots: mean ± SD. Flag any floor (<0.10) or ceiling (>0.90) hits.

### N needed
3–4 pilots. A single d_final value is not enough to judge convergence quality — need a distribution.

### Stats
Descriptive only: mean, SD, floor/ceiling count.

### Gate
d_final in [0.10, 0.90] for ≥3 out of 4 pilots. Verifier exits 0. No crash.

---

## Phase 3 — Scenario generation

### How to test
Runs automatically after Phase 2. Generates calibration (C1, C2) and adaptation scenarios from the pilot's d_final.

### How to verify
- All 3 files written to `experiment/scenarios/` without error.
- Run `tests/verification/verify_pilot_scenarios.py` on the generated files: checks marker payloads, token substitution, segment durations, event counts.

### N needed
Verify for every pilot's d_final — purely mechanical, not a participant-number question.

### Gate
All generated files pass scenario syntax checks. No token substitution gaps. Verified for the full d_final range seen across pilots.

---

## Phase 4 — Calibration runs (2 × 9-min)

### How to test
Run both counterbalanced calibration scenarios. Collect NASA-TLX on paper after each block. Stage A: behavioural only. Stage B: with LabRecorder + XDFs.

### How to verify
- `tests/verification/verify_pilot.py` on each CSV + manifest (structural checks).
- Correct `calibration/BN/LEVEL/START` and `END` markers present in each CSV.
- TLX total scores per workload level (LOW, MODERATE, HIGH) across pilots.

### Stats
Friedman test on TLX totals across the 3 levels (repeated measures, non-parametric).
- At N=5: need Kendall's W ≈ 0.5 for p < .05. Set expectations accordingly.
- If W < 0.3, the workload manipulation is likely too weak — revisit event rates.
- Fallback at N=4: Wilcoxon signed-rank LOW vs HIGH (more robust at small N).

### N needed
4–5 for the Friedman test to be informative.

### Gate
TLX scores separate across levels (at minimum LOW < HIGH trending). All markers present in CSV. No crashes.

---

## Phase 5 — Model calibration (warm-start LogReg)

### How to test
Run `calibrate_participant_logreg.py calibrate` on each pilot's two 9-min calibration XDFs from Phase 4.

### How to verify
- Deployment artefacts written and loadable: `pipeline.pkl`, `selector.pkl`, `norm_stats.json`.
- Per-participant AUC: use the two calibration runs as a 2-fold split (train on one, test on the other). Compute AUC + 95% binomial CI.
- Flag any participant where the CI lower bound ≤ 0.5 (not above chance).
- Calibration runtime: should complete in <5 min.

### Stats
AUC per pilot + 95% binomial CI. No inferential test needed at this N — decision is per-participant.

### ⚠️ Cross-paradigm risk
The group model was trained on a different paradigm. If AUC ≤ 0.55 for ≥3 out of 4 pilots, the warm-start is likely harmful. Prepare a cold-start fallback (train from scratch on participant calibration data only) before entering the full study — do not wait until mid-study to discover this.

### N needed
3–4 pilots with physio (Stage B only).

### Gate
AUC > 0.60 and CI lower bound > 0.5 for ≥3 out of 4 pilots. All artefacts loadable. Calibration completes in <5 min.

---

## Phase 6 — MWL adaptation condition

### How to test
Run `--mwl-adaptation` with the pilot's deployed model from Phase 5. Debrief the participant on automation awareness after the block.

### How to verify
- `tests/verification/verify_mwl_adaptation_session.py` exits 0.
  - Key internal checks: 0 cooldown violations (14.5 s threshold), schema correct.
- Toggle count per session: flag if 0 (MWL never crossed threshold — model not discriminating in real time).
- Signal quality: flag if >50% of audit ticks have quality < 0.5 (real-time estimation unreliable).
- `scripts/analyse_adaptation_session.py` output written cleanly.
- Qualitative debrief: could the participant notice the automation? Did it feel appropriate or jarring?

### N needed
3–4 pilots with physio (Stage B).

### Gate
Verifier exits 0. ≥1 toggle fired per session. 0 cooldown violations. Participant can describe automation behaviour in debrief.

---

## Phase 7 — Control condition

### How to test
Run the same 8-min adaptation scenario without `--mwl-adaptation`. Covered by Stage A runs automatically.

### How to verify
- CSV and manifest written; scenario runs to completion.
- Performance metrics (tracking RMSE, sysmon hit rate) in a plausible range.
- All expected markers present in CSV.

### N needed
Covered by Stage A — no additional participants needed specifically for this phase.

### Gate
Scenario completes without crash. Performance metrics plausible. No missing markers.

---

## Summary gate: ready for full study

All of the following must be true before running the full study:

1. Phase 1: No task confusion in debrief across ≥2 pilots.
2. Phase 2: d_final in [0.10, 0.90] for ≥3/4 pilots; verifier passing.
3. Phase 3: Generated scenarios pass syntax checks for full observed d_final range.
4. Phase 4: TLX separation credible (LOW < HIGH at minimum); markers correct.
5. Phase 5: AUC > 0.60 (CI > 0.5) for ≥3/4 pilots; cross-paradigm risk resolved (warm-start or cold-start fallback confirmed).
6. Phase 6: Verifier passing; toggles firing; 0 cooldown violations; participant debrief coherent.
7. Phase 7: Control scenario completes cleanly.
