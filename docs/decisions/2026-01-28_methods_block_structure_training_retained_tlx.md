# Methods decisions — block structure, training, calibration blocks, TLX (v0)

Status: draft (method decision record)

Last updated: 2026-01-28

Purpose: capture decisions about the session’s block structure (training vs calibration), block durations, breaks, TLX placement, and pause/abort policy.

Primary authority: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)

---

## Decision 1: Training uses 3 × 5:00 blocks at Low → Moderate → High

**Decision statement (exact v0):** Training blocks are T1=LOW (5:00), T2=MODERATE (5:00), T3=HIGH (5:00), with 1:00 breaks between.

**Source of truth:** The exact structure is defined in [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).

**Rationale (partly specified; fill in / confirm):**
- Protocol goal: stabilize task proficiency before calibration data.
- [ ] Why 3 blocks rather than fewer/more (learning curve, task-set breadth).
- [ ] Why fixed Low→Moderate→High order (safety/learning, reducing early overload).
- [ ] Why 5:00 duration (tradeoff: stable EEG windows vs fatigue/time).

**Alternatives considered (fill in):**
- [ ] Longer single training block
- [ ] Counterbalanced training order
- [ ] Training until performance criterion met

**Replication record (must capture):**
- [ ] Exact scenario segment that implements training (scenario file path + commit).
- [ ] Any operator instructions (what counts as failure to proceed to calibration blocks).

---

## Decision 2: calibration data are 3 × 5:00 blocks at Low/Moderate/High in counterbalanced order

**Decision statement (exact v0):** calibration blocks B1–B3 are each 5:00, with workload level order determined by `seq_id` (Latin-square), and 1:00 breaks between blocks.

**Source of truth:** [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)

**Rationale (fill in / confirm):**
- [ ] Why 3 calibration blocks (within-subject comparison; training separated).
- [ ] Why 5:00 calibration blocks (EEG window coverage; stationarity; participant tolerance).

**Alternatives considered (fill in):**
- [ ] More blocks with shorter durations
- [ ] Fewer blocks with longer durations

**Replication record (must capture):**
- [ ] The exact counterbalancing scheme and how `seq_id` is assigned.
- [ ] Verification that calibration blocks have deterministic markers in CSV/LSL.

---

## Decision 3: NASA-TLX is administered after each calibration block, but not during training

**Decision statement (exact v0):** TLX is self-paced immediately after B1/B2/B3, and not administered during training.

**Rationale (partly specified; fill in / confirm):**
- Existing rationale in spec: reduce interruptions while the participant is still learning controls.
- [ ] Why TLX after each calibration block (temporal specificity vs burden).

**Implementation evidence:**
- Spec: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)
- Build plan: [docs/pilot/PILOT_BUILD_PLAN_V0.md](../pilot/PILOT_BUILD_PLAN_V0.md)

**Replication record (must capture):**
- [ ] Questionnaire asset used (e.g., `nasatlx_en.txt`) and any modifications.
- [ ] Whether the questionnaire requires interaction with all sliders before continuing.

---

## Decision 4: No pausing during calibration blocks; interruptions handled via ABORT + restart

**Decision statement (exact v0):** No pause/resume within B1–B3; if interruption occurs, abort and restart.

**Why this matters:** Avoids ambiguous labeling windows that straddle pauses; keeps calibration blocks comparable.

**Implementation evidence:**
- Policy is specified in [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- Rationale for per-block process restarts: [docs/decisions/ADR-0002-wrapper-run-per-block-for-clean-task-state.md](ADR-0002-wrapper-run-per-block-for-clean-task-state.md).

**Replication record (must capture):**
- [ ] Abort criteria and operator actions.
- [ ] How ABORT is encoded and detected in logs.

---

## Open items / TBD

- [ ] Add the explicit “why 5:00?” rationale (method section wording) once final.
- [ ] Add the “calibration/baseline EEG segment” decision here once the calibration design is finalized (currently tracked in [docs/decisions/PENDING_DECISIONS_INFERENCE.md](PENDING_DECISIONS_INFERENCE.md)).
