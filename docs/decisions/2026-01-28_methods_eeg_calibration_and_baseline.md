# Methods decisions — EEG calibration and baseline segments (v0)

Status: proposed (method decision record)

Last updated: 2026-01-28

Purpose: document the calibration/baseline procedure used to compute participant-specific normalization statistics (and/or baseline MWL), including timing, instructions, and how it is aligned to EEG windows.

This decision is required by the input contract:
- [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md) (baseline/reference segments; normalization source)

Related pending note:
- [docs/decisions/PENDING_DECISIONS_INFERENCE.md](PENDING_DECISIONS_INFERENCE.md)

---

## Decision 1: Whether to include explicit calibration blocks

**Decision statement:** The protocol MUST/SHOULD include explicit calibration (baseline/reference) segments before calibration task blocks.

**Why this matters (replication):**
- Participant-specific normalization depends on baseline/reference windows.
- Without a declared baseline segment, “normalization_source” becomes ambiguous and may drift across analyses.

**Options (to decide):**
- Option A: No explicit calibration; use population stats only.
- Option B: One baseline segment (e.g., eyes open rest) of fixed duration.
- Option C: Multiple baseline segments (e.g., eyes open, eyes closed, fixation) to estimate stability.
- Option D: Task-based calibration (e.g., very low workload “static” MATB segment) rather than rest.

**Rationale (fill in / confirm):**
- [ ] Why baseline is needed for this study’s modeling goals.
- [ ] Why rest vs task-based baseline.

**Replication record (must capture):**
- [ ] Exact baseline segment definitions (instructions shown; eyes open/closed; posture).
- [ ] Exact durations and number of segments.
- [ ] Markers for baseline start/end in the same timebase as EEG.

---

## Decision 2: If using “3 × 5-minute calibration (static) rounds”, define exactly what that means

**Decision statement (if adopted):** Calibration consists of 3 segments of duration 5:00 each, under defined conditions.

**Open questions to resolve before adopting:**
- [ ] What tasks (if any) run during “static” calibration?
- [ ] Are workload parameters fixed to LOW, or are tasks paused but display remains?
- [ ] Are there breaks between calibration segments?
- [ ] Are segments eyes-open fixation / eyes-closed / etc.

**Rationale prompts (fill in):**
- [ ] Why 3 rounds (test-retest reliability; reduce variance).
- [ ] Why 5:00 (enough windows for stable stats at the chosen window+step).

**Verification/acceptance criteria (suggested):**
- [ ] Minimum valid-window count per calibration segment is satisfied.
- [ ] Calibration-derived μ/σ are finite, stable, and reproducible.

---

## Implementation hooks (where this would be implemented)

- Scenario support (if calibration is run inside OpenMATB): would require a dedicated scenario and deterministic markers.
- Wrapper orchestration: [src/python/run_openmatb.py](../../src/python/run_openmatb.py) (playlist + manifests).

---

## Next concrete step

- Decide baseline segment definition and duration, then update:
  - [docs/contracts/mwl_eeg_input_contract.md](../contracts/mwl_eeg_input_contract.md) (`TBD_BASELINE_REFERENCE_SEGMENTS_V0` and related fields)
  - pilot protocol docs (so operators can run it consistently)
