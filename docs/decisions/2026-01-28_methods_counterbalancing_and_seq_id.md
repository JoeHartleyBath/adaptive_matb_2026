# Methods decisions — counterbalancing and `seq_id` assignment (v0)

Status: draft (method decision record)

Last updated: 2026-01-28

Purpose: document why and how workload order is counterbalanced across participants using `SEQ1|SEQ2|SEQ3`, and what must be recorded for replication.

Primary authority: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)

---

## Decision 1: Counterbalance calibration block order via a 3-level Latin-square

**Decision statement (exact v0):** calibration block order uses three sequences:
- `SEQ1`: LOW → MODERATE → HIGH
- `SEQ2`: MODERATE → HIGH → LOW
- `SEQ3`: HIGH → LOW → MODERATE

Training order is fixed (LOW → MODERATE → HIGH) and does not depend on `seq_id`.

**Rationale (fill in / confirm):**
- [ ] Why counterbalancing is required (order effects, fatigue, learning).
- [ ] Why Latin-square with 3 sequences is sufficient for v0.

**Implementation evidence:**
- Defined in [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- Implemented in wrapper playlist mapping: [src/python/run_openmatb.py](../../src/python/run_openmatb.py).

**Replication record (must capture):**
- [ ] How participants are assigned to sequences (operator rule; balancing target across sample).
- [ ] Where sequence assignment is recorded (operator log + run manifest + encoded markers).

---

## Decision 2: Encode `seq_id` into markers to make stream mixing detectable

**Decision statement:** Every `STUDY/V0/...` marker includes `pid`, `sid`, and `seq` in its payload.

**Rationale:**
- Enables automatic detection of file mix-ups during synchronization and labeling.

**Implementation evidence:**
- Marker payload template in [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- Runtime token substitution is implemented by the wrapper/bootstrap: [src/python/run_openmatb.py](../../src/python/run_openmatb.py).

**Replication record (must capture):**
- [ ] Marker transport choice (CSV only vs CSV+LSL).
- [ ] Any transformations applied to marker payloads (must preserve the `pid/sid/seq` fields).

---

## Decision 3: Use three combined session scenario files (one per `seq_id`)

**Decision statement:** v0 uses three combined session scenarios (`pilot_seq1.txt`, `pilot_seq2.txt`, `pilot_seq3.txt`), each embedding the same training segment then calibration blocks ordered per `seq_id`.

**Rationale (fill in / confirm):**
- [ ] Why use combined session scenarios rather than smaller per-block scenario files.

**Implementation evidence:**
- Spec: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- Related “restart per block” stability decision: [docs/decisions/ADR-0002-wrapper-run-per-block-for-clean-task-state.md](ADR-0002-wrapper-run-per-block-for-clean-task-state.md).

**Replication record (must capture):**
- [ ] Exact scenario file path and commit hash.
- [ ] Verification that the chosen scenario matches the assigned `seq_id`.

---

## Open items / TBD

- [ ] Define the operator assignment rule for `seq_id` (e.g., rotating schedule, blocked randomization, stratification variables).
- [ ] Confirm how `seq_id` is persisted into the manifest for each OpenMATB run.
