# Methods decisions — workload levels and operationalisation (v0)

Status: draft (method decision record)

Last updated: 2026-01-28

Purpose: capture *why* the study uses three workload levels and *how* each level is operationalised (event rate + overlap) while holding the task set constant.

Primary authority: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)

---

## Decision 1: Use three discrete workload levels (LOW / MODERATE / HIGH)

**Decision statement:** The pilot uses three levels (LOW, MODERATE, HIGH) rather than a binary low/high design.

**Rationale (partly specified; fill in / confirm):**
- Build plan rationale: using >2 levels reduces inverted-U ambiguity.
- [ ] Why these labels/spacing are appropriate for v0.

**Implementation evidence:**
- Design principle and level names: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)
- Build plan: [docs/pilot/PILOT_BUILD_PLAN_V0.md](../pilot/PILOT_BUILD_PLAN_V0.md)

**Replication record (must capture):**
- [ ] The exact mapping from “level” to scenario generator parameters.
- [ ] The achieved per-task event rates in the produced scenarios (audit output).

---

## Decision 2: Keep the task set constant; manipulate MWL primarily via event rate + overlap

**Decision statement (exact v0):** The task set is constant across levels (sysmon, track, communications, resman, scheduling display), and MWL is manipulated primarily via event rate + overlap.

**Overlap rules (v0, as specified):**
- Distribution/overlap is determined by OpenMATB difficulty logic / vendor logic generator.
- Minimum spacing between communications prompts: best-effort target 8 seconds; exact integer non-overlap not enforced.

**Implementation evidence:**
- Spec section “Task set (constant across all levels)” and “Overlap rules”: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)

**Replication record (must capture):**
- [ ] Exact overlap/spacing logic in the generator used for scenario creation.
- [ ] Any additional hard constraints added by this repo (beyond vendor logic).

---

## Decision 3: Target per-level total event rates (events/min)

**Decision statement (exact v0):** Total event-rate targets are LOW=3, MODERATE=8, HIGH=18 (events/min), defined as the sum of:
- sysmon gauge failure onsets
- communications radio prompts
- resman pump failures

**Scaling guidance (exact v0):** Increase from LOW to HIGH is ~6× (18/3).

**Implementation evidence:**
- Spec “Per-level event-rate targets”: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md)
- Build plan “v0 workload targets”: [docs/pilot/PILOT_BUILD_PLAN_V0.md](../pilot/PILOT_BUILD_PLAN_V0.md)

**Operationalisation in code (current repo):**
- Scenario generator constants and difficulty mapping: [src/python/generate_pilot_scenarios.py](../../src/python/generate_pilot_scenarios.py)
  - Uses 300s blocks (5:00) and maps LOW/MODERATE/HIGH to numeric difficulty values.

**Replication record (must capture):**
- [ ] Exact scenario generator version used (repo commit) and the output scenario file hashes.
- [ ] A machine-checkable report of the event counts per block (recommended: derived metrics saved alongside manifests).

---

## Decision 4: Operationalise tracking difficulty via specific OpenMATB parameters

**Decision statement:** Tracking difficulty is manipulated via concrete OpenMATB `track` parameters (e.g., update time / control force), rather than changing the task itself.

**Implementation evidence:**
- Parameter mapping exists in [src/python/generate_pilot_scenarios.py](../../src/python/generate_pilot_scenarios.py).

**Rationale (fill in / confirm):**
- [ ] Why these parameters best reflect tracking difficulty (and why they’re stable across runs).

**Replication record (must capture):**
- [ ] Exact numeric mapping for each level.

---

## Decision 5: Operationalise sysmon/comms/resman event rates via scenario generator rules

**Decision statement:** Discrete event loads are defined via generator rules that produce deterministic schedules once authored.

**Implementation evidence:**
- Generator logic: [src/python/generate_pilot_scenarios.py](../../src/python/generate_pilot_scenarios.py).

**Rationale (fill in / confirm):**
- [ ] Why these event types were chosen as the primary event-rate levers.

**Replication record (must capture):**
- [ ] Exact per-task event schedules or a deterministic seed/algorithm reference (depending on how the generator works).
- [ ] Evidence that “events/min” are computed over the active 5:00 only (excluding TLX/breaks).

---

## Open items / TBD

- [ ] Add a table mapping each level → each task parameter/event rate (single place to copy into methods).
- [ ] Add a verification script/output that computes achieved event rates from the scenario files.
