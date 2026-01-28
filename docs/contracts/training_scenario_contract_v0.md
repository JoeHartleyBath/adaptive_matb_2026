# Training Scenario Contract (v0)

Status: draft (implementation in-progress)

Last updated: 2026-01-23

Purpose: define the **deterministic, contract-compliant structure** of the OpenMATB v0 pilot session scenarios (training + calibration) used in this repository.

Scope:

- Covers **scenario structure + marker semantics** for both training and calibration blocks described in `docs/pilot/PILOT_STUDY_SPEC_V0.md`.
- Specifies the minimum inputs required to run the scenario in a way that is **one-shot reproducible** and **log-bracketed**.
- Defines the scenario-side requirements only (not EEG ingestion). EEG alignment and labeling are governed by:
  - `docs/contracts/mwl_eeg_input_contract.md`
  - `docs/contracts/eeg_mwl_label_contract_v0.md`

Non-goals (v0):

- Does not define adaptive logic.
- Does not redefine OpenMATB’s native task mechanics.

---



## 1) Definitions

- **Training blocks**: T1–T3. Familiarization only; not calibration for analysis/modeling.
- **calibration blocks**: B1–B3. Intended for dry-run verification artifacts and pilot analyses (subject to ethics + QC gates).
- **Workload levels**: `LOW`, `MODERATE`, `HIGH`.
- **Sequence ID**: calibration order assignment `SEQ1|SEQ2|SEQ3`.

---

## 2) Required run inputs

The run must provide:

- `participant_id` (required; safe for folder names)
- `session_id` (required; safe for folder names)
- `seq_id` ∈ {`SEQ1`, `SEQ2`, `SEQ3`} (required; calibration order control)

These inputs must be made available to the scenario/marker layer (either by direct string substitution during generation or by runtime templating).

---

## 3) Scenario outputs (normative)

A contract-compliant run must yield:

- Three deterministic combined session scenario `.txt` files (one per calibration `seq_id`) whose timelines are fully specified:
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq1.txt`
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq2.txt`
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq3.txt`
- An OpenMATB CSV event log (written outside git per `docs/DATA_MANAGEMENT.md`).
- A `.manifest.json` adjacent to the CSV that includes repository/submodule provenance.

---

## 4) Session structure (exact)

The scenario must implement the pilot session structure from `docs/pilot/PILOT_STUDY_SPEC_V0.md`:


### 4.0 Instructions / familiarisation (no separate scenario phase in v0)

- v0 has no separate in-scenario “Phase 0”. Training (T1–T3) is the first in-scenario segment.
- Any pre-run briefing is out-of-band (operator procedure) and not represented as a separate scenario phase.
- A participant ID entry popup must appear before training begins, and the scenario must not proceed until a valid ID is submitted.
- Pilot scenarios MUST NOT reuse the upstream demo scenario (`default.txt`) and MUST NOT reference any demo/French instruction assets.
- Any on-screen instruction content used by the pilot scenario MUST be study-owned English instruction assets (see `docs/pilot/PILOT_BUILD_PLAN_V0.md`).



### 4.1 Training (not calibration)

- T1: `LOW`, 5:00
- break: 1:00 (quiet rest; no tasks)
- T2: `MODERATE`, 5:00
- break: 1:00
- T3: `HIGH`, 5:00

### 4.2 calibration blocks + TLX (calibration)

- B1: per `seq_id`, 5:00
- TLX after B1: self-paced, untimed; all sliders must be interacted with before continuing
- break: 1:00
- B2: per `seq_id`, 5:00
- TLX after B2: self-paced, untimed; all sliders must be interacted with before continuing
- break: 1:00
- B3: per `seq_id`, 5:00
- TLX after B3: self-paced, untimed; all sliders must be interacted with before continuing

### 4.3 calibration order mapping (exact)

- `SEQ1`: LOW → MODERATE → HIGH
- `SEQ2`: MODERATE → HIGH → LOW
- `SEQ3`: HIGH → LOW → MODERATE

---

## 5) MWL manipulation requirements (exact)

Design principle (v0): manipulate MWL primarily via **event rate + overlap** while keeping the **task set constant** across levels.

### 5.1 Task set (constant across all levels)

- `sysmon`, `track`, `communications`, `resman`, `scheduling` (enabled as per scenario defaults)

### 5.2 Overlap rules (exact)

- Communications prompts must not overlap.
- Minimum spacing between communications prompts: 8 seconds.

### 5.3 Per-level targets (exact)

Per-minute event rates over each active 5:00 block (normative for v0):

Scaling rule (normative for v0): HIGH total event rate is exactly **6×** LOW total event rate (18 vs 3 events/min).

These rates are identical to the pilot study spec.

### 5.4 Deterministic per-block scheduling template (authoritative; v0)

This section is the authoritative definition for v0 scenario authoring. A contract-compliant scenario must implement this template exactly.

Definitions (exact):

- Block duration is 300 seconds.
- All event times are integer **offsets in seconds from block start**.
- Allowed offsets are **15..289 inclusive**.
  - No events are scheduled at offsets **0..14**.
  - No events are scheduled at offsets **290..299**.
- No two events (across Sysmon, Communications, Resman) are scheduled at the same offset.
- Communications prompts satisfy minimum spacing of **8 seconds** between consecutive communications offsets.

Counts per 5:00 block (exact):

- **LOW**: Sysmon=5, Communications=5, Resman=5 (total=15)
- **MODERATE**: Sysmon=15, Communications=15, Resman=10 (total=40)
- **HIGH**: Sysmon=30, Communications=30, Resman=30 (total=90)

Base offset generation rule (exact):

For a task with N events, generate its base offset sequence $o_i$ for $i=0..N-1$ as:

$$o_i = 15 + \left\lfloor \frac{(i+1)\cdot 275}{N+1} \right\rfloor$$

This produces a strictly non-decreasing integer sequence within the allowed window.

Deterministic collision + constraint resolution (exact):

1) Create a single combined event list by taking all generated base offsets for Sysmon, Communications, Resman and ordering by:
   - offset ascending, then
   - task priority order: Sysmon, then Communications, then Resman, then
   - within-task index ascending.
2) Traverse the ordered list from first to last. Maintain a set of already-assigned offsets.
3) When processing an event, assign it an offset as follows:
   - Start with the event’s current offset.
   - While any constraint below is violated, increment the event’s offset by **+1 second** and re-check all constraints.
   - If incrementing would move the offset beyond 289, scenario generation must fail (contract violation).
4) Constraints checked during assignment are:
   - offset is within 15..289 inclusive
   - offset is not already assigned
   - if the event is a Communications prompt, its offset is at least 8 seconds after the previous assigned Communications prompt offset

Deterministic communications target/distractor assignment (exact):

- For each block, index Communications prompts in chronological order starting at 1.
- Prompts with indices divisible by 5 are distractors; all others are targets.

Note (normative): the schedule produced by the rules above is the only valid v0 schedule definition; scenario authoring must not introduce alternative timing, jitter, randomization, or manual edits to event times.

---

## 6) Marker contract (normative)

### 6.1 Transport requirements

- **CSV**: all markers must appear in the OpenMATB CSV event log.
- **LSL**: all markers must appear in the OpenMATB LSL marker stream if LSL is the chosen synchronization timebase.

### 6.2 Marker payload format (exact)

All markers must use:

- Prefix: `STUDY/V0/`
- Template: `STUDY/V0/<MARKER_NAME>|pid=<P>|sid=<S>|seq=<SEQ>`

### 6.3 Marker list (exact)

The scenario must emit the exact markers defined in `docs/pilot/PILOT_STUDY_SPEC_V0.md` Section 4.

### 6.4 Pause/abort semantics

PAUSE/RESUME: not supported as deterministic study markers in v0

If pausing occurs, it must be recorded as a runtime log event (CSV/manual) and/or in the manifest notes.

Operator procedure: do not pause during B1–B3; if pause is needed, abort and restart.

ABORT: defined as early termination

ABORT is inferred by: missing STUDY/V0/SESSION_END marker (or whatever your end marker is) and a non-success run status in the manifest.

Scenario-level markers cover only events that the scenario author can deterministically schedule. UI-driven pause/close events are not scenario-addressable in v0.

---

## 7) Determinism requirements

A compliant scenario must be deterministic under fixed inputs:

- No RNG is permitted in scenario generation unless the seed strategy is explicit and auditable.
- If any generation step is used, it must be stable (same inputs → identical scenario text).

---

## 8) OpenMATB compatibility requirements

Scenario files must comply with OpenMATB’s scenario parser constraints:

- Line format: `H:MM:SS;<plugin>;<command>` or `H:MM:SS;<plugin>;<parameter>;<value>`
- Each plugin used must have a `start` command and (if non-blocking) a `stop` command.
- Maximum command length is 2 fields (parameter;value).

---

## 9) Provenance and data boundaries

- OpenMATB outputs (sessions CSV + error log) must remain outside git and untracked.
- The run manifest must include:
  - `OPENMATB_REPO_COMMIT`
  - `OPENMATB_SUBMODULE_COMMIT`

---

## 10) Compliance checklist (v0)

A run is contract-compliant iff:

- All required inputs (`participant_id`, `session_id`, `seq_id`) are provided.
- All blocks and TLX segments are deterministically bracketed by markers.
- Marker payloads include pid/sid/seq per the template.
- Communications prompts never violate the ≥8s spacing rule.
- Scenario is deterministic and versioned.
- Outputs land outside git and include a manifest with repo/submodule commits.
