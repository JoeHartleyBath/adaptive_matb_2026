# Pilot Study Spec (v0)

Status: draft (locks the intended pilot protocol; update only via versioned edit)

Last updated: 2026-01-22

Purpose: this document is the single source of truth for the **pilot session structure**, **MWL manipulations**, **marker set**, and **provisional synchronization tolerance wording**.

Constraints:

- No raw/identifiable data in git; follow `docs/DATA_MANAGEMENT.md`.
- Logging/manifest/non-overwrite infrastructure is frozen (use as-is).
- Canonical contracts (must remain consistent with this spec):
  - `docs/contracts/mwl_eeg_input_contract.md`
  - `docs/contracts/eeg_mwl_label_contract_v0.md`

Operational context (implementation already present in repo):

- Run OpenMATB via wrapper: `src/python/run_openmatb.py`.
- Scenario selection via: `src/python/vendor/openmatb/config.ini` (`scenario_path=...`).
- Scenario files live under: `src/python/vendor/openmatb/includes/scenarios/`.
- Primary event log is CSV + adjacent `.manifest.json` written externally (outside repo).
- LSL outlet plugin exists: `src/python/vendor/openmatb/plugins/labstreaminglayer.py`.

---

## 1) Session structure (exact; training vs retained)

Definitions:

- **Training blocks**: participant familiarization; not used for analysis/modeling.
- **Retained blocks**: the pilot blocks intended to be used for dry-run verification artifacts and pilot analyses (subject to ethics + QC gates).

Session phases (exact):

### Phase A — Setup (not timed; operator-controlled)

- Fit EEG cap, impedance checks, verify signal quality.
- Confirm LSL streams visible if LSL is used (EEG stream + OpenMATB marker stream).
- Confirm scenario selection in `config.ini`.

### Phase B — Training (not retained)

Goal: stabilize task proficiency before any retained data.

- Training Block T1: **Low** workload, **5:00**
- Break: **1:00** (quiet rest; no tasks)
- Training Block T2: **Moderate** workload, **5:00**
- Break: **1:00**
- Training Block T3: **High** workload, **5:00**

NASA-TLX during training (exact):

- TLX is **not** administered during training in v0.
  - Rationale (protocol stability): reduce interruptions while the participant is still learning controls.
  - Exception (operator discretion): if training indicates obvious inability to operate tasks, stop and reschedule; do not proceed to retained blocks.

### Phase C — Retained pilot blocks (retained)

Goal: collect comparable blocks at multiple workload levels with deterministic boundaries + subjective ratings.

- Retained Block B1: workload level per counterbalancing (see Section 2), **5:00**
- NASA-TLX: immediately after B1, **self-paced**, max **3:00**
- Break: **1:00**
- Retained Block B2: per counterbalancing, **5:00**
- NASA-TLX: immediately after B2, **self-paced**, max **3:00**
- Break: **1:00**
- Retained Block B3: per counterbalancing, **5:00**
- NASA-TLX: immediately after B3, **self-paced**, max **3:00**

Total planned retained time (excluding setup):

- Task time: 15:00
- TLX time: up to 9:00
- Breaks: 2:00

Pause policy (exact):

- The participant must not use pause unless instructed.
- If pause occurs, the run is still recorded but must be flagged during QC as a protocol deviation; pause handling must be explicitly reflected in markers (see marker list).

---

## 2) Workload levels and order control (exact)

Workload levels:

- `LOW`
- `MODERATE`
- `HIGH`

Training order (fixed):

- T1=LOW → T2=MODERATE → T3=HIGH

Retained order (counterbalanced):

- Default: Latin-square over the 3 levels.
- The operator assigns each participant a sequence ID (e.g., `SEQ1`, `SEQ2`, `SEQ3`) prior to the session.
- The assigned sequence ID must be recorded in session metadata (operator log) and encoded into marker strings (see Section 4).

Allowed sequences (exact):

- `SEQ1`: LOW → MODERATE → HIGH
- `SEQ2`: MODERATE → HIGH → LOW
- `SEQ3`: HIGH → LOW → MODERATE

---

## 3) MWL manipulations per level (exact; table)

Design principle (v0): MWL manipulation is primarily via **event rate + overlap** while keeping the **task set constant** across levels.

Task set (constant across all levels):

- System monitoring (`sysmon`)
- Tracking (`track`)
- Communications (`communications`)
- Resource management (`resman`)
- Scheduling display (`scheduling`) enabled as per scenario defaults

Overlap rules (exact):

- Communications prompts must respect audio non-overlap constraints.
- Minimum spacing between communications prompts: **8 seconds**.
- Within a level, events may coincide across tasks (e.g., sysmon failure at same time as resman pump failure) except where the communications spacing rule would be violated.

Per-level event-rate targets (exact):

| Component | Knob / event type | LOW | MODERATE | HIGH | Notes / invariants |
|---|---:|---:|---:|---:|---|
| Sysmon | Gauge failure onsets (count per minute) | 2 / min | 4 / min | 6 / min | Implement via `scales-*-failure` / `lights-*-failure` schedule; keep failure types comparable across levels |
| Communications | Radio prompts (count per minute) | 2 / min | 3 / min | 4 / min | Enforce ≥8s spacing; mix of `own` vs `other` prompts fixed ratio (see below) |
| Communications | Target:distractor prompt ratio | 80:20 | 80:20 | 80:20 | Keep task set constant; change rate, not semantics |
| Resman | Pump failures (count per minute) | 1 / min | 2 / min | 3 / min | Use `pump-*-state;failure` events; keep tank targets constant |
| Resman | Automation availability | On | On | On | Keep automation policy constant; do not introduce new controls across levels |
| Track | Target proportion | fixed | fixed | fixed | Keep tracking geometry constant; objective is overlap/rate elsewhere |
| Scheduling | Display | On | On | On | Keep constant |

Notes:

- These are **targets** for scenario design; the scenario files must be deterministic once authored.
- “Count per minute” is defined over the active 5:00 block duration, excluding TLX and breaks.

---

## 4) Marker set (exact) and where markers appear

Goal: every retained block and questionnaire segment must be deterministically bracketed in logs.

Marker transport requirements (v0):

- **CSV**: all markers must appear in the OpenMATB CSV event log.
- **LSL**: all markers must appear as samples in the OpenMATB LSL marker stream **if LSL is the chosen synchronization timebase**.
- For LSL, use the existing `labstreaminglayer` plugin.

Marker naming format (exact):

- Prefix all markers with `STUDY/V0/`.
- Include participant/session IDs and retained sequence ID in each marker payload to make stream mixing detectable.

Marker payload template (exact):

- `STUDY/V0/<MARKER_NAME>;pid=<P>;sid=<S>;seq=<SEQ>`

Markers (exact list):

| Marker name | When emitted | Appears in CSV | Appears in LSL | Purpose |
|---|---|---:|---:|---|
| `SESSION_START` | immediately after OpenMATB begins the session | Yes | Yes | anchors session start |
| `TRAINING/T1/START` | start of training block T1 | Yes | Yes | brackets training |
| `TRAINING/T1/END` | end of training block T1 | Yes | Yes | brackets training |
| `TRAINING/T2/START` | start of training block T2 | Yes | Yes | brackets training |
| `TRAINING/T2/END` | end of training block T2 | Yes | Yes | brackets training |
| `TRAINING/T3/START` | start of training block T3 | Yes | Yes | brackets training |
| `TRAINING/T3/END` | end of training block T3 | Yes | Yes | brackets training |
| `RETAINED/B1/<LEVEL>/START` | start of retained block B1 | Yes | Yes | primary label interval start |
| `RETAINED/B1/<LEVEL>/END` | end of retained block B1 | Yes | Yes | primary label interval end |
| `TLX/B1/START` | immediately before NASA-TLX for B1 | Yes | Yes | brackets TLX |
| `TLX/B1/END` | immediately after TLX for B1 | Yes | Yes | brackets TLX |
| `RETAINED/B2/<LEVEL>/START` | start of retained block B2 | Yes | Yes | primary label interval start |
| `RETAINED/B2/<LEVEL>/END` | end of retained block B2 | Yes | Yes | primary label interval end |
| `TLX/B2/START` | immediately before NASA-TLX for B2 | Yes | Yes | brackets TLX |
| `TLX/B2/END` | immediately after TLX for B2 | Yes | Yes | brackets TLX |
| `RETAINED/B3/<LEVEL>/START` | start of retained block B3 | Yes | Yes | primary label interval start |
| `RETAINED/B3/<LEVEL>/END` | end of retained block B3 | Yes | Yes | primary label interval end |
| `TLX/B3/START` | immediately before NASA-TLX for B3 | Yes | Yes | brackets TLX |
| `TLX/B3/END` | immediately after TLX for B3 | Yes | Yes | brackets TLX |
| `SESSION_END` | immediately before clean OpenMATB exit | Yes | Yes | anchors session end |
| `ABORT` | if run is aborted early | Yes | Yes | makes early termination machine-detectable |
| `PAUSE` | if pause is engaged | Yes | Yes | flags protocol deviation |
| `RESUME` | when resuming after pause | Yes | Yes | flags protocol deviation |

Notes:

- `<LEVEL>` must be one of `LOW|MODERATE|HIGH`.
- Training markers exist for completeness but training blocks are not intended for analysis.

---

## 5) Provisional alignment tolerance wording (to be finalized after dry run)

This wording is normative for v0 until replaced by a versioned update.

Provisional statement (exact):

- **Synchronization tolerance (provisional):** The maximum allowed absolute alignment error between a task event marker and the corresponding EEG time is
  $|\Delta t| \le 20$ ms (target).
- This tolerance is **provisional** and will be **finalized after dry-run measurement** on the intended acquisition machine and recording stack.

How the dry run finalizes it (summary requirement):

- Measure empirical alignment error between OpenMATB markers and EEG on the chosen reference timebase (recommended: LSL timestamps).
- Record the observed distribution (at minimum: median and 95th percentile of $|\Delta t|$).
- Update the finalized tolerance in a versioned doc update if the empirical distribution requires a looser bound.

Acceptance criterion for v0 dry run (until finalized):

- 95th percentile of $|\Delta t|$ must be ≤ 20 ms, or the tolerance must be revised (with justification) before pilot recruitment.
