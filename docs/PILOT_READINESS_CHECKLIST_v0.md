# Pilot Readiness Checklist (v0)

Purpose: define **objective, checkable** conditions under which the MATB(-II) paradigm + EEG pipeline are ready to begin pilot data collection.

Pilot definition: A small number of non-final participants used to validate task timing, signal quality, and pipeline integrity; pilot data may or may not be used in final analyses depending on ethics approval.

Scope: scenario setup, EEG acquisition, time alignment, contracts, and compliance. **No modelling** and no assumptions that data already exists.

Last updated: 2026-01-22

---

## Status rubric (must be unambiguous)

### NOT READY

- [ ] Any item marked **HARD GATE** is unchecked.
- [ ] Any required contract field cannot be populated from planned recordings/logs.
- [ ] Any synchronization/timing tolerance is unspecified.

### READY FOR DRY RUN (no participant recruitment)

- [ ] All items in **Repo & Runability (Dry-run gate)** are checked.
- [ ] All items in **EEG Acquisition & Synchronisation (Dry-run gate)** are checked.
- [ ] All items in **Contract Compliance (Dry-run gate)** are checked.
- [ ] Dry run is approved/authorized for research-team self-test (see **Ethics & Governance**).

### READY FOR PILOT DATA COLLECTION

- [ ] All items in this document are checked, including **Ethics & Governance (HARD GATE)**.

---

## Ethics & Governance (HARD GATE)

Ethics approval status (required before any participant data collection):

- [ ] Ethics approval is **granted** (not “submitted”/“pending”) for this protocol **before** recruiting/recording any non-team participants.
- [ ] Ethics approval identifier (IRB/REC ref) and approved version/date are recorded in a non-sensitive repo doc (e.g., `docs/`), without embedding participant data.
- [ ] Any protocol deviations needed for the pilot (hardware, tasks, duration, questionnaires) are either (a) already covered by the approval, or (b) formally approved as an amendment.

Consent materials existence/location:

- [ ] Participant information sheet exists and is versioned (location recorded; do not store signed forms in git).
- [ ] Consent form exists and is versioned (location recorded; do not store signed forms in git).
- [ ] A withdrawal procedure is documented (what gets deleted, what can’t be deleted, and the time window).

Data handling and storage confirmation (aligned with `docs/DATA_MANAGEMENT.md`):

- [ ] External data root location is defined per machine via `config/paths.yaml` (local-only) and follows the structure described in `docs/DATA_MANAGEMENT.md`.
- [ ] Confirmed: raw/identifiable participant data is stored **outside** git; only small, non-sensitive derived artifacts may enter `results/`.
- [ ] A retention policy exists (what is kept, for how long, where), consistent with institutional policy and the approved ethics protocol.
- [ ] A de-identification plan exists for any files that could contain quasi-identifiers (e.g., timestamps + session IDs), consistent with `docs/DATA_MANAGEMENT.md`.

---

## Repo & Runability (Dry-run gate)

OpenMATB runability and external logging:

- [ ] OpenMATB runs via the supported wrapper entrypoint `src/python/run_openmatb.py` on the intended acquisition machine.
- [ ] A single run produces both a session CSV and a `.manifest.json` **outside** the repo (per `docs/openmatb/SETUP_AND_RUN.md`).
- [ ] The manifest contains required keys (per `docs/run_logging_verification.md`) and `ended_at_local` is non-null after a clean exit.
- [ ] Output directories written by OpenMATB (e.g., `sessions/`) remain ignored by git (per `docs/DATA_MANAGEMENT.md`).

Scenario file provenance:

- [ ] The exact scenario(s) intended for dry run and pilot are identified by **path and filename** under `src/python/vendor/openmatb/includes/scenarios/` (or an approved study-specific location).
- [ ] The exact OpenMATB version is pinned via the submodule commit, and the parent repo commit is recorded in the run manifest.

---

## Task / Scenario Definition

Scenarios defined with explicit MWL manipulations:

- [ ] A finite set of pilot scenarios is defined (no “we’ll decide at runtime”).
- [ ] Each scenario has an explicit, written mapping: **condition → intended MWL manipulation** (e.g., low/medium/high demand), including which task parameters/events implement the manipulation.
- [ ] Each condition’s duration is defined and bounded (start/end), including breaks and questionnaires if present.

Deterministic start/end events for each condition:

- [ ] Each condition has deterministic **start** and **end** markers that appear in the OpenMATB event log (CSV and/or LSL marker stream).
- [ ] Condition boundaries are defined by logged events, not by “wall clock memory” or manual notes.
- [ ] Any pause/escape handling rule is defined: whether paused time counts toward condition time, and how it is reflected in logs.

Operator procedure consistency:

- [ ] A run sheet exists that specifies exactly what the operator does (launch, checks, start, stop, abort) so that condition timing is repeatable.

---

## EEG Acquisition & Synchronisation

EEG device and driver defined:

- [ ] The EEG device make/model and firmware version are defined.
- [ ] The acquisition software and driver stack are defined (including version), and are compatible with the OS used for acquisition.
- [ ] The EEG channel list/montage used for this study is defined (names + ordering) and is stable across sessions.

Clock strategy chosen (e.g., LSL):

- [ ] A single synchronization strategy is chosen and documented (e.g., LSL for EEG + task events).
- [ ] The timebase used for alignment is explicitly stated (e.g., LSL timestamps), including which stream is the reference.
- [ ] OpenMATB event markers are available on the same alignment timebase as EEG (directly or via a deterministic mapping).

Event-to-EEG alignment tolerance explicitly stated:

- [ ] The maximum allowed absolute alignment error between a task event marker and the corresponding EEG time ($\Delta t$ tolerance) is stated (e.g., “$|\Delta t| \le X$ ms”).
- [ ] The method for confirming the tolerance is defined as an observable check (e.g., comparing task markers vs EEG markers in a test run), without requiring any modelling.

Failure modes and fallbacks:

- [ ] If LSL/event streaming fails mid-run, the failure is detectable from logs and the run is classified as unusable (or explicitly downgraded) by a written rule.
- [ ] If the EEG recording has missing timestamps or non-monotonic timing, it is rejected by a written rule.
- [ ] Runs that violate synchronization or contract rules are never silently used for modelling or adaptation (must be explicitly flagged or excluded).

---

## Contract Compliance

Ability to populate all required fields of Input Contract v0 (`docs/mwl_eeg_input_contract.md`) (Dry-run gate):

- [ ] All `TBD_*` placeholders in `docs/mwl_eeg_input_contract.md` are resolved to concrete values (window length, step, effective sampling rate, bandpass/notch settings, channel order, artifact/bad-channel policy).
- [ ] The study’s EEG montage can be mapped deterministically to `TBD_CHANNELS_ORDERED_V0` (no ambiguous channel names).
- [ ] The acquisition recording format contains enough metadata to verify sampling rate, units, and reference scheme.
- [ ] A deterministic windowing definition exists that matches the contract (window center timestamps, grid spacing).

Ability to populate all required fields of Label Contract v0 (`docs/eeg_mwl_label_contract_v0.md`) (Dry-run gate):

- [ ] The MWL label source(s) for the pilot are declared and admissible under the label contract.
- [ ] The label temporal support is declared (block/trial/continuous) and yields deterministic window-label assignment.
- [ ] All `TBD_*` placeholders in the label contract are resolved (target space/range/levels, if used).

Cross-file consistency:

- [ ] The contract-required identifiers needed to join EEG ↔ events ↔ labels are defined (session ID, timestamps, condition IDs).
- [ ] Condition IDs used in scenarios/logs match the label contract naming without ad-hoc translation.

---

## Data Quality Gates

Minimum usable recording criteria:

- [ ] A run-level “usable vs unusable” rule exists for EEG (e.g., minimum duration recorded, minimum fraction of valid samples/windows, maximum dropout, impedance criteria if applicable).
- [ ] A run-level “usable vs unusable” rule exists for events (e.g., all condition boundary markers present; no missing segments; monotonic timestamps).
- [ ] A run is automatically classified as unusable if synchronization criteria are violated (as defined above).

Explicit failure handling rules:

- [ ] Abort criteria are defined (what failures stop the run immediately).
- [ ] Restart criteria are defined (when a run may be restarted vs must be rescheduled).
- [ ] Partial runs are handled by rule (kept as unusable test artifacts vs deleted), consistent with the governance plan.

---

## Dry-Run Verification (no modelling)

Single test recording (researcher as participant):

- [ ] A single end-to-end test recording is completed by a research team member on the intended hardware/software stack.
- [ ] Authorization is documented for this dry run (either ethics approval covers it, or an explicit institutional determination of exemption/non-retention is recorded).

Confirmation that windowed data can be generated matching the contract:

- [ ] From the dry-run EEG recording, it is possible to generate fixed windows that match the Input Contract v0 timing definition.
- [ ] From the OpenMATB logs, it is possible to assign condition/label intervals to EEG windows deterministically (per Label Contract v0).
- [ ] All required fields for the v0 input unit and label unit can be populated for at least one complete condition block.

Basic QC outputs exist (signal sanity + event timeline):

- [ ] A basic signal sanity output exists for the dry run (e.g., per-channel amplitude summary, missing-data summary), stored outside git or as a non-sensitive small artifact.
- [ ] A basic event timeline output exists for the dry run (e.g., condition boundary markers and key events), stored outside git or as a non-sensitive small artifact.
- [ ] QC outputs make it obvious when timing is broken (e.g., missing markers, non-monotonic time).

---

## Current repo blockers / missing elements (as of 2026-01-22)

These items are **expected to be unchecked** until the corresponding repo artifact exists.

Ethics & consent blockers:

- [ ] Ethics approval identifier/status document exists under `docs/` (or an approved location).
- [ ] Consent materials are present or their controlled-storage location is recorded in `docs/`.

EEG acquisition blockers:

- [ ] EEG device + acquisition software/driver stack are defined in a repo doc.
- [ ] Synchronization timebase + alignment tolerance ($\Delta t$) are stated in a repo doc.

Contract blockers:

- [ ] `docs/mwl_eeg_input_contract.md` has no unresolved `TBD_*` items.
- [ ] `docs/eeg_mwl_label_contract_v0.md` has no unresolved `TBD_*` items.

Dry-run pipeline blockers:

- [ ] A documented, reproducible way exists to produce contract-shaped windows and an event/label timeline from a single recording (no modelling required).
