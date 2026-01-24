# Pilot Build Plan (v0)

Status: draft

Last updated: 2026-01-22

Purpose: build a pilot-ready MATB(-II)/OpenMATB study pipeline that can run end-to-end on the acquisition machine and produce:

1) deterministic condition markers + logs
2) EEG acquisition + synchronisation (LSL or equivalent)
3) a dry-run verification artifact proving we can generate contract-shaped EEG windows and deterministically align labels to windows
4) a clear operator run procedure
5) readiness checklist gates satisfied (ethics is a hard gate)

Non-goals (v0):

- No modelling, pretraining, fine-tuning, online learning, or optimization.
- No redesign of logging/manifest/non-overwrite infrastructure.
- No changes to the semantics of the canonical contracts.

Canonical constraints (non-negotiable):

- Input contract: `docs/contracts/mwl_eeg_input_contract.md`
- Label contract: `docs/contracts/eeg_mwl_label_contract_v0.md`
- Logging/manifest/non-overwrite: treat as frozen; only use as documented.
- Data boundaries: no raw/identifiable participant data or large artifacts in git; follow `docs/DATA_MANAGEMENT.md`.

---

## Repository audit findings (concrete paths inspected)

OpenMATB wrapper / entrypoint:

- Supported wrapper entrypoint: `src/python/run_openmatb.py`
  - Requires `--participant` and `--session` (or env vars) and validates identifiers.
  - Sets external output routing via `OPENMATB_OUTPUT_ROOT` and `OPENMATB_OUTPUT_SUBDIR`.
  - Injects commit identifiers `OPENMATB_REPO_COMMIT` and `OPENMATB_SUBMODULE_COMMIT` (required by the manifest writer).

Scenario selection and scenario file location:

- Scenario path is selected via `scenario_path` in `src/python/vendor/openmatb/config.ini`.
- Scenario files are loaded from `src/python/vendor/openmatb/includes/scenarios/` (relative paths allowed).
  - Upstream demo scenarios present in-tree (not to be used/referenced by pilot scenarios):
    - `src/python/vendor/openmatb/includes/scenarios/basic.txt`
    - `src/python/vendor/openmatb/includes/scenarios/default.txt`
- Scenario parser logs the active scenario path into the CSV as a manual record:
  - `Scenario.__init__` in `src/python/vendor/openmatb/core/scenario.py` calls `logger.log_manual_entry(..., key='scenario_path')`.

Primary event log (CSV) and run manifest:

- CSV schema (v0): `logtime, scenario_time, type, module, address, value`.
  - Writer: `src/python/vendor/openmatb/core/logger.py`.
- Run manifest is written adjacent to the CSV as `<stem>.manifest.json`.
  - Writer: `Logger._write_manifest()` in `src/python/vendor/openmatb/core/logger.py`.
  - `ended_at_local` is set only on clean shutdown (`Logger.finalize()`), otherwise remains null.
- Output base directory is resolved from env vars:
  - `OUTPUT_BASE_DIR` in `src/python/vendor/openmatb/core/constants.py` uses `OPENMATB_OUTPUT_ROOT` + `OPENMATB_OUTPUT_SUBDIR`.

LSL integration points:

- LSL *outlet* plugin exists: `src/python/vendor/openmatb/plugins/labstreaminglayer.py`.
  - Stream name: `OpenMATB` (type `Markers`, 1 string channel).
  - Supports:
    - `streamsession=True` to stream full CSV rows (semicolon-separated) over LSL.
    - `marker=<string>` to emit explicit marker samples.
- Unified event stream hook (for audit context only): `Logger.write_row_queue()` streams rows when `logger.lsl` is set.

Verification docs already present:

- Manifest + CSV expectations: `docs/run_logging_verification.md`.
- Two-run no-overwrite verification example: `docs/two_run_no_overwrite_verification.md`.

---

## Pilot design principles to implement (constraints turned into checkable requirements)

Workload manipulation lever:

- Primary lever is event rate + overlap (concurrency), while keeping the task set constant across levels.
- Use >2 workload levels to reduce inverted-U ambiguity: Low / Moderate / High.

v0 workload targets (authoritative; must match the study spec + scenario contract):

- Total event-rate targets (events/min): `LOW`=3, `MODERATE`=8, `HIGH`=18.
- Scaling rule (v0): `HIGH` total event rate is ~5–6× `LOW` (operationalized as 18/3 = 6×).

Order control:

- Workload level order must be counterbalanced across participants via retained `seq_id` assignment (`SEQ1|SEQ2|SEQ3`).
- `seq_id` applies to retained block order only; training is invariant across participants and does not depend on `seq_id`.
- The repo uses three combined session scenario files (one per `seq_id`) that each embed the same training segment, followed by retained blocks ordered per `seq_id`.
- Order assignment must be reproducible and checkable from the run manifest or session metadata (at minimum: recorded `seq_id`).

Training:

- Provide structured training (recommended: 3 × 5 minutes) and document it as part of the operator run sheet.
- The scenario must present a participant ID entry popup before training begins and must not proceed until a valid ID is submitted.

Subjective validation:

- NASA-TLX after each retained block (B1–B3); TLX is not administered during training in v0.
- NASA-TLX is untimed and requires interaction with all sliders before continuing.
- Use OpenMATB `genericscales` + `nasatlx_en.txt` as the in-task questionnaire mechanism.

Objective validation:

- Track objective tracking error using OpenMATB tracking performance metric `center_deviation` (`plugins/track.py`).
- Also record task-level performance metrics already emitted by OpenMATB for sysmon/resman/comms, and global performance if the `performance` plugin is enabled.

Reproducibility requirements:

- Explicitly record all task parameters needed for reproducibility:
  - event rates per task (sysmon failures, comms prompts, resman pump failures, etc.)
  - overlap rules (which tasks can generate events simultaneously; any enforced spacing)
  - level durations (bounded start/end)
  - total session duration

---

## Interfaces and invariants (what must not change)

### Canonical contracts (interfaces)

- EEG windows must match `docs/contracts/mwl_eeg_input_contract.md` (v0): fixed-length, fixed-step windows on a declared effective sampling rate.
- Label assignment must match `docs/contracts/eeg_mwl_label_contract_v0.md` (v0): window-center assignment; deterministic boundary rule.

### Logging and storage (frozen infrastructure)

- OpenMATB outputs must remain outside git:
  - default external root on Windows is `C:\data\adaptive_matb` unless overridden.
  - wrapper sets `OPENMATB_OUTPUT_SUBDIR = openmatb/<participant>/<session>`.
- A valid run must produce (outside repo):
  - session CSV under `.../sessions/YYYY-MM-DD/`.
  - adjacent `.manifest.json`.
  - `last_scenario_errors.log`.

### Timebases (must be explicitly chosen and stated)

OpenMATB provides two timebases by default:

- `logtime`: monotonic host time (`perf_counter()`), in seconds.
- `scenario_time`: scenario-relative time advanced by pyglet `dt`.

LSL (if used) provides a third timebase:

- LSL timestamps in the recording system (recommended to be the alignment reference timebase when EEG is also recorded via LSL).

In v0 we require:

- one declared reference timebase for EEG↔events alignment
- one declared maximum allowed absolute alignment error $|\Delta t| \le X$ ms

---

## Condition markers and labels (deterministic by construction)

### Condition markers in OpenMATB (requirements)

Each workload block must have deterministic start and end markers that appear in:

- the OpenMATB event log (CSV), and
- the LSL marker stream (if LSL is the chosen timebase).

Recommended mechanism (uses existing capabilities only):

- Use the `labstreaminglayer` plugin to emit markers at:
  - session start
  - block start and block end
  - NASA-TLX start and end
  - abort / early exit if applicable

Marker naming convention (v0):

- `STUDY/SESSION_START`
- `STUDY/BLOCK/<level>/START`
- `STUDY/BLOCK/<level>/END`
- `STUDY/NASA_TLX/START`
- `STUDY/NASA_TLX/END`
- `STUDY/SESSION_END`

### Labels under the label contract

For v0 pilot, the simplest admissible label is task-demand (block-level):

- Label temporal support: block-level intervals.
- Label values: ordered levels `{LOW, MODERATE, HIGH}` (or numeric encoding) mapped to the contract’s declared target space.

Deterministic label assignment rule (must match contract):

- each EEG window gets the label of the interval that contains the window-center timestamp
- if the centre lies exactly on a boundary, assign to the earlier interval

---

## Step-by-step Pilot Build Plan (smallest reversible steps)

Each step includes: objective, files it touches, success criteria, and failure modes.

### 1) Ethics hard gate documentation

Objective:

- Record ethics approval status and constraints needed for a pilot and for an internal dry run.

Files touched:

- `docs/ethics/` (add a non-sensitive status memo)
- `docs/pilot/PILOT_READINESS_CHECKLIST_v0.md` (checklist updates only)

Success criteria:

- Ethics approval is explicitly granted for participant recruitment/recording (or a documented determination forbids recruitment and limits work to non-retained researcher dry runs).
- Approval identifier and approved protocol version/date recorded.

Failure modes:

- Pilot attempted without approval → hard stop.
- Ambiguity about whether a dry run counts as human-subject data collection → hard stop until documented.

### 2) Acquisition machine storage configuration (data boundaries)

Objective:

- Ensure the acquisition machine uses an external data root and never writes large/sensitive outputs into git.

Files touched:

- `config/paths.yaml` (local-only; must remain untracked)
- `docs/DATA_MANAGEMENT.md` (reference only; update only if policy changes)

Success criteria:

- `config/paths.yaml` exists locally and points to an external drive root.
- OpenMATB outputs go to the external root (`OPENMATB_OUTPUT_ROOT`) and are organized by participant/session.

Failure modes:

- outputs written inside the repo → stop and fix configuration
- uncertainty about where raw EEG files are stored → stop

### 3) Choose synchronization strategy and reference timebase (LSL recommended)

Objective:

- Make alignment assumptions explicit and testable.

Files touched:

- `docs/pilot/PILOT_BUILD_PLAN_V0.md` (this file)
- (optionally) `docs/pilot/DECISIONS_PILOT_V0.md` (if you prefer a separate decision ledger)

Success criteria:

- A single reference timebase is declared (e.g., LSL timestamps).
- $|\Delta t| \le X$ ms tolerance is declared.
- A concrete verification method is declared (see Dry Run Protocol).

Failure modes:

- mixed clocks without deterministic mapping
- tolerance not stated or cannot be verified from recorded artifacts

### 4) Define the pilot scenario set and parameter ledger

Objective:

- Specify a finite, versioned set of scenarios and the exact MWL manipulation mapping.

Files touched:

- Scenario files under `src/python/vendor/openmatb/includes/scenarios/` (new pilot scenario files)
- `docs/pilot/PILOT_STUDY_SPEC_V0.md` (new; non-sensitive parameters/targets)

Success criteria:

- Exactly which scenarios are used is specified by path and filename.
- Scenario set is exactly three combined session scenario files (v0; one per retained `seq_id`), loaded via `scenario_path`:
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq1.txt` (for `SEQ1`)
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq2.txt` (for `SEQ2`)
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq3.txt` (for `SEQ3`)
- Each scenario file begins with the same training segment (T1–T3; `LOW`→`MODERATE`→`HIGH`), then runs retained blocks (B1–B3) ordered per `seq_id`.
- For each workload level within the session, all manipulated parameters and event-rate targets are recorded.
- Level durations are fixed and bounded.

Failure modes:

- “decide at runtime” parameters
- scenario filenames not pinned
- inability to map scenario differences to MWL levels deterministically

### 5) Add deterministic block markers (CSV + LSL)

Objective:

- Ensure every block boundary has a machine-observable marker.

Files touched:

- Pilot scenario files under `src/python/vendor/openmatb/includes/scenarios/`

Success criteria:

- For each block, markers exist for START and END.
- Markers appear in the LSL stream (if enabled) and can be recovered from the recorded LSL file.
- Markers are also recoverable from the CSV (directly as scenario events and/or as session-row stream over LSL).

Failure modes:

- missing/duplicated markers
- ambiguous boundaries (overlapping intervals)
- pause/abort behavior not reflected in markers

### 6) Embed NASA-TLX after each block

Objective:

- Capture subjective MWL after each retained block (B1–B3) using a consistent, in-task instrument.

Files touched:

- Pilot scenario files under `src/python/vendor/openmatb/includes/scenarios/`
- Questionnaire assets reference: `src/python/vendor/openmatb/includes/questionnaires/nasatlx_en.txt` (reference only)

Success criteria:

- NASA-TLX starts after each block and is bounded by explicit markers.
- NASA-TLX is untimed and the continue action is blocked until all sliders are interacted with.
- The operator run sheet includes when the participant is instructed to complete it.

Failure modes:

- questionnaire timing not bracketed by markers
- questionnaire skipped without being detectable from logs
- questionnaire allows continuation without slider interaction

### 7) Declare contract resolutions (replace all `TBD_*` with concrete values)

Objective:

- Fully resolve the v0 contracts to concrete parameters for this pilot.

Files touched:

- `docs/contracts/mwl_eeg_input_contract.md`
- `docs/contracts/eeg_mwl_label_contract_v0.md`
- `docs/pilot/PILOT_CONTRACT_RESOLUTIONS_V0.md` (recommended as the single ledger of resolved values)

Success criteria:

- Input contract: all `TBD_*` fields are resolved (window length, step, effective FS, notch mains, channel order, artifact/bad-channel policy, baseline reference segments, minimum valid coverage).
- Label contract: `TBD_MWL_TARGET_SPACE_V0`, `TBD_MWL_TARGET_RANGE_V0`, `TBD_MWL_NUM_LEVELS_V0` are resolved.

Failure modes:

- unresolved placeholders
- mismatch between EEG acquisition metadata and the declared sampling/units/reference

### 8) Define “usable vs unusable” rules (run-level QC gates)

Objective:

- Convert data quality assumptions into explicit accept/reject rules.

Files touched:

- `docs/pilot/PILOT_STUDY_SPEC_V0.md` (QC rules)
- `docs/pilot/PILOT_READINESS_CHECKLIST_v0.md`

Success criteria:

- Run-level rejection rules exist for:
  - EEG timing (missing/non-monotonic)
  - marker completeness (all block boundaries present)
  - alignment tolerance violations
  - minimum valid-window fraction

Failure modes:

- silent inclusion of unusable runs
- inability to classify runs from recorded artifacts

### 9) Dry run: end-to-end recording on intended hardware

Objective:

- Produce a single researcher-run end-to-end session with sufficient evidence to prove alignment and contract-shaped window generation.

Files touched:

- `docs/pilot/DRY_RUN_VERIFICATION_V0.md` (new; non-sensitive verification report)
- External-only artifacts:
  - OpenMATB CSV + manifest
  - EEG recording file(s)
  - LSL recording file (if used)

Success criteria:

- OpenMATB run produces CSV + `.manifest.json` outside git with `ended_at_local` non-null.
- If LSL is used: the OpenMATB marker stream and EEG stream are recorded in the same session.
- A verification artifact exists (see Dry Run Protocol) proving:
  - windows matching the input contract can be generated
  - window labels can be assigned deterministically from markers
  - observed alignment error is within tolerance

Failure modes:

- manifest `ended_at_local` is null (crash/forced exit)
- missing marker stream or missing EEG stream
- cannot reconcile timebases deterministically

---

## Contract mapping (what gets resolved where)

Recommended: maintain a single ledger file `docs/pilot/PILOT_CONTRACT_RESOLUTIONS_V0.md` that lists the resolved values and where they are implemented/recorded.

Input contract (`docs/contracts/mwl_eeg_input_contract.md`) resolution checklist:

- `TBD_FIXED_WINDOW_S`
- `TBD_STEP_SIZE_S`
- `TBD_EFFECTIVE_FS_HZ`
- `TBD_CHANNELS_ORDERED_V0`
- `TBD_BANDPASS_HZ` (confirm)
- `TBD_MAINS_HZ`, `TBD_NOTCH_HARMONICS`
- `TBD_BAD_CHANNEL_POLICY_V0`
- `TBD_ARTIFACT_POLICY_V0`
- `TBD_BASELINE_REFERENCE_SEGMENTS_V0`
- `TBD_MIN_VALID_WINDOW_FRACTION`

Label contract (`docs/contracts/eeg_mwl_label_contract_v0.md`) resolution checklist:

- `TBD_MWL_TARGET_SPACE_V0`
- `TBD_MWL_TARGET_RANGE_V0`
- `TBD_MWL_NUM_LEVELS_V0`

---

## Dry Run Protocol (no participant recruitment; no modelling)

Goal: produce a single end-to-end recording plus a non-sensitive verification report demonstrating deterministic alignment and contract-shaped windowing.

### Pre-flight (before starting any recording)

1) Confirm authorization for the dry run (ethics approval or documented exemption/non-retention determination).
2) Confirm external data root exists and is writable (per `config/paths.yaml`).
3) Confirm OpenMATB venv is set up on the acquisition machine (per `docs/openmatb/SETUP_AND_RUN.md`).
4) Confirm the pilot scenario file to use (path under `src/python/vendor/openmatb/includes/scenarios/`) and set it in `src/python/vendor/openmatb/config.ini`.

### Recording procedure (researcher as participant)

1) Start the EEG acquisition software.
2) Start LSL recording (if used) and verify it sees:
   - EEG stream
   - OpenMATB marker stream `OpenMATB` (type `Markers`)
3) Run OpenMATB via wrapper with explicit IDs.

Command example (PowerShell; acquisition machine):

```powershell
cd src/python/vendor/openmatb
.\.venv\Scripts\Activate.ps1
python ..\..\run_openmatb.py --participant PDRYRUN --session S001
```

4) Complete the entire scenario including all workload blocks and NASA-TLX screens.
5) Exit cleanly (so the manifest is finalized).

### Required artifacts (external-only)

For the dry run session, record:

- OpenMATB CSV and adjacent `.manifest.json` (paths are recorded in the manifest).
- `last_scenario_errors.log` (must indicate no fatal scenario errors).
- EEG raw recording (format defined by acquisition system).
- If using LSL: the recorded LSL file containing EEG + OpenMATB markers.

### Verification checks (must be written up as a repo artifact)

Create a non-sensitive report:

- `docs/pilot/DRY_RUN_VERIFICATION_V0.md`

The report must include:

- the participant/session IDs used
- the absolute paths (external) to the CSV/manifest and EEG/LSL files
- manifest checks:
  - `ended_at_local` is non-null
  - `scenario_name` matches the intended pilot scenario
  - `lsl_enabled` matches expectation
- marker checks:
  - all block start/end markers present exactly once
  - markers are monotonic and bracket expected durations
- contract-shaped windowing checks:
  - window duration and step match `TBD_FIXED_WINDOW_S` / `TBD_STEP_SIZE_S`
  - windows can be generated across at least one full block
- deterministic label assignment checks:
  - each window center is assigned to a unique block label interval
  - unlabeled windows (if any) are explained and excluded per contract
- synchronization tolerance check:
  - measured alignment error $|\Delta t|$ is within the declared threshold

---

## Pilot Run Sheet (operator procedure; start → checks → run → stop → abort)

This is the operator-facing run procedure for both dry run and pilot sessions (pilot sessions require ethics hard gate completion).

### Before the participant arrives

- Verify ethics approval status for data collection (hard gate).
- Verify external data root is available and has sufficient disk space.
- Verify EEG hardware connections, impedance checks (per device SOP), and acquisition software configuration.
- Verify OpenMATB scenario selection in `src/python/vendor/openmatb/config.ini`.
- Verify LSL configuration (if used): acquisition software publishes EEG stream; recorder can see streams.

### Participant briefing and training

- Explain tasks and controls.
- Run structured training: 3 × 5 minutes (Low → Moderate → High), with short breaks.
- Confirm the participant can operate the joystick and respond to communications prompts.

## Instruction assets (English-only) — build requirement

### Goal
All on-screen instruction content used by the pilot one-shot scenario MUST be in English and stored in-repo. The French demo instruction screens shipped with OpenMATB MUST NOT be referenced by any pilot scenario.

### Required deliverables (one-shot build outputs)
The one-shot build must produce study-owned English instruction screens:

- welcome_screen_en.txt
- sysmon_en.txt
- track_en.txt
- communications_en.txt
- resman_en.txt
- scheduling_en.txt
- full_en.txt

These files must be placed in a location resolvable by OpenMATB scenario includes via relative paths (same resolution mechanism used by default/demo scenarios).

`nasatlx_en.txt` is already present and must remain the TLX screen.

### Content constraints
- Concise, on-screen readable (heading + ≤6 bullets).
- Include controls and success criteria per task; no theory, no workload/hypothesis language.
- Training/familiarisation screens may guide; retained-phase screens must not provide performance strategies.

### Hard disallow list (must not be referenced by pilot scenarios)
Pilot scenarios must not reference any of:
- welcome_screen.txt
- sysmon.txt
- track.txt
- communications.txt
- resman.txt
- scheduling.txt
- full.txt

### Acceptance checks (must pass before running participants)
- Repo search shows ZERO references in pilot scenarios to the disallowed French filenames.
- Spot-check: all *_en.txt files contain only English (no French strings).
- Dry run: instruction screens render correctly in the OpenMATB UI.

Example mechanical checks:
- `rg -n "welcome_screen\.txt|sysmon\.txt|track\.txt|communications\.txt|resman\.txt|scheduling\.txt|full\.txt" <pilot_scenario_dir>`
- `rg -n "Présentation|tâche|logiciel|l'écran|automation" <path_to_english_instructions>`

### Instruction language requirements (participant-facing)

All on-screen instructions must be written in clear, participant-facing plain English.

Mandatory rules:
- Instructions must NOT reference internal task names, plugin names, or system labels
  (e.g., SYSMON, RESMAN, COMM, scheduling, automation).
- Instructions must NOT use developer-style phrasing
  (e.g., “Do X to Y”, “Respond to SYSMON alerts”).
- Instructions must describe only:
  - what appears on the screen,
  - what the participant should do,
  - what counts as a correct response.
- Instructions must assume no prior knowledge.

Tone and structure:
- Short sentences.
- Imperative voice.
- One task per instruction screen.
- Concise and readable on screen.

Violation of these rules is a build blocker.


### Start of recording

- Start EEG recording.
- Start LSL recording (if used); confirm OpenMATB marker stream will be recorded.
- Launch OpenMATB via wrapper with participant/session IDs.
- Confirm the participant ID entry popup appears and is completed before training begins.

### During the run

- Monitor for abort criteria (below).
- Do not change scenario parameters mid-run.
- After each block: ensure the NASA-TLX screen is completed and all sliders are interacted with before continuing.

### Stop / clean exit

- End OpenMATB cleanly (do not force-kill) so the manifest finalizes.
- Stop EEG recording.
- Stop LSL recording.

### Abort criteria (examples; must be finalized in QC rules)

- EEG stream drops out or timestamps become non-monotonic.
- OpenMATB marker stream is not recorded (if LSL is required).
- OpenMATB exits unexpectedly (manifest `ended_at_local` would remain null).

### Restart criteria

- Restart only if the abort condition is detected immediately and a new session ID is used.
- Never overwrite prior external outputs.

---

## Readiness gates (aligned to the pilot readiness checklist)

Reference checklist: `docs/pilot/PILOT_READINESS_CHECKLIST_v0.md`.

### HARD GATE (ethics)

Must be true before any participant recruitment/recording:

- ethics approval granted and recorded
- consent/withdrawal procedure documented
- data storage and retention policy documented

### READY FOR DRY RUN (no recruitment)

Must be true before an internal researcher dry run:

- OpenMATB runs via `src/python/run_openmatb.py` on the acquisition machine
- external outputs + manifest written outside repo; `ended_at_local` set on clean exit
- pilot scenario file(s) pinned by path and filename
- synchronization strategy chosen and documented, including $|\Delta t|$ tolerance
- contracts have no unresolved `TBD_*`

### READY FOR PILOT DATA COLLECTION

Must be true before pilot recruitment/recording:

- all checklist items checked, including ethics hard gate
- dry-run verification report exists and demonstrates:
  - contract-shaped windows can be generated
  - deterministic label-to-window assignment works
  - alignment tolerance is met

---

## Decision Required (minimize ambiguity before implementation)

1) Reference timebase for EEG↔event alignment
   - Default: LSL timestamps (record EEG and OpenMATB markers via LSL in the same recording).
   - Alternative: derive alignment from OpenMATB CSV `logtime` + acquisition-system timestamps (only if deterministic mapping exists).

2) Maximum allowed alignment error $|\Delta t|$ tolerance
   - Default: 20 ms (tight enough for clear event timing, while acknowledging OS jitter).
   - Alternative: 50 ms (if acquisition/LSL path introduces unavoidable jitter; must be justified).

3) Workload block durations and full session structure
  - Default: session structure is exactly as defined in `docs/pilot/PILOT_STUDY_SPEC_V0.md` (training T1–T3 with breaks; retained B1–B3 with TLX after each retained block and breaks).
   - Alternative: 3 × 7 minutes if more stable EEG windows are needed; keep total session bounded.

4) Event rate targets per level (per task)
  - v0 targets are fixed and must match `docs/pilot/PILOT_STUDY_SPEC_V0.md` and `docs/contracts/training_scenario_contract_v0.md`.
  - Total event-rate targets (events/min): `LOW`=3, `MODERATE`=8, `HIGH`=18.
  - Scaling rule (v0): `HIGH` total event rate is ~5–6× `LOW` (operationalized as 18/3 = 6×).
  - Scheduling is deterministic and must follow the contract’s authoritative per-block template (integer second offsets, guard bands, collision rules).

5) Counterbalancing scheme
  - v0 retained counterbalancing is exactly the 3-sequence Latin-square defined by `seq_id` in `docs/pilot/PILOT_STUDY_SPEC_V0.md` (`SEQ1|SEQ2|SEQ3`).

6) Label target space choice under label contract
   - Default: 3-level ordinal label `{LOW, MODERATE, HIGH}` with `TBD_MWL_NUM_LEVELS_V0 = 3`.
   - Alternative: continuous numeric label in a bounded range (requires a defined mapping from block condition to numeric level).

7) Where the resolved parameter ledger lives (non-sensitive)
   - Default: `docs/pilot/PILOT_CONTRACT_RESOLUTIONS_V0.md` + `docs/pilot/PILOT_STUDY_SPEC_V0.md`.
   - Alternative: a single consolidated `docs/pilot/PILOT_STUDY_SPEC_V0.md` appendix.
