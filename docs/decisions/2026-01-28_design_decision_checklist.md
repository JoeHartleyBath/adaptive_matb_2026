# 2026-01-28 — Design Decision Checklist (Replication Ledger)

This document is a **living checklist** and **decision ledger** for the adaptive MATB (OpenMATB-based) study repo.

Scope:
- **Includes** repo-owned docs/code + **how we use** vendor/submodule code.
- **Excludes** vendor file contents as “design decisions” (but captures integration/usage decisions).

Goal: ensure **every study-relevant design choice** has:
- a clear justification
- sufficient parameters/versions to replicate
- a verification method (or acceptance criteria)

---

## How to use this

- Treat this file as the **index**.
- For any non-trivial choice, create/maintain a dedicated ADR-style note in `docs/decisions/` and link it from here.
- When a choice is still open, record it under **Pending decisions** with an explicit deadline/owner (you).

### Method decision docs (split by topic)

These are the method-focused decision notes created to support replication write-up:

- [2026-01-28_methods_paradigm_and_platform.md](2026-01-28_methods_paradigm_and_platform.md)
- [2026-01-28_methods_block_structure_training_calibration_tlx.md](2026-01-28_methods_block_structure_training_calibration_tlx.md)
- [2026-01-28_methods_workload_levels_and_operationalisation.md](2026-01-28_methods_workload_levels_and_operationalisation.md)
- [2026-01-28_methods_counterbalancing_and_seq_id.md](2026-01-28_methods_counterbalancing_and_seq_id.md)
- [2026-01-28_methods_eeg_preprocessing_filters_and_reference.md](2026-01-28_methods_eeg_preprocessing_filters_and_reference.md)
- [2026-01-28_methods_eeg_calibration_and_baseline.md](2026-01-28_methods_eeg_calibration_and_baseline.md)

---

## Decision record template (copy/paste)

Use this template for any new decision entry.

- [ ] **Decision:** <one sentence>
- [ ] **Status:** proposed | accepted | superseded | deprecated
- [ ] **Date:** YYYY-MM-DD
- [ ] **Context / problem:** <what forced the decision>
- [ ] **Chosen option:** <what we do>
- [ ] **Alternatives considered:** <brief list>
- [ ] **Rationale:** <why this is best for the study>
- [ ] **Implications:** <what this affects; risks>
- [ ] **Parameters to record:** <values + units>
- [ ] **Where implemented:** <file paths / scripts / configs>
- [ ] **Evidence / references:** <docs links, papers, vendor docs>
- [ ] **Replication steps:** <exact steps someone else would run>
- [ ] **Verification:** <how to confirm it works; tolerances>

---

## Replication checklist (high-level)

### Repository + provenance

- [ ] Record repo commit hash (and confirm working tree clean)
- [ ] Record OpenMATB submodule commit hash
- [ ] Record whether vendor code was modified (should be “no” unless explicitly documented)
- [ ] Record OS + version (Windows assumed for current tooling)
- [ ] Record machine + hardware notes relevant to timing (CPU, display refresh, USB hubs)

### Python environment + dependencies

- [ ] Record Python version (major.minor.patch)
- [ ] Record `pip freeze` for the repo venv
- [ ] Record vendor/OpenMATB Python requirements (from submodule `requirements.txt`)
- [ ] Record how packages were installed (e.g., `pip install -r …`, manual)

### Data boundaries + filesystem layout

- [ ] Confirm raw/PII data is **not** committed (policy)
- [ ] Record external data root location (local, not committed)
- [ ] Record the directory layout used for runs and outputs
- [ ] Confirm outputs are **outside the git repo**

### Experiment runner + execution contract

- [ ] Record runner script and exact CLI used (participant/session/seq)
- [ ] Record environment variables used (especially output root)
- [ ] Record playlist mapping and counterbalancing approach
- [ ] Record attended vs verification (fast-forward) mode behavior

### Scenarios + instructions

- [ ] Record exact scenario files used (names + hashes)
- [ ] Record whether scenarios were generated and by which script/version
- [ ] Record block durations and timing assumptions
- [ ] Record instruction text versions (what participants saw)
- [ ] Record NASA-TLX integration choices (when, where, instrument)

### Logging + event schema

- [ ] Record canonical data sources (OpenMATB CSV + manifest)
- [ ] Record marker naming scheme + token substitution rules
- [ ] Record performance metric extraction rules
- [ ] Record any post-processing outputs (derived JSON summaries)

### Timing + alignment

- [ ] Record which timebase is used for cross-device alignment (e.g., LSL time)
- [ ] Record tolerances and verification method for timing alignment
- [ ] Record any clock-speed manipulation (verification fast-forward)

### EEG acquisition + preprocessing

- [ ] Record EEG stream discovery rules (LSL name/type/source_id)
- [ ] Record sampling rate and expected channel order/labels
- [ ] Record buffering parameters (seconds)
- [ ] Record preprocessing pipeline (filters, CAR, mains frequency)
- [ ] Record validation checks for the preprocessing chain

### Verification + acceptance criteria

- [ ] Run static scenario/instruction validation checks (repo-level)
- [ ] Verify OpenMATB logging and manifest content (atomic writes, commit hashes)
- [ ] Verify no-overwrite behavior across two runs
- [ ] Verify marker payload correctness (no token leaks)
- [ ] Verify segment duration tolerances and event count expectations

---

## Current decisions inventory (found in repo)

This section enumerates explicit decisions (ADRs) plus replication-critical implicit decisions implemented in repo-owned code.

### Repo structure + governance

- [x] **Repo structure and data boundaries**
  - Evidence: [docs/decisions/ADR-0001-repo-structure.md](ADR-0001-repo-structure.md)
  - Key points to replicate: separation of code vs external data storage; no raw/PII in git.

- [x] **Solo-optimised workflow and “clarity-first” rules**
  - Evidence: [docs/REPO_RULES.md](../REPO_RULES.md), [docs/WORKFLOW.md](../WORKFLOW.md), [docs/STYLEGUIDE.md](../STYLEGUIDE.md)

- [x] **Ignore local environments and local path configs**
  - Evidence: [.gitignore](../../.gitignore), [config/paths.example.yaml](../../config/paths.example.yaml)
  - Decision: do not commit `.venv` and do not commit `config/paths.yaml`.

### Third-party (vendor/submodule) integration

- [x] **OpenMATB included as a git submodule under `src/python/vendor/openmatb`**
  - Evidence: [.gitmodules](../../.gitmodules)
  - Replication-critical: record the submodule commit used for any run.

- [x] **Treat OpenMATB as upstream; integrate via wrapper + runtime bootstrap injection**
  - Evidence: [docs/decisions/ADR-0003-realtime-eeg-adaptation-architecture.md](ADR-0003-realtime-eeg-adaptation-architecture.md)
  - Implementation: wrapper constructs an injected bootstrap script rather than editing vendor files.

### Pilot block structure + counterbalancing

- [x] **One OpenMATB process per block (playlist of scenario files)**
  - Evidence: [docs/decisions/ADR-0002-wrapper-run-per-block-for-clean-task-state.md](ADR-0002-wrapper-run-per-block-for-clean-task-state.md)
  - Implementation: `run_openmatb.py` runs each scenario filename sequentially.

- [x] **Counterbalancing encoded as three calibration-order sequences (SEQ1/SEQ2/SEQ3)**
  - Implementation: playlist mapping in `run_openmatb.py`:
    - SEQ1: LOW → MODERATE → HIGH
    - SEQ2: MODERATE → HIGH → LOW
    - SEQ3: HIGH → LOW → MODERATE
  - Replication-critical: document how participants are assigned to `SEQ*`.
  - Related (deferred): [docs/decisions/TODO-counterbalancing.md](TODO-counterbalancing.md)

### Runner contract (OpenMATB wrapper)

- [x] **External output root (default Windows path) and safe identifiers**
  - Implementation: `OPENMATB_OUTPUT_ROOT` default `C:\data\adaptive_matb`; IDs restricted to `[A-Za-z0-9_-]+`.
  - Replication-critical: record chosen output root, and confirm outputs are outside repo.

- [x] **Run provenance injected into OpenMATB manifests**
  - Implementation: `run_openmatb.py` collects newly created `*.manifest.json` and writes:
    - `seq_id`, `scenario_name`, `dry_run`, `unattended=false`, `abort_reason` (if applicable)
    - `openmatb.scenario_path` and `identifiers.seq_id`
  - Replication-critical: ensure exactly one new manifest is produced per block.

- [x] **Token substitution for scenario marker payloads**
  - Implementation: bootstrap patches `Event.parse_from_string` to replace:
    - `${OPENMATB_PARTICIPANT}`, `${OPENMATB_SESSION}`, `${OPENMATB_SEQ_ID}`

- [x] **Pilot instruction files are repo-managed but staged into vendor `includes/`**
  - Implementation: copy `instructions/*.txt` into `includes/instructions/pilot_en/`.
  - Replication-critical: keep instruction content versioned with the repo commit.

- [x] **Scenario compatibility rewrites before running OpenMATB**
  - Implementation: wrapper rewrites:
    - legacy instruction paths like `../../../../assets/instructions/pilot_en/` → `pilot_en/`
    - `tank-A-...` → `tank-a-...` (case sensitivity)
    - `genericscales;load;` → `genericscales;filename;` and removes invalid `genericscales;create`

- [x] **Verification fast-forward supported, but disabled in attended runs**
  - Implementation: `--speed` is ignored unless `--verification` is set; bootstrap wires speed into `Clock._speed`.

- [x] **Abort detection policy**
  - Implementation: treat as failed if:
    - OpenMATB writes actionable scenario errors, or
    - observed `scenario_time` in CSV is < 90% of the scenario’s max scheduled time

### Scenario design + markers

- [x] **Markers use a fixed naming scheme for segmentation**
  - Pattern: `STUDY/V0/<SEGMENT>/START|pid=...|sid=...|seq=...` and matching `.../END|...`
  - Implemented in generated calibration/training scenarios (e.g., `pilot_calibration_low.txt`).

- [x] **Training blocks fixed order; calibration blocks include NASA-TLX**
  - Training marker names: `TRAINING/T1`, `TRAINING/T2`, `TRAINING/T3`.
  - calibration marker names: `calibration/LOW|MODERATE|HIGH`.
  - NASA-TLX injected after calibration block using vendor `genericscales` + `nasatlx_en.txt`.

- [x] **Block duration is 300 seconds for generated blocks**
  - Implemented in `generate_pilot_scenarios.py` (`BLOCK_DURATION_SEC = 300`).

- [x] **Difficulty mapping for generated blocks**
  - Implemented in `generate_pilot_scenarios.py`:
    - LOW=0.2, MODERATE=0.55, HIGH=0.95
  - Replication-critical: record these values; changing them changes event schedules.

### Scenario generation (repo-owned use of vendor logic)

- [x] **Scenario generator uses copied/adapted vendor scheduling logic**
  - Evidence: comments in `generate_pilot_scenarios.py` (“Copied/Adapted from …/vendor/…/scenario_generator.py”).
  - Replication-critical: record that schedules are pseudo-random unless seeded (currently unseeded).

- [x] **Tracking difficulty control surface**
  - Implemented mapping (difficulty → parameters):
    - `taskupdatetime` linear from 50 ms (easy) to 10 ms (hard)
    - `joystickforce` linear from 3 (easy) to 1 (hard)

- [x] **Resman additional pump failure generation**
  - Implemented in `generate_pilot_scenarios.py` with a linear formula (documented in code).

### Logging + derived summaries

- [x] **OpenMATB CSV is the canonical event timeline; manifest links to it**
  - Evidence: [docs/run_logging_verification.md](../run_logging_verification.md)

- [x] **Derived performance summaries are written as JSON (best-effort)**
  - Implemented: `summarise_openmatb_performance.py` (reads `type=performance` rows; segments by LSL markers).
  - Runner integration: optional `--summarise-performance`.

- [x] **No-overwrite and provenance verification practices**
  - Evidence: [docs/two_run_no_overwrite_verification.md](../two_run_no_overwrite_verification.md), [docs/pre_merge_logging_verification.md](../pre_merge_logging_verification.md)

### EEG acquisition + preprocessing

- [x] **LSL is the acquisition interface**
  - Implemented: `EegInlet` uses `pylsl` to resolve/pull chunks.

- [x] **Stream identity defaults**
  - Implemented defaults in `EegStreamConfig`:
    - `stream_name='eego'`, `stream_type='EEG'`, `expected_srate=500.0`, `mains_freq=50`, buffer 10s
  - Replication-critical: record the actual device stream name/type/source_id.

- [x] **Ring-buffer format and chunk orientation**
  - Implemented: buffer stored as `(n_channels, n_samples)`; incoming chunks are transposed to match.

- [x] **Preprocessing pipeline**
  - Implemented: bandpass (Butterworth SOS), notch (IIR notch via SOS), optional CAR.
  - Default parameters in `EegPreprocessingConfig`: 0.5–40 Hz bandpass (order 4), 50 Hz notch (Q=30), CAR enabled.

- [x] **Preprocessing verification by synthetic-signal test**
  - Implemented: `verify_preprocessing.py` checks drift removal, 50 Hz suppression, and 10 Hz preservation.

### Pilot verification harnesses

- [ ] **Unified pilot run+verify harness exists but currently references a missing module**
  - Observed: `src/python/verification/verify_pilot.py` imports `verify_pilot_scenarios` which is not present in repo.
  - Replication impact: dynamic verification cannot run until the missing verifier module is restored/renamed.

- [x] **EEG inlet connectivity verification**
  - Implemented: `src/python/verification/verify_eeg_inlet.py` includes an optional mock LSL outlet.

---

## Pending decisions (explicit)

- [ ] Inference-engine decisions list
  - Evidence: [docs/decisions/PENDING_DECISIONS_INFERENCE.md](PENDING_DECISIONS_INFERENCE.md)

---

## Open replication gaps to close (actionable)

These are not “new scope”; they are required to make replication defensible.

- [ ] Add a repo-owned dependency lockfile or install instructions capturing non-vendor deps (numpy/scipy/pylsl, etc.)
- [ ] Resolve missing `verify_pilot_scenarios` module (restore file or update import)
- [ ] Add/restore the referenced dry-run scenario (`pilot_dry_run_v0.txt`) or remove `--dry-run` pathway
- [ ] Decide and document participant→SEQ assignment rule (counterbalancing procedure)

