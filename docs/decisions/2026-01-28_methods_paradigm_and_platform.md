# Methods decisions — paradigm and task platform (v0)

Status: draft (method decision record)

Last updated: 2026-01-28

Purpose: capture *method-level* decisions about the experimental paradigm and platform (MATB(-II) via OpenMATB), including what must be recorded for replication.

---

## Decision 1: Use an MATB(-II)-style multitask paradigm

**Decision statement:** The pilot protocol uses a multi-task MATB(-II)-style paradigm (sysmon + tracking + communications + resource management, plus scheduling display).

**Why this matters (replication):** Defines the cognitive/task ecology, performance outputs available, and the alignment problem (continuous EEG + dense task events).

**Rationale (fill in / confirm):**
- [ ] Why MATB(-II) rather than a single-task workload manipulation (e.g., n-back, oddball)?
- [ ] Why a *multitask* workload construct is needed for the study’s aims.

**Alternatives considered (fill in):**
- [ ] Single-task paradigms (n-back, Stroop, SART, etc.)
- [ ] Driving / flight simulator paradigms
- [ ] Custom-built multitask UI

**Implementation evidence (current repo):**
- Task set and design principle are specified in [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- OpenMATB is used as the concrete MATB implementation (see below).

**Replication record (must capture):**
- [ ] Exact task set enabled (sysmon/track/communications/resman/scheduling) and plugin parameters that differ from OpenMATB defaults.
- [ ] Hardware setup relevant to input control (mouse/joystick), display resolution, audio, and any operator procedures that affect task behavior.

---

## Decision 2: Use OpenMATB as the MATB implementation

**Decision statement:** Use OpenMATB (vendor implementation) as the executable task platform.

**Rationale (fill in / confirm):**
- [ ] Why OpenMATB vs other MATB implementations (MATB-II distribution, other forks, commercial toolkits).
- [ ] Why an open-source baseline is required (auditability, reproducibility, no licensing ambiguity).
- [ ] Why OpenMATB’s instrumentation surfaces (CSV + manifest + LSL markers) meet requirements.

**Alternatives considered (fill in):**
- [ ] Official MATB-II distribution (if available)
- [ ] Other open-source MATB implementations
- [ ] In-house task implementation

**Implementation evidence (current repo):**
- OpenMATB is integrated as a vendor subtree under `src/python/vendor/openmatb/`.
- Wrapper entrypoint: [src/python/run_openmatb.py](../../src/python/run_openmatb.py).

**Replication record (must capture):**
- [ ] The exact OpenMATB commit hash used (submodule commit).
- [ ] Any OpenMATB configuration (`config.ini`) that affects task behavior.

---

## Decision 3: Treat OpenMATB as upstream; integrate via wrapper + bootstrap injection

**Decision statement:** Do not edit vendor OpenMATB for study logic; integrate via a repo-owned wrapper and runtime bootstrapping.

**Rationale (evidence-based):**
- Reduces maintenance burden across upstream updates.
- Keeps study logic (IDs, provenance injection, scenario rewriting, verification speed-ups) in a single repo-controlled layer.

**Evidence / existing decisions:**
- Wrapper-per-block rationale: [docs/decisions/ADR-0002-wrapper-run-per-block-for-clean-task-state.md](ADR-0002-wrapper-run-per-block-for-clean-task-state.md).
- Architecture overview: [docs/decisions/ADR-0003-realtime-eeg-adaptation-architecture.md](ADR-0003-realtime-eeg-adaptation-architecture.md).

**Replication record (must capture):**
- [ ] The exact wrapper CLI invocation (participant/session/seq_id; flags).
- [ ] Wrapper-controlled env vars (output routing; provenance fields).
- [ ] Any bootstrap patches enabled (e.g., verification speed).

---

## Decision 4: Use OpenMATB CSV + manifest as canonical provenance, and LSL markers for synchronization (when EEG is used)

**Decision statement:** CSV is the canonical event timeline for OpenMATB; LSL markers are emitted for cross-device alignment when EEG is recorded.

**Implementation evidence:**
- Marker requirements and names: [docs/pilot/PILOT_STUDY_SPEC_V0.md](../pilot/PILOT_STUDY_SPEC_V0.md).
- Logging verification: [docs/run_logging_verification.md](../run_logging_verification.md).

**Replication record (must capture):**
- [ ] Whether LSL was used; if yes, what recorder/software captured the streams.
- [ ] Stream names/types (OpenMATB markers; EEG).
- [ ] Chosen timebase for EEG-label alignment.

---

## Open items / TBD

- [ ] Add a short “Why MATB(-II)” paragraph in study terms (hypotheses/aims) once finalized.
- [ ] Record why this specific OpenMATB fork/version was selected (and any Windows constraints).
