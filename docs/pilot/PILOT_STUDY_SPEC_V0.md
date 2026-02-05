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
- v0 pilot scenarios are exactly three combined session scenario files (one per calibration `seq_id`):
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq1.txt` (for `SEQ1`)
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq2.txt` (for `SEQ2`)
  - `src/python/vendor/openmatb/includes/scenarios/pilot_seq3.txt` (for `SEQ3`)
- Each scenario file begins with the same training segment (T1–T3), then runs calibration blocks (B1–B3) ordered per `seq_id`.
- Primary event log is CSV + adjacent `.manifest.json` written externally (outside repo).
- LSL outlet plugin exists: `src/python/vendor/openmatb/plugins/labstreaminglayer.py`.

---

## 1) Session structure (exact; training vs calibration)

Definitions:

- **Training blocks**: participant familiarization; not used for analysis/modeling.
- **calibration blocks**: the pilot blocks intended to be used for dry-run verification artifacts and pilot analyses (subject to ethics + QC gates).

Session phases (exact):

### Phase A — Setup (not timed; operator-controlled)

- Fit EEG cap, impedance checks, verify signal quality.
- Confirm LSL streams visible if LSL is used (EEG stream + OpenMATB marker stream).
- Confirm scenario selection in `config.ini`.

### Phase B — Training (not calibration)

Goal: stabilize task proficiency before any calibration data.

- A participant ID entry popup appears before T1, and the scenario does not proceed until a valid ID is submitted.

- Training Block T1: **Low** workload, **5:00**
- Break: **1:00** (quiet rest; no tasks)
- Training Block T2: **Moderate** workload, **5:00**
- Break: **1:00**
- Training Block T3: **High** workload, **5:00**

NASA-TLX during training (exact):

- TLX is **not** administered during training in v0.
  - Rationale (protocol stability): reduce interruptions while the participant is still learning controls.
  - Exception (operator discretion): if training indicates obvious inability to operate tasks, stop and reschedule; do not proceed to calibration blocks.

### Phase C — calibration pilot blocks (calibration)

Goal: collect comparable blocks at multiple workload levels with deterministic boundaries + subjective ratings.

- calibration Block B1: workload level per counterbalancing (see Section 2), **5:00**
- NASA-TLX: immediately after B1, **self-paced**, untimed; all sliders must be interacted with before continuing
- Break: **1:00**
- calibration Block B2: per counterbalancing, **5:00**
- NASA-TLX: immediately after B2, **self-paced**, untimed; all sliders must be interacted with before continuing
- Break: **1:00**
- calibration Block B3: per counterbalancing, **5:00**
- NASA-TLX: immediately after B3, **self-paced**, untimed; all sliders must be interacted with before continuing

Total planned calibration time (excluding setup):

- Task time: 15:00
- TLX time: self-paced (no fixed maximum)
- Breaks: 2:00

Pause policy (exact):

- No pausing during calibration blocks B1–B3.
  - If an interruption occurs or pause would be needed during B1–B3, the operator must abort the run and restart (do not pause/resume within a calibration block).
- Pause/resume state changes may still appear in OpenMATB CSV/runtime logs and/or operator notes, but they are not emitted as `STUDY/V0/` markers (see Section 4).

---

## 2) Workload levels and order control (exact)

Workload levels:

- `LOW`
- `MODERATE`
- `HIGH`

Training order (fixed):

- T1=LOW → T2=MODERATE → T3=HIGH

calibration order (counterbalanced):

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

Overlap rules:

- Distribution is determined by the OpenMATB difficulty logic / Vendor Logic generator.
- Minimum spacing between communications prompts: **8 seconds** (best effort target).
- Exact integer second non-overlap is not enforced.

Per-level event-rate targets (exact):

Definition (exact): the **total event rate** (events/min) is the sum of per-task event rates for:

- Sysmon gauge failure onsets
- Communications radio prompts
- Resman pump failures

Scaling guidance (Option A; review-aligned): the intended total event-rate increase from LOW to HIGH is **~6×**, operationalised as **18/3 = 6×**.

Notes:

- These are **targets** for scenario design; the scenario files must be deterministic once authored.
- “Events/min” is defined over the active 5:00 block duration, excluding TLX and breaks.

---

## 4) Marker set (exact) and where markers appear

Goal: every calibration block and questionnaire segment must be deterministically bracketed in logs.

Marker transport requirements (v0):

- **CSV**: all markers must appear in the OpenMATB CSV event log.
- **LSL**: all markers must appear as samples in the OpenMATB LSL marker stream **if LSL is the chosen synchronization timebase**.
- For LSL, use the existing `labstreaminglayer` plugin.

Marker naming format (exact):

- Prefix all markers with `STUDY/V0/`.
- Include participant/session IDs and calibration sequence ID in each marker payload to make stream mixing detectable.

Marker payload template (exact):

- `STUDY/V0/<MARKER_NAME>|pid=<P>|sid=<S>|seq=<SEQ>`

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
| `calibration/B1/<LEVEL>/START` | start of calibration block B1 | Yes | Yes | primary label interval start |
| `calibration/B1/<LEVEL>/END` | end of calibration block B1 | Yes | Yes | primary label interval end |
| `TLX/B1/START` | immediately before NASA-TLX for B1 | Yes | Yes | brackets TLX |
| `TLX/B1/END` | immediately after TLX for B1 | Yes | Yes | brackets TLX |
| `calibration/B2/<LEVEL>/START` | start of calibration block B2 | Yes | Yes | primary label interval start |
| `calibration/B2/<LEVEL>/END` | end of calibration block B2 | Yes | Yes | primary label interval end |
| `TLX/B2/START` | immediately before NASA-TLX for B2 | Yes | Yes | brackets TLX |
| `TLX/B2/END` | immediately after TLX for B2 | Yes | Yes | brackets TLX |
| `calibration/B3/<LEVEL>/START` | start of calibration block B3 | Yes | Yes | primary label interval start |
| `calibration/B3/<LEVEL>/END` | end of calibration block B3 | Yes | Yes | primary label interval end |
| `TLX/B3/START` | immediately before NASA-TLX for B3 | Yes | Yes | brackets TLX |
| `TLX/B3/END` | immediately after TLX for B3 | Yes | Yes | brackets TLX |
| `SESSION_END` | immediately before clean OpenMATB exit | Yes | Yes | anchors session end |
| `ABORT` | if run is aborted early | Yes | Yes | makes early termination machine-detectable |

Notes:

- `<LEVEL>` must be one of `LOW|MODERATE|HIGH`.
- Training markers exist for completeness but training blocks are not intended for analysis.
- Pause/resume state changes may appear in OpenMATB CSV/runtime logs and operator notes, but they are not represented as `STUDY/V0/` markers; interruptions during B1–B3 must be handled via `ABORT` + restart.
- Marker payloads MUST NOT contain semicolons (;). Semicolons are reserved for OpenMATB scenario field separation.

---

## 5) XDF↔CSV Marker Alignment QC (Pilot 1)

All physiological data (EEG, EDA) and OpenMATB markers are recorded together via LabRecorder into a single `.xdf` file. After each run, the alignment between LSL-transported markers and OpenMATB CSV timestamps must be verified.

### QC Thresholds (exact)

| Metric | Pass Threshold | Hard Fail |
|--------|----------------|-----------|
| Median absolute error | ≤ 20 ms | — |
| 95th percentile error | ≤ 50 ms | — |
| Drift (ms/min) | ≤ 5 ms/min | > 20 ms/min |
| Discontinuities | 0 | Any present |

### QC Procedure

1. Run pilot session with `--pilot1` flag
2. After playlist completes, provide LabRecorder `.xdf` path when prompted
3. Automatic QC runs via `src/python/verification/verify_xdf_alignment.py`
4. QC report written to `run_manifest_*.qc_alignment.json`
5. Run manifest updated with pass/fail status

### Manual QC (optional)

```powershell
python src/python/verification/verify_xdf_alignment.py \
    --xdf C:\data\adaptive_matb\physiology\P001_S001.xdf \
    --csv C:\data\adaptive_matb\openmatb\P001\S001\session.csv \
    --out qc_report.json
```

Or using run manifest:
```powershell
python src/python/verification/verify_xdf_alignment.py \
    --run-manifest C:\data\adaptive_matb\openmatb\P001\S001\run_manifest_*.json
```

---

## 6) Physiological Recording (Pilot 1)

### Recording Stack

- **EEG**: g.USBamp → LSL outlet (via g.NEEDaccess or g.HIsys)
- **EDA**: Shimmer GSR3 → LSL outlet (via `scripts/stream_shimmer_eda.py`)
- **Markers**: OpenMATB → LSL outlet (via `labstreaminglayer` plugin)
- **Recording**: LabRecorder captures all streams to single `.xdf` file

### EDA Streaming Setup

1. Pair Shimmer GSR3 via Bluetooth (note COM port)
2. Start EDA streamer:
   ```powershell
   python scripts/stream_shimmer_eda.py --port COM5
   ```
3. Verify stream appears in LabRecorder (name: `ShimmerEDA`, type: `EDA`)
4. Start LabRecorder recording
5. Run OpenMATB with `--pilot1`

EDA stream properties:
- Sample rate: ~51.2 Hz
- Channels: 1 (GSR in microsiemens)
- LSL type: `EDA`

### Pilot 1 Scope (offline-only)

Pilot 1 is **offline-only**:
- No real-time adaptation during task
- All physiology is recorded for post-hoc analysis
- Focus: validate recording pipeline, marker alignment, data quality

---

## 7) Provisional alignment tolerance wording (finalized for Pilot 1)

This wording is normative for v0. QC thresholds have been finalized based on the requirements in Section 5.

Finalized statement (exact):

- **Synchronization tolerance (Pilot 1):** The maximum allowed median absolute alignment error between task event markers (OpenMATB CSV) and LSL-transported markers (XDF) is **≤ 20 ms**.
- The 95th percentile must be **≤ 50 ms**.
- Clock drift must be **≤ 5 ms/min**, with hard failure at **> 20 ms/min**.
- Any timestamp discontinuities (jumps, out-of-order) are a hard fail.

Acceptance criterion for Pilot 1:

- All QC metrics must pass before physiology data is considered valid for analysis.
- Failed runs must be logged and either repeated or excluded with documented justification.
