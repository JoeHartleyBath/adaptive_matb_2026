# ADR-0003: Real-time EEG-based MWL Adaptation Architecture

**Status:** proposed  
**Date:** 2026-01-27  
**Author:** (auto-generated implementation plan)

---

## Context

This repository requires a real-time closed-loop adaptation system where:
1. EEG is acquired continuously during MATB task performance
2. Mental workload (MWL) is estimated online from the EEG signal
3. MATB difficulty is adjusted dynamically based on MWL estimates
4. All decisions are logged deterministically for offline replay and debugging

This ADR documents the target architecture and a PR-by-PR implementation plan that respects existing repo rules.

---

## Current State Findings

### 1. Runner Infrastructure (`run_openmatb.py`)

**Location:** [src/python/run_openmatb.py](../../src/python/run_openmatb.py)

**What it does:**
- Wraps OpenMATB execution with repo-safe output paths
- Validates participant/session/seq-id identifiers
- Executes a *playlist* of scenarios in sequence (e.g., training blocks then calibration blocks)
- Injects environment variables for output routing and provenance
- Dynamically bootstraps OpenMATB with Clock speed patching, token substitution, and plugin shims
- Writes/patches `.manifest.json` files with run metadata (seq_id, dry_run, abort_reason, provenance)
- Detects early termination and scenario errors post-hoc

**Key integration point:** The `_run_single_scenario()` function constructs a bootstrap script and runs OpenMATB as a subprocess. Adaptation logic must either:
- (a) Run *inside* the bootstrap as a runtime hook (recommended), or
- (b) Run as an external controller communicating via IPC

*Evidence of feasibility:* The current `run_openmatb.py` already successfully uses method (a) to:
- Shim `AbstractPlugin.log_manual_entry` for compatibility
- Monkey-patch `Event.parse_from_string` for token substitution
- Monkey-patch `Window.display_session_id` to hide session numbers in replay mode

This confirms that injecting a custom `Scheduler` hook or listening plugin via the bootstrap script is a safe, established pattern in this repo.

### 2. Scenario Structure

**Location:** [scenarios/](../../scenarios/)

**Format:** OpenMATB plaintext scenario files with lines:
```
HH:MM:SS;plugin;command[;param]
```

**Difficulty parameters already exposed:**
- `track;taskupdatetime;42` (ms per frame — lower = harder)
- `track;joystickforce;3` (lower = harder)
- `resman;tank-X-lossperminute;320` (higher = harder)
- `sysmon` failure frequency via scheduled events
- `communications` prompt frequency via scheduled events

**Current pilot scenarios:** Static difficulty (LOW/MODERATE/HIGH) defined by event density and parameter values. No online adaptation.

### 3. LSL Marker Integration (Already Present)

**OpenMATB's `labstreaminglayer` plugin:**
- Creates an LSL *outlet* for markers (`type='Markers'`)
- Pushes structured marker strings at block start/end
- Can stream full session CSV rows if `streamsession=True`

**Current scenario usage:**
```
0:00:00;labstreaminglayer;start
0:00:00;labstreaminglayer;marker;STUDY/V0/calibration/LOW/START|pid=...
...
0:05:00;labstreaminglayer;marker;STUDY/V0/calibration/LOW/END|pid=...
0:05:00;labstreaminglayer;stop
```

**Gap:** OpenMATB has *no LSL inlet* — it cannot receive MWL values. This must be implemented externally.

### 4. Logging & Manifest Structure

**Log outputs (external, per DATA_MANAGEMENT.md):**
- `C:\data\adaptive_matb\openmatb\{participant}\{session}\sessions\*.csv`
- `*.manifest.json` adjacent to each CSV

**Manifest fields (current):**
```json
{
  "identifiers": {"seq_id": "SEQ1", ...},
  "seq_id": "SEQ1",
  "scenario_name": "pilot_calibration_low",
  "openmatb": {"scenario_path": "..."},
  "paths": {"session_csv": "...", "scenario_errors_log": "..."},
  "dry_run": false,
  "abort_reason": null
}
```

### 5. Existing Contracts

**Input contract:** `docs/contracts/mwl_eeg_input_contract.md` — **not yet written; proposals only**
- Proposed windowing: 5s window, 0.25s step, 250 Hz effective rate
- Proposed preprocessing: causal bandpass 0.5–40 Hz, CAR, notch
- Proposed representation: `EEGLogBandpowerFeatureVectorV0`
- Proposed normalisation: per-participant baseline → z-score
- All values are open until Pilot 1 EEG data quality is assessed

**Label contract:** `docs/contracts/eeg_mwl_label_contract_v0.md` — **not yet written; proposals only**
- Proposed MWL semantic: task-related demand/effort
- Proposed label sources: task-demand (block condition), subjective (NASA-TLX)
- Proposed window–label alignment: window centre timestamp
- All values are open until Pilot 1

**Adaptation design:** [docs/openmatb/ADAPTATION_DESIGN.md](../openmatb/ADAPTATION_DESIGN.md)
- Proposes two-process model (MWL estimator external, policy inside MATB loop)
- Recommends hysteresis + cooldown + step limits
- Identifies safe parameters to adapt (rates/pressure vs geometry)

### 6. Missing Components

| Component | Status |
|-----------|--------|
| EEG acquisition interface | Not implemented |
| Online preprocessing pipeline | Not implemented |
| Windowing + feature extraction | Not implemented |
| MWL inference runtime | Not implemented |
| Adaptation controller | Not implemented (design exists) |
| MWL → MATB IPC bridge | Not implemented |
| Per-participant calibration protocol | Not implemented |
| Adaptation logging (decision audit trail) | Not implemented |
| Simulated MWL for dry-run testing | Not implemented |
| Offline training data assembly | Not implemented |

---

## Target Architecture

### Integration strategy (recommended: external controller + minimal in-process bridge)

This ADR uses “bootstrap injection” into OpenMATB as the integration hook. To keep OpenMATB treated as upstream and to keep real-time EEG/model code isolated from UI timing jitter, the recommended operational model is:

- **Process A (repo-owned controller):** acquires EEG (or simulated MWL), runs preprocessing/features/inference, computes adaptation decisions, writes the run-level manifest + adaptation logs.
- **Process B (OpenMATB UI):** runs vendor OpenMATB unchanged on disk. It hosts a **minimal** in-process IPC receiver created via the bootstrap (no vendor edits) that:
  - accepts actuation commands
  - calls `plugin.set_parameter(key, value)` on the already-loaded plugins
  - logs each applied actuation into the OpenMATB CSV (e.g., via `logger.log_manual_entry(..., key='adaptation_actuation')`) so the task log remains the canonical timeline.

Rationale:
- Keeps EEG dependencies and model runtime swappable without touching the OpenMATB process.
- Makes latency profiling and failure isolation easier.
- Preserves “one place to look” debugging: OpenMATB CSV (canonical), plus an external adaptation decision log and a run-level manifest.

Bootstrap injection remains valuable, but is limited to the bridge + logging glue.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Acquisition Machine                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐         ┌──────────────────────────────────────────┐    │
│  │ EEG Amplifier │───LSL──▶│ EEG Recording & Online Processing       │    │
│  └───────────────┘         │                                          │    │
│                            │  ┌──────────────────────────────────┐    │    │
│                            │  │ eeg_inlet.py                     │    │    │
│                            │  │  - LSL inlet (raw EEG)           │    │    │
│                            │  │  - Ring buffer (N seconds)       │    │    │
│                            │  │  - Timestamp alignment           │    │    │
│                            │  └──────────────┬───────────────────┘    │    │
│                            │                 ▼                         │    │
│                            │  ┌──────────────────────────────────┐    │    │
│                            │  │ eeg_preprocessor.py              │    │    │
│                            │  │  - Causal bandpass (0.5–40 Hz)   │    │    │
│                            │  │  - Notch (50/60 Hz)              │    │    │
│                            │  │  - CAR re-reference              │    │    │
│                            │  │  - Bad-channel / artifact flags  │    │    │
│                            │  └──────────────┬───────────────────┘    │    │
│                            │                 ▼                         │    │
│                            │  ┌──────────────────────────────────┐    │    │
│                            │  │ eeg_windower.py                  │    │    │
│                            │  │  - 5s window, 0.25s step         │    │    │
│                            │  │  - Warmup tracking               │    │    │
│                            │  │  - Validity tagging              │    │    │
│                            │  └──────────────┬───────────────────┘    │    │
│                            │                 ▼                         │    │
│                            │  ┌──────────────────────────────────┐    │    │
│                            │  │ eeg_features.py                  │    │    │
│                            │  │  - Log-bandpower extraction      │    │    │
│                            │  │  - Normalisation (z-score)       │    │    │
│                            │  │  - Feature vector output         │    │    │
│                            │  └──────────────┬───────────────────┘    │    │
│                            │                 ▼                         │    │
│                            │  ┌──────────────────────────────────┐    │    │
│                            │  │ mwl_inference.py                 │    │    │
│                            │  │  - Load calibrated model         │    │    │
│                            │  │  - Predict MWL estimate          │    │    │
│                            │  │  - Output confidence + quality   │    │    │
│                            │  └──────────────┬───────────────────┘    │    │
│                            │                 │                         │    │
│                            └─────────────────┼─────────────────────────┘    │
│                                              │                               │
│                                              ▼ (IPC: LSL outlet or pipe)     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ OpenMATB + Adaptation Controller                                     │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ run_openmatb.py (bootstrap)                                    │ │  │
│  │  │                                                                │ │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │  │
│  │  │  │ adaptation_controller.py                                 │  │ │  │
│  │  │  │  - LSL inlet for mwl_estimate                           │  │ │  │
│  │  │  │  - MwlSmoother (EMA)                                    │  │ │  │
│  │  │  │  - AdaptationPolicy (hysteresis, cooldown, rate-limit)  │  │ │  │
│  │  │  │  - DifficultyActuator (plugin.set_parameter)            │  │ │  │
│  │  │  │  - AdaptationLogger (decision audit)                    │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  │                           │                                     │ │  │
│  │  │                           ▼ (hooks into Scheduler.update)       │ │  │
│  │  │                                                                │ │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │  │
│  │  │  │ OpenMATB Core (vendor, unmodified)                       │  │ │  │
│  │  │  │  - Scheduler.update(dt)                                  │  │ │  │
│  │  │  │  - Plugins (track, sysmon, comms, resman)                │  │ │  │
│  │  │  │  - Logger (CSV + LSL markers)                            │  │ │  │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │  │
│  │  │                                                                │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │ EEG Raw Recording (parallel, via LabRecorder or XDF writer)           ││
│  │  - Raw EEG + MATB markers → XDF file (external storage)               ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Interfaces & Contracts

The contracts below are intentionally explicit about **timestamps** and **window metadata** so that (a) online adaptation, (b) offline training dataset assembly, and (c) offline policy replay can share the same semantics.

Timebase rule (v0):
- **Primary alignment clock:** LSL timestamps.
- OpenMATB `scenario_time` remains useful for within-task timing, but the cross-device join key should be LSL time.

#### 1. `EegStreamConfig` (dataclass)

```python
@dataclass
class EegStreamConfig:
    stream_name: str           # LSL stream name (e.g., "ActiChamp-0")
    stream_type: str           # LSL type (e.g., "EEG")
    expected_srate: float      # Expected sampling rate (e.g., 500.0)
    expected_channels: list[str]  # Ordered channel names
    mains_freq: Literal[50, 60]   # For notch filter
    buffer_duration_s: float   # Ring buffer size (e.g., 10.0)


  #### 1b. `EegSampleChunk` (wire unit)

  In addition to the windowed representation, the online pipeline needs a raw “chunk” unit for buffering.

  ```python
  @dataclass
  class EegSampleChunk:
    stream_id: str
    ts_lsl_s: list[float]             # one timestamp per sample (or chunk start + fs)
    data_uv: "np.ndarray"            # shape (n_channels, n_samples)
    channel_map: list[str]            # ordered names
    fs_hz: float
    dropped_samples: int
    meta: dict                        # device name/serial, units, etc.
  ```
```

#### 2. `EegWindow` (dataclass)

```python
@dataclass
class EegWindow:
    data: np.ndarray           # Shape: (n_channels, n_samples)
    channels: list[str]        # Ordered channel names
    srate: float               # Effective sampling rate
    t_start: float             # LSL timestamp of first sample
    t_end: float               # LSL timestamp of last sample
    t_center: float            # Midpoint timestamp (for label alignment)
    is_valid: bool             # False if bad-channel/artifact contaminated
    quality_flags: dict        # {"bad_channels": [...], "artifact_detected": bool, ...}
```

#### 3. `MwlEstimate` (dataclass)

```python
@dataclass
class MwlEstimate:
  # MWL output
  value: float               # MWL estimate in target range (e.g., [0, 1])
  confidence: float          # Model confidence [0, 1]

  # Signal quality (separate from confidence)
  signal_quality: float      # Signal quality index [0, 1]
  window_is_valid: bool      # Whether source window passed validity checks
  quality_flags: dict        # e.g., {"dropout_frac": 0.0, "bad_channel_count": 0, "artifact": false}

  # Window metadata (needed for deterministic alignment and replay)
  window_t_start: float      # LSL timestamp of first sample in window
  window_t_end: float        # LSL timestamp of last sample in window
  window_t_center: float     # LSL timestamp of window midpoint (label contract uses window centre)

  # Provenance
  model_id: str              # External artifact identifier/path+hash
  normalization_source: Literal["participant", "global", "none"]
  timestamp_produced: float  # LSL timestamp when estimate was produced
```

#### 4. `DifficultyState` (dataclass)

```python
@dataclass
class DifficultyState:
  # Bounded, rate-limited parameters (normalized 0–1 internally, mapped to plugin values)
  # NOTE: v0 should prefer “rate/pressure” and event-injection actuators over layout changes.
  tracking_difficulty: float     # 0=easiest, 1=hardest
  sysmon_difficulty: float
  comms_difficulty: float
  resman_difficulty: float
    
    # Global overrides
    locked: bool                   # If True, ignore adaptation requests
    
    # Bounds (per-parameter)
    min_values: dict[str, float]
    max_values: dict[str, float]
    
    # Rate limits
    max_delta_per_step: float      # Max change per adaptation tick
    max_delta_per_minute: float    # Max cumulative change per minute


  #### 4b. Actuator surface (what we can change online)

  OpenMATB already exposes the actuation surface via `plugin.set_parameter(key, value)`.

  Recommended v0 actuator mapping (conservative defaults; bounded + rate-limited):
  - **Tracking (`track`)**: `taskupdatetime` (ms), `joystickforce` (int)
  - **System monitoring (`sysmon`)**: `alerttimeout` (ms) and failure toggles (`lights-*-failure`, `scales-*-failure`)
  - **Communications (`communications`)**: `radioprompt` injections (`own`/`other`) at policy-chosen times
  - **Resource management (`resman`)**: `pump-*-state=failure` injections; optionally tank leak parameters within safe bounds

  All applied actuator changes must be logged into:
  - OpenMATB CSV (canonical timeline)
  - External adaptation log (structured, replay-friendly)
```

#### 5. `AdaptationDecision` (dataclass)

```python
@dataclass
class AdaptationDecision:
    timestamp: float               # When decision was made (LSL time)
    scenario_time: float           # MATB scenario time
    
    # Input
    mwl_smoothed: float            # Smoothed MWL value
    mwl_raw: float                 # Raw MWL value
    signal_quality: float          # Signal quality at decision time
    
    # Decision
    action: Literal["increase", "decrease", "hold"]
    reason: str                    # Human-readable (e.g., "mwl_below_theta_low_for_T_hold")
    
    # State change
    difficulty_before: DifficultyState
    difficulty_after: DifficultyState
    parameters_changed: dict[str, tuple[Any, Any]]  # {param: (old, new)}
    
    # Policy state
    cooldown_remaining_s: float
    hold_counter_s: float
```

#### 6. `RunManifestAdaptive` (schema extension)

```json
{
  "identifiers": {...},
  "adaptation": {
    "mode": "adaptive|simulated|disabled",
    "model_id": "base_v1_participant_P001_calib_20260127T1400",
    "policy": {
      "theta_low": 0.35,
      "theta_high": 0.65,
      "t_hold_s": 3.0,
      "t_cooldown_s": 15.0,
      "max_delta_per_step": 0.1,
      "smoother_alpha": 0.1
    },
    "calibration": {
      "baseline_segments": ["rest_eyes_open", "rest_eyes_closed"],
      "calibration_timestamp": "2026-01-27T14:00:00Z",
      "normalization_stats_path": "...(external)..."
    }
  },
  "adaptation_log_path": "...(external)...",
  "eeg_recording_path": "...(external, XDF)...",
  "paths": {...}
}
```

---

## Latency Budget & Responsiveness

### End-to-End Latency Target: ≤ 500 ms (acquisition → difficulty change visible)

Important clarification (windowing latency):
- The v0 input contract uses fixed windows (suggested 5 s) and causal filtering.
- This implies an inherent “lookback” of the window length; the *actionable* responsiveness target is therefore the **post-window** latency (compute + decision + actuation), which should be well under the 0.25 s step.

| Stage | Budget | Notes |
|-------|--------|-------|
| EEG acquisition latency | ~10–30 ms | LSL network/driver latency |
| Preprocessing | ~5 ms | Causal IIR, vectorized |
| Windowing + features | ~10 ms | Single window extraction |
| Inference | ~5–20 ms | Small model, CPU-only acceptable |
| IPC (LSL outlet → inlet) | ~1–5 ms | Local loopback |
| Smoothing + policy decision | ~1 ms | Pure computation |
| Plugin parameter update | <1 frame | Next MATB update tick |
| **Total** | **~50–100 ms** | Well under 500 ms target |

### Recommended Update Rates

| Component | Rate | Rationale |
|-----------|------|-----------|
| EEG windowing step | 0.25 s (4 Hz) | Per input contract; sub-second responsiveness |
| MWL inference | 4 Hz | One estimate per window |
| Adaptation policy tick | 1 Hz | Decisions don't need to be faster than cooldown resolution |
| Difficulty parameter update | ≤ 1 Hz | Rate-limited by policy (cooldown 10–30 s typical) |

Additional v0 guidance:
- **Inference cadence:** 4 Hz is the maximum; if CPU load is high, degrade to 2 Hz by skipping every other window (must be logged).
- **Actuation cadence:** avoid sub-second changes; use dwell/cooldown so the participant perceives stable task conditions.

### Smoothing & Hysteresis (Anti-Oscillation)

1. **EMA smoother** on MWL estimates: α = 0.1 → effective time constant ~2.5 s
2. **Dual-threshold hysteresis**: θ_low = 0.35, θ_high = 0.65 (deadband in between)
3. **Hold timer**: MWL must exceed threshold for T_hold = 3 s before action
4. **Cooldown**: After any change, block further changes for T_cooldown = 15 s
5. **Rate limit**: Max Δdifficulty = 0.1 per step, max 0.3 per minute

### Fallback Behaviour (Signal Quality Drops)

1. If `signal_quality < 0.5` for > 2 s: **hold difficulty** (no changes)
2. If `signal_quality < 0.3` for > 5 s: **lock difficulty** at current level + emit warning marker
3. If MWL inference fails (exception, timeout): **hold** + log error
4. If LSL stream disconnects: **lock difficulty** + emit disconnect marker + auto-reconnect loop

---

## EEG Recording & Model Lifecycle

### Phase 1: Recording During Pilots/Study Runs

**Recording setup:**
- Run LabRecorder (or XDF writer) alongside MATB
- Streams captured: EEG raw, MATB markers (from labstreaminglayer plugin)
- Output: Single `.xdf` file per session in external storage

**Marker schema for alignment:**
```
STUDY/V0/calibration/{LEVEL}/START|pid=P001|sid=S001|seq=SEQ1
STUDY/V0/calibration/{LEVEL}/END|pid=P001|sid=S001|seq=SEQ1
```

**Post-session:**
- XDF file stored in `{data_root}/raw/eeg/{participant}/{session}/`
- MATB CSV + manifest stored in `{data_root}/openmatb/{participant}/{session}/`

**Run-level recording manifest (recommended, external JSON):**

In addition to the OpenMATB per-scenario manifest, write a **single run manifest** that ties together:
- OpenMATB manifests/CSVs for all playlist blocks
- EEG recording file path(s) (e.g., XDF)
- MWL estimate log path
- Adaptation decision log path
- calibration artifact path (if any)

This run manifest must live outside git, under a configured external root (see `config/paths.yaml` guidance in `docs/DATA_MANAGEMENT.md`).

### Phase 2: Offline Training Dataset Assembly

**Script:** `scripts/build_mwl_training_dataset.py`

**Process:**
1. Load XDF files (MNE-Python or pyxdf)
2. Extract EEG epochs aligned to block markers
3. Apply preprocessing pipeline (same as online: causal bandpass, notch, CAR)
4. Segment into 5s windows with 0.25s step
5. Extract log-bandpower features
6. Assign block-level MWL labels (LOW=0, MODERATE=0.5, HIGH=1.0)
7. Output: HDF5 or Parquet in `{data_root}/processed/training/`

**Contract compliance:** Same windowing, preprocessing, features as online pipeline.

### Phase 3: Base Model Training (Offline)

**Script:** `scripts/train_mwl_model.py`

**Process:**
1. Load training dataset from `{data_root}/processed/training/`
2. Train classifier/regressor (e.g., logistic regression, small MLP)
3. Compute global normalization statistics (training set mean/std)
4. Save model artifact + global stats to `{data_root}/models/base_v{N}/`

**Outputs (external, never in git):**
- `model.pkl` or `model.onnx`
- `global_norm_stats.json`
- `model_card.json` (training metadata, metrics)

### Phase 4: Per-Participant Calibration

**Calibration protocol:**
1. Participant does a short (~2 min) baseline: eyes-open rest + eyes-closed rest
2. EEG is recorded during baseline
3. Compute participant-specific normalization statistics (mean, std) from baseline
4. Optionally: run short calibration task (e.g., 1 block each of LOW/HIGH) to fit output scale/bias

**Calibration script:** `scripts/calibrate_participant.py`

**Outputs:**
- `{data_root}/calibration/{participant}/norm_stats.json`
- `{data_root}/calibration/{participant}/calibration_manifest.json`

### Phase 5: Deployable Model for Adaptive Session

**Runtime model loading:**
```python
model = load_model(f"{data_root}/models/base_v1/model.pkl")
norm_stats = load_json(f"{data_root}/calibration/{participant}/norm_stats.json")
```

**Inference:**
```python
features = extract_features(eeg_window)  # log-bandpower
features_norm = (features - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
mwl_estimate = model.predict(features_norm)
```

---

## PR Sequence

### PR 1: `feat(eeg): add EEG stream inlet and ring buffer`

**Files to add/change:**
- `src/python/eeg/eeg_inlet.py` (new)
- `src/python/eeg/__init__.py` (new)
- `docs/contracts/eeg_stream_config_v0.md` (new)

**Scope:**
- Implement `EegInlet` class using pylsl
- Configure via `EegStreamConfig` dataclass
- Ring buffer with thread-safe access
- Timestamp alignment utilities

**Acceptance tests:**
- Unit test with mock LSL outlet
- Integration test: connect to ActiChamp LSL stream, verify samples received
- Verify buffer overflow handling

**Rollback:** Delete `src/python/eeg/` directory

---

### PR 2: `feat(eeg): add causal online preprocessing pipeline`

**Files to add/change:**
- `src/python/eeg/eeg_preprocessor.py` (new)
- `src/python/eeg/eeg_filters.py` (new)

**Scope:**
- Causal IIR bandpass (0.5–40 Hz)
- Causal IIR notch (50/60 Hz configurable)
- Common average reference
- Bad-channel detection (threshold-based)
- Artifact flagging (amplitude threshold)

**Acceptance tests:**
- Unit tests with synthetic signals
- Verify causality (no future samples used)
- Benchmark: <5ms for 250 samples × 32 channels

**Rollback:** Delete added files

---

### PR 3: `feat(eeg): add windowing and feature extraction`

**Files to add/change:**
- `src/python/eeg/eeg_windower.py` (new)
- `src/python/eeg/eeg_features.py` (new)

**Scope:**
- Fixed-length windowing (5s window, 0.25s step per contract)
- Warmup tracking (don't emit windows until buffer full)
- Log-bandpower feature extraction (delta, theta, alpha, beta bands)
- Normalisation (z-score with provided stats)

**Acceptance tests:**
- Unit tests with synthetic data
- Verify window timing (correct t_start, t_end, t_center)
- Verify feature vector dimensionality matches contract

**Rollback:** Delete added files

---

### PR 4: `feat(mwl): add MWL inference runtime`

**Files to add/change:**
- `src/python/mwl/mwl_inference.py` (new)
- `src/python/mwl/__init__.py` (new)

**Scope:**
- Load model from external path (pickle or ONNX)
- Load normalisation stats from external path
- Predict MWL from feature vector
- Output `MwlEstimate` dataclass
- Confidence estimation (if model supports)

**Acceptance tests:**
- Unit test with mock model
- Integration test with saved model artifact
- Verify output schema matches contract

**Rollback:** Delete `src/python/mwl/` directory

---

### PR 5: `feat(mwl): add MWL estimate LSL outlet`

**Files to add/change:**
- `src/python/mwl/mwl_outlet.py` (new)

**Scope:**
- Create LSL outlet for MWL estimates (type='MWL')
- Push `MwlEstimate` as structured sample (JSON or named channels)
- Include signal quality and validity flags

**Acceptance tests:**
- Unit test: verify outlet creation and sample format
- Integration test: receive samples from outlet in separate process

**Rollback:** Delete file

---

### PR 6: `feat(adapt): add adaptation controller skeleton`

**Files to add/change:**
- `src/python/adaptation/adaptation_controller.py` (new)
- `src/python/adaptation/mwl_smoother.py` (new)
- `src/python/adaptation/adaptation_policy.py` (new)
- `src/python/adaptation/difficulty_state.py` (new)
- `src/python/adaptation/__init__.py` (new)

**Scope:**
- `MwlSmoother`: EMA implementation
- `AdaptationPolicy`: hysteresis, cooldown, rate-limit logic
- `DifficultyState`: parameter bounds and current values
- `AdaptationController`: orchestrates inlet → smoother → policy → decisions

**Acceptance tests:**
- Unit tests for each component
- Verify anti-oscillation: synthetic MWL hovering at threshold → no rapid toggling
- Verify rate limiting enforced

**Rollback:** Delete `src/python/adaptation/` directory

---

### PR 7: `feat(adapt): add difficulty actuator and logging`

**Files to add/change:**
- `src/python/adaptation/difficulty_actuator.py` (new)
- `src/python/adaptation/adaptation_logger.py` (new)

**Scope:**
- `DifficultyActuator`: maps normalized difficulty → plugin parameter values
- Calls `plugin.set_parameter()` on OpenMATB plugins
- `AdaptationLogger`: writes `AdaptationDecision` records to CSV
- Emits LSL markers for each adaptation event

**Acceptance tests:**
- Unit test actuator parameter mapping
- Verify log file written with correct schema
- Verify LSL markers emitted

**Rollback:** Delete added files

---

### PR 8: `feat(runner): integrate adaptation controller into run_openmatb.py`

**Files to add/change:**
- `src/python/run_openmatb.py` (modify)
- `scenarios/pilot_adaptive_template.txt` (new)

**Scope:**
- Add `--adaptation-mode` flag: `disabled|simulated|adaptive`
- In bootstrap script, inject adaptation controller hook into `Scheduler.update`
- Create adaptive scenario template (no pre-scheduled events, dynamic)
- Update manifest schema with adaptation metadata

**Note (preferred split for reversibility):**
- Keep “controller” code in repo process (outside OpenMATB).
- Use bootstrap injection only to start an IPC receiver thread and to log applied actuation events.
- This keeps the OpenMATB subprocess thin and reduces risk of UI/frame regressions.

**Acceptance tests:**
- Run with `--adaptation-mode=disabled`: behaves like current
- Run with `--adaptation-mode=simulated`: uses simulated MWL, logs decisions
- Verify manifest includes adaptation metadata

**Rollback:** Revert changes to `run_openmatb.py`, delete new scenario

---

### PR 9: `feat(sim): add simulated MWL source for dry-run testing`

**Files to add/change:**
- `src/python/mwl/simulated_mwl.py` (new)

**Scope:**
- Generate deterministic MWL waveforms (sine, ramp, step, noise)
- Simulate signal quality dropouts (configurable schedule)
- Seed-based reproducibility for regression testing
- LSL outlet compatible with real MWL stream format

**Acceptance tests:**
- Unit test: verify waveform shapes
- Verify determinism: same seed → same outputs
- Integration test with adaptation controller

**Rollback:** Delete file

---

### PR 10: `feat(calib): add per-participant calibration protocol`

**Files to add/change:**
- `scripts/calibrate_participant.py` (new)
- `scenarios/calibration_baseline.txt` (new)
- `docs/pilot/CALIBRATION_PROTOCOL_V0.md` (new)

**Scope:**
- Run short baseline recording (eyes-open, eyes-closed rest)
- Compute and save normalisation statistics
- Output calibration manifest

**Acceptance tests:**
- Run calibration with simulated EEG
- Verify stats file written
- Verify stats can be loaded by inference runtime

**Rollback:** Delete added files

---

### PR 11: `feat(data): add offline training dataset builder`

**Files to add/change:**
- `scripts/build_mwl_training_dataset.py` (new)
- `docs/contracts/training_dataset_schema_v0.md` (new)

**Scope:**
- Load XDF files from external storage
- Segment by block markers
- Apply preprocessing and feature extraction
- Assign labels from block level
- Output HDF5/Parquet to external storage

**Acceptance tests:**
- Run on mock XDF file
- Verify output schema matches contract
- Verify label alignment

**Rollback:** Delete added files

---

### PR 12: `feat(model): add base model training script`

**Files to add/change:**
- `scripts/train_mwl_model.py` (new)
- `docs/pilot/MODEL_TRAINING_V0.md` (new)

**Scope:**
- Load training dataset
- Train logistic regression or MLP
- Compute and save global normalization stats
- Save model artifact and model card

**Acceptance tests:**
- Train on synthetic data
- Verify model can be loaded by inference runtime
- Verify model card written

**Rollback:** Delete added files

---

### PR 13: `docs: add adaptation logging and replay specification`

**Files to add/change:**
- `docs/contracts/adaptation_log_schema_v0.md` (new)
- `docs/contracts/run_manifest_adaptive_v0.md` (new)

**Scope:**
- Document adaptation log CSV schema
- Document extended manifest schema
- Specify replay requirements

**Acceptance tests:**
- N/A (documentation only)

**Rollback:** Delete added files

---

### PR 14: `feat(replay): add offline adaptation replay tool`

**Files to add/change:**
- `scripts/replay_adaptation_run.py` (new)

**Scope:**
- Load adaptation log + EEG recording
- Replay MWL estimates through policy
- Compare actual vs replayed decisions
- Output discrepancy report

**Acceptance tests:**
- Replay a recorded run
- Verify identical decisions with identical inputs

**Rollback:** Delete file

---

### PR 15: `feat(analysis): update performance summariser for adaptive runs`

**Files to add/change:**
- `src/python/summarise_openmatb_performance.py` (modify)

**Scope:**
- Read adaptation logs (if present)
- Correlate performance metrics (RMSE, response time) with active difficulty state
- Output stratified metrics (e.g., "Performance during High Difficulty" vs "Global Performance")
- Ensure backward compatibility with static pilot runs

**Acceptance tests:**
- Run summary on adaptive session data and verify stratified output
- Run summary on old static session data and verify no regression

**Rollback:** Revert changes to `summarise_openmatb_performance.py`

---

## Testing & Validation

### Unit Tests (per PR)

Each PR includes unit tests for new modules. Run with:
```powershell
python -m pytest src/python/tests/ -v
```

### Integration Tests

1. **Simulated full loop:**
   ```powershell
  python src/python/run_openmatb.py --participant PTEST --session STEST --seq-id SEQ1 --adaptation-mode=simulated
   ```
   - Verify difficulty changes appear in logs
   - Verify adaptation log written

  Recommended additional check (reuse existing harness):
  - Run [src/python/verify_pilot.py](../../src/python/verify_pilot.py) to ensure scenario markers and segment boundaries remain contract-compliant under any new adaptive scenario templates.

  Recommended additional check (reuse existing summariser):
  - Use [src/python/summarise_openmatb_performance.py](../../src/python/summarise_openmatb_performance.py) on produced manifests to validate that performance metrics and marker segmentation remain parseable.

2. **EEG hardware loop:**
   - Connect EEG amplifier (LSL stream active)
   - Run calibration
   - Run adaptive session
   - Verify EEG recording captured
   - Verify adaptation decisions logged

### Latency Benchmarks

**Script:** `scripts/benchmark_latency.py`

**Metrics:**
- Preprocessing latency (ms per window)
- Feature extraction latency (ms per window)
- Inference latency (ms per window)
- IPC round-trip (ms)
- End-to-end (acquisition → decision logged)

**Acceptance criteria:**
- P95 end-to-end < 200 ms
- No sample drops under normal load

Add a “post-window” latency criterion:
- P95 time from `MwlEstimate.timestamp_produced` → actuation applied/logged in OpenMATB CSV should be < 300 ms.

### Stability / Anti-Oscillation Tests

**Script:** `scripts/test_policy_stability.py`

**Tests:**
- MWL oscillating around θ_low: verify ≤1 change per minute
- MWL step change: verify exactly 1 change after T_hold
- MWL drop in signal quality: verify difficulty locked

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| OpenMATB update breaks hook | High | Medium | Pin submodule; test before pull |
| LSL stream disconnects mid-run | High | Low | Auto-reconnect + lock difficulty |
| Model drift over session | Medium | Medium | Per-participant baseline; log quality |
| Oscillation despite hysteresis | Medium | Low | Tune T_hold, θ gap; log decisions |
| Latency spikes cause missed windows | Medium | Low | Ring buffer absorbs; log warnings |
| Participant distress from rapid changes | High | Low | Rate limits + operator override key |
| Bad calibration → poor inference | High | Medium | QC check on calibration quality |
| XDF file corruption | High | Low | Verify file integrity post-session |

### Guardrails Implemented

1. **Difficulty bounds:** Hard min/max per parameter (cannot exceed safe range)
2. **Rate limits:** Max change per step + per minute
3. **Operator override:** Keyboard shortcut to lock difficulty instantly
4. **Signal quality gate:** Auto-lock when quality drops
5. **Cooldown enforcement:** No rapid toggling
6. **Logging of all decisions:** Full audit trail for debugging

Additional guardrail (v0):
7. **Deterministic dry-run mode:** simulated MWL must be seed-driven and include deterministic dropout segments so adaptation logic can be regression-tested without EEG hardware.

---

## Open Questions

1. **Which EEG system is primary target?** (ActiChamp assumed; affects channel map)
2. **What are the exact difficulty parameter mappings?** (Need pilot data to calibrate)
3. **Should calibration include a short task block, or just rest?** (Impacts protocol duration)
4. **Is ONNX preferred over pickle for model portability?** (Affects inference dependencies)
5. **What is the minimum acceptable calibration quality threshold?** (Needs empirical validation)

---

## Summary

This plan provides a path from the current static-scenario pilot runner to a fully adaptive EEG-driven system in 14 small, testable PRs. Each PR is reversible and respects the existing repo rules. The architecture separates EEG processing from MATB adaptation, uses LSL for IPC, and includes extensive safeguards against oscillation, latency issues, and signal quality problems.

The definition of done for the overall system:
- ✅ Simulated MWL dry-run shows difficulty adapting + logs saved
- ✅ Real EEG recording captures raw + markers in XDF
- ✅ Swapping to real inference requires only config change
- ✅ Adaptation is responsive (≤500ms) but stable (no oscillation)
- ✅ Everything logged and reproducible

Implementation notes (repo constitution compliance):
- Do not commit raw EEG, XDF, OpenMATB session CSVs, or model weight artifacts.
- Use `config/paths.yaml` (local-only) to point to the external data root, per `config/paths.example.yaml` and `docs/DATA_MANAGEMENT.md`.
