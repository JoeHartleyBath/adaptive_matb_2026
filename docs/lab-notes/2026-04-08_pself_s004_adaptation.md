# Lab notes — 2026-04-08 — PSELF S004 adaptation + MWL bug-fix session

## Goals

- Diagnose why real-time `p_high` was stuck at ~0.000 during S003 adaptation
- Fix the root cause and verify the fix before running S004
- Complete PSELF S004 adaptation condition
- Review adaptation behaviour (assists, toggles) from XDF

**Outcome: Root cause diagnosed and fixed. S004 adaptation ran successfully (p_high mean=0.398). One assist toggle fired (assist_on only). LSL event logging added to scheduler. Ready for S005.**

---

## Root cause diagnosis — p_high ≈ 0.000

### Symptom
During S003 adaptation, `p_high` published to the MWL LSL outlet was consistently ~0.011 (mean), whereas offline calibration blocks for the same participant yielded p_high ~0.371. The model was receiving near-zero probabilities despite valid EEG signal.

### Root cause: per-window fresh EegPreprocessor (IIR startup transient)
The inference loop in `mwl_estimator.py` was creating a **new `EegPreprocessor` for every 2-second window**. Because IIR bandpass and notch filters (Butterworth, zero-phase via `sosfilt`) require several seconds to settle from a zero-initial-state, every window was dominated by the filter's startup transient rather than real brain signal. Filtered RMS was ~1e-8 (near machine zero) compared to ~2e-6 offline. The model returned P(HIGH)≈0 because the features were all noise from the transient.

### Comparison: training vs. real-time (before fix)
| | Training (offline) | Real-time (before fix) |
|---|---|---|
| Filter initialisation | Once at file start, run continuously | Fresh per 2-s window |
| Filtered RMS (2-s window) | ~2e-6 | ~1e-8 (transient-dominated) |
| p_high mean | 0.371 | 0.011 |
| Windows in [0.05, 0.50) | ~87% | ~0% |

---

## Code changes made today

### 1. `src/mwl_estimator.py` — persistent filter architecture (major fix)

**What changed:**
- Added `preproc_cfg` and `decim_factor` to `_DualEegInlet.__init__`
- Added `_filt_buffer`, `_filt_write_ptr`, `_filt_samples_filled` (second ring buffer for filtered data)
- Added `_preprocessor: EegPreprocessor | None` field
- In `connect()`: initialise persistent `EegPreprocessor` once, no prewarm — runs continuously from first sample
- In `pull_chunk()`: after raw ring-buffer update, decimate chunk and pass through persistent filter → write to `_filt_buffer`
- Added `get_filt_window()`: slices from filtered ring buffer (analogous to `get_window()` on raw)
- In `run()`: removed per-window `EegPreprocessor`; quality check on raw decimated data; inference on `get_filt_window()`; warmup now waits until `_filt_samples_filled >= window_samples`
- Added `--debug-verbose` CLI flag for per-window feature diagnostics

**Verification (replay script):**

|Metric|Before fix|After fix|Offline reference|
|---|---|---|---|
|p_high mean|0.011|0.347|0.371|
|p_high std|0.018|0.119|0.120|
|Windows in [0.05, 0.50)|~0%|~87%|~87%|

Fix confirmed — real-time now matches offline.

---

### 2. `src/eeg/online_features.py` — FIXED_BANDS bug

**What changed:** `self.bands` was being re-assigned from `iaf_bands(resolved_iaf)` on every feature extraction call, using individual-alpha-frequency-adjusted bands. Training used `FIXED_BANDS` (fixed theta/alpha/beta). Changed to `self.bands = FIXED_BANDS` to match the training contract.

---

### 3. `src/run_full_study_session.py` — LabRecorder ordering bug

**What changed:** When resuming with `--start-phase 7`, `_ensure_labrecorder_running()` was called **after** `phase_pre_adaptation_baseline()`. If LabRecorder wasn't already running, the baseline would start without recording. Fixed by moving the `_ensure_labrecorder_running()` call to **before** the baseline/skip-refresh branch for all paths where `start_phase <= 7`.

---

### 4. `src/adaptation/mwl_adaptation_scheduler.py` — LSL event outlet (new)

**What changed:** Added a `pylsl.StreamOutlet("AdaptationEvents", "Markers", 1, cf_string)` in `_setup_adaptation()`, and `self._event_outlet.push_sample([action])` in `_run_mwl_policy()` immediately after each toggle fires.

**Why:** Previously, `assist_on` / `assist_off` events were only `print()`ed to the subprocess console. LabRecorder never captured them, so XDF files contained no record of adaptation toggles. Now they are published as LSL string markers and will appear in the XDF alongside physio streams.

**Stream details:**
- Name: `"AdaptationEvents"`, Type: `"Markers"`, 1 channel, irregular rate (`nominal_srate=0`)
- Sample values: `"assist_on"` or `"assist_off"` (plain string)

---

## S004 adaptation run

### Run command used
```powershell
.\.venv\Scripts\Activate.ps1; python src/run_full_study_session.py --participant PSELF --labrecorder-rcs --eda-auto-port --post-phase-verify --start-phase 7 --skip-stream-check --skip-baseline-refresh
```

`--skip-baseline-refresh` was used because this was primarily a debugging run; the LSL outlet for adaptation events had not yet been added. The model threshold from calibration (0.298) was correctly restored from `model_config.json` even at `--start-phase 7`.

### XDF file
`sub-PSELF_ses-S004_task-matb_acq-adaptation_physio.xdf` — recorded at 12:32

### MWL inference (post-run analysis via `_tmp_check_adaptation_events.py`)
| Metric | Value |
|---|---|
| MWL samples | 1708 |
| Duration | ~426.8 s |
| p_high mean | 0.398 |
| p_high std | 0.151 |
| Quality windows (quality=1) | 1708 / 1708 (100%) |
| Confidence mean | 0.315 |
| Confidence std | 0.183 |

p_high is now in a sensible range. Fix is working.

### Adaptation events
- **assist_on** fired once (~early in session)
- **assist_off** never fired
- Assistance was ON for the entire remainder of the session

### Why only one toggle?
With `threshold=0.298` and `hysteresis=0.02`:
- `assist_on` zone: p_high > 0.318
- `assist_off` zone: p_high < 0.278

With mean p_high=0.398, the signal sat comfortably above the on-threshold all session and never dropped below 0.278 long enough to trigger assist_off. This is not a code bug — the toggle logic worked correctly. The issue is calibration context (see below).

**Note:** These events were only `print()`ed in S004 (no LSL outlet yet). The XDF contains no adaptation markers for this run. The LSL outlet fix has now been added and will apply from S005 onwards.

### Confidence interpretation
Confidence = `|p_high − 0.5| × 2`. With p_high clustering at 0.30–0.45 (near the SVM decision boundary), mean confidence ~0.315 is expected. This is a structural property of the model's calibration, not a bug.

---

## Threshold and calibration notes

**Youden threshold for PSELF:** `0.298065` (from `C:\data\adaptive_matb\models\PSELF\model_config.json`)
- `youdens_j = 0.598764`
- `n_classes = 3`
- `calibrated_at = 2026-04-08T10:02:22`

The threshold is set at the Youden J-optimal cut for the 3-class P(HIGH) column. It is correct statistically.

**Likely explanation for p_high staying high during S004:** There was a large gap between calibration (10:02) and the adaptation run (12:32) due to debugging. During that gap:
- EEG net dried out / electrode impedances changed
- Participant fatigue/state shifted
- No fresh baseline was recorded before the run (`--skip-baseline-refresh` was used)

In the real study (S005+), calibration and adaptation will run back-to-back with a fresh baseline update immediately before the adaptation condition. This should bring the pre-task p_high distribution closer to the calibration distribution, making the threshold more meaningful and producing real toggles.

**Do not panic about single toggle in S004** — it was a debugging run under non-ideal conditions.

---

## Issues / risks

1. **S004 XDF missing adaptation events** — expected (LSL outlet was not present). S005 will have them.
2. **`--skip-baseline-refresh` used for S004** — acceptable for a debug run; do NOT skip for real S005 data collection.
3. **Calibration–adaptation gap** — keep this as short as practically possible in the real study. The baseline refresh (`phase_pre_adaptation_baseline`) is designed to re-anchor the model to the current session's resting state; skipping it inflates baseline p_high.
4. **Low confidence is structural** — p_high near 0.35–0.45 means SVM sits close to the decision boundary. This is inherent to the model and expected. Monitor whether it improves when calibration and adaptation are on the same day without a gap.

---

## Next actions

- [ ] **S005 run:** use full pipeline, no `--skip-baseline-refresh`, no `--skip-stream-check`. Run command:
  ```powershell
  .\.venv\Scripts\Activate.ps1; python src/run_full_study_session.py --participant PSELF --labrecorder-rcs --eda-auto-port --post-phase-verify --start-phase 7
  ```
- [ ] **After S005:** verify `AdaptationEvents` stream is present in XDF and contains `assist_on`/`assist_off` markers.
- [ ] **After S005:** re-run `_tmp_check_adaptation_events.py` on the new XDF and compare p_high/confidence/toggle count with S004.
- [ ] **If threshold still seems too low after S005:** consider whether to re-calibrate or apply a post-hoc threshold correction. Do not change the threshold ad hoc — document the decision.

---

## Files modified today

| File | Change |
|---|---|
| `src/mwl_estimator.py` | Persistent filter architecture in `_DualEegInlet`; `get_filt_window()`; warmup on filtered buffer; `--debug-verbose` flag |
| `src/eeg/online_features.py` | `FIXED_BANDS` fix — bands no longer re-assigned from IAF |
| `src/run_full_study_session.py` | LabRecorder launch ordering fix (before baseline, not after) |
| `src/adaptation/mwl_adaptation_scheduler.py` | LSL `AdaptationEvents` outlet added; `push_sample([action])` on each toggle |
