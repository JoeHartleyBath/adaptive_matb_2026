# Pilot 1 Session Checklist

This checklist covers the complete procedure for running a Pilot 1 session with OpenMATB, EEG, and EDA recording.

## Pre-Session Setup (one-time)

### Software Dependencies

```powershell
# From repo root, activate venv
.\.venv\Scripts\Activate.ps1

# Verify dependencies
pip list | Select-String "pyxdf|pylsl|pyshimmer|PyYAML"
```

Required packages:
- `pylsl` - Lab Streaming Layer Python bindings
- `pyxdf` - XDF file parsing for QC
- `pyshimmer` - Shimmer GSR3 communication
- `PyYAML` - Configuration files

### Participant Assignment

Before first session, add participant to assignments:

```powershell
python scripts/generate_participant_assignments.py \
    --participant-ids P001 P002 P003 \
    --sequences SEQ1 SEQ2 SEQ3
```

Or manually edit `config/participant_assignments.yaml`.

---

## Equipment Setup

### 1. EEG Setup

- [ ] Fit EEG cap on participant
- [ ] Check impedances (target: < 5 kΩ)
- [ ] Verify EEG amplifier streaming to LSL
- [ ] Confirm LSL stream visible in LabRecorder (name: varies by amp)

### 2. EDA Setup

- [ ] Attach Shimmer GSR3 electrodes to participant
- [ ] Power on Shimmer device
- [ ] Pair via Bluetooth (note COM port, e.g., COM5)
- [ ] Start EDA streamer:

```powershell
python scripts/stream_shimmer_eda.py --port COM5
```

- [ ] Verify stream in LabRecorder (name: `ShimmerEDA`, type: `EDA`)

### 3. LabRecorder Setup

- [ ] Open LabRecorder
- [ ] Refresh streams and verify all present:
  - [ ] EEG stream
  - [ ] `ShimmerEDA` (type: EDA)
  - [ ] `OpenMATB` (type: Markers) - will appear when OpenMATB starts
- [ ] Set output directory: `C:\data\adaptive_matb\physiology\`
- [ ] Set filename template: `P%p%_%b.xdf` or similar

---

## Session Execution

### 1. Start LabRecorder

- [ ] Click "Start" in LabRecorder **before** running OpenMATB
- [ ] Note start time and filename

### 2. Run OpenMATB Session

```powershell
cd C:\phd_projects\adaptive_matb_2026
.\.venv\Scripts\Activate.ps1

# Full pilot session (familiarization + practice + calibration)
python src/python/run_openmatb.py --pilot1

# Or calibration-only (for self-testing)
python src/python/run_openmatb.py --pilot1 --calibration-only
```

Interactive prompts will ask for:
- Participant number (e.g., `1` → `P001`)
- Sequence is auto-assigned from `participant_assignments.yaml`
- Session is auto-incremented

### 3. During Session

- [ ] Monitor participant for issues
- [ ] **Do not pause** during calibration blocks (B1-B3)
- [ ] If must abort, use Ctrl+C and restart

### 4. Post-Session

- [ ] Stop LabRecorder recording
- [ ] Note final XDF filename
- [ ] When prompted by runner, enter XDF path

---

## QC Verification

### Automatic QC (via runner)

When using `--pilot1`, the runner automatically:
1. Prompts for XDF path after playlist completes
2. Runs XDF↔CSV marker alignment QC
3. Reports pass/fail with metrics

### Manual QC (if needed)

```powershell
python src/python/verification/verify_xdf_alignment.py \
    --run-manifest C:\data\adaptive_matb\openmatb\P001\S001\run_manifest_*.json
```

### QC Pass Criteria

| Metric | Pass Threshold |
|--------|----------------|
| Median absolute error | ≤ 20 ms |
| 95th percentile error | ≤ 50 ms |
| Drift | ≤ 5 ms/min |
| Discontinuities | None |

---

## Troubleshooting

### EDA Stream Not Appearing

```powershell
# Test pyshimmer connection
python scripts/stream_shimmer_eda.py --port COM5 --test
```

Common issues:
- Wrong COM port (check Device Manager)
- Shimmer not paired
- Battery low

### OpenMATB Markers Not in XDF

- Ensure LabRecorder started **before** OpenMATB
- Check `labstreaminglayer` plugin enabled in OpenMATB config
- Verify "OpenMATB" stream appears in LabRecorder

### QC Fails with High Drift

- Check LabRecorder was running throughout entire session
- Verify no stream dropouts
- May need to repeat session

---

## Output Files

After a successful session:

```
C:\data\adaptive_matb\
├── openmatb\
│   └── P001\
│       └── S001\
│           ├── session.csv                    # OpenMATB event log
│           ├── session.manifest.json          # Per-scenario metadata
│           ├── run_manifest_*.json            # Run-level manifest
│           └── run_manifest_*.qc_alignment.json  # QC report
└── physiology\
    └── P001_S001.xdf                          # LabRecorder output
```

---

## Quick Reference

### Start EDA Streaming
```powershell
python scripts/stream_shimmer_eda.py --port COM5
```

### Run Pilot Session
```powershell
python src/python/run_openmatb.py --pilot1
```

### Run QC Only
```powershell
python src/python/verification/verify_xdf_alignment.py --run-manifest <path>
```

### Check Assignments
```powershell
cat config/participant_assignments.yaml
```
