# Pilot 1 Session Checklist

Physical setup steps before launching the runner. All software, stream, battery, and recording checks are automated — this document covers only things the code cannot verify.

---

## Physical Setup

### 1. EEG

- [ ] **Plug ethernet cable into PC** — without ethernet the EEG signal is very noisy (confirmed 2026-03-26)
- [ ] Unplug EEG amplifier chargers and laptop charger before recording starts (noise increases significantly when plugged in)
- [ ] Fit EEG cap on participant
- [ ] Open eego software and check impedances (target: < 30 kΩ for all channels)
- [ ] Confirm EEG is streaming to LSL in eego (two amplifiers → two EEG streams)
- [ ] Confirm EEG amplifier battery is adequate (check eego or hardware LED) — you will be prompted to confirm this at session start

### 2. Shimmer GSR3 (EDA)

- [ ] Attach Shimmer electrodes to participant (index and middle finger, non-dominant hand)
- [ ] Power on Shimmer device
- [ ] Pair via Bluetooth; confirm COM port in Device Manager (e.g., COM5)

### 3. Polar H10 (HR/ECG)

- [ ] Strap Polar H10 on participant (chest strap, moistened)
- [ ] Confirm device is worn and detects heart rate (LED indicates connection)

### 4. Participant

- [ ] Participant seated, electrodes attached, instructions read
- [ ] Remind participant: Communications task requires pressing **ENTER** to submit a frequency — tuning alone is scored as MISS

---

## Run the Session

```powershell
cd C:\phd_projects\adaptive_matb_2026
.\.venv\Scripts\Activate.ps1

python src/python/run_openmatb.py --pilot1 --eda-port COM5
```

The runner will automatically:
1. Start the Shimmer EDA streamer and verify EDA battery (fails if < 25%)
2. Start the Polar H10 streamer and verify HR battery (fails if < 20%)
3. Prompt to confirm EEG amplifier battery
4. Check all LSL streams are live and streaming data
5. Start the Python LSL recorder
6. Ask to confirm participant/session/sequence before first scenario
7. Stop all streamers and recorder when the session ends

To abort: press **Ctrl+C** — all subprocesses will be cleaned up automatically.

### Optional: record XDF via LabRecorder without using the GUI

This uses LabRecorder's Remote Control Socket (RCS) to start/stop recording.

1. Launch LabRecorder (RCS enabled):
    - Install location (local): `C:\LabRecorder`
    - Ensure RCS is enabled in LabRecorder config: `RCSEnabled=1`, `RCSPort=22345`
2. Run the session with integrated LabRecorder control:

```powershell
python src/python/run_openmatb.py --pilot1 --labrecorder-rcs --no-python-recorder --eda-port COM5 --labrecorder-required-stream "OpenMATB::Markers"
```

The runner will:
- Send `update/select all/filename/start` to LabRecorder RCS before the playlist
- Auto-compute the expected `.xdf` path under `C:\data\adaptive_matb\physiology\sub-...\ses-...\...`
- Send `stop` to LabRecorder RCS after the playlist

Fallback (manual control):

```powershell
python scripts/control_labrecorder_rcs.py start-bids --participant P001 --session S001 --print-expected-path
python src/python/run_openmatb.py --pilot1 --no-python-recorder --xdf-path <PATH_PRINTED_ABOVE> --eda-port COM5
python scripts/control_labrecorder_rcs.py stop
```

---

## Calibration-Only (self-testing)

```powershell
python src/python/run_openmatb.py --pilot1 --calibration-only --eda-port COM5
```

---

## Output Files

```
C:\data\adaptive_matb\
├── openmatb\P001\S001\
│   ├── session.csv                       # OpenMATB event log
│   ├── session.manifest.json             # Per-scenario metadata
│   ├── run_manifest_*.json               # Run-level manifest
│   └── run_manifest_*.qc_alignment.json  # Marker-alignment QC report
└── physiology\P001\S001\
    └── lsl_recording_*.jsonl             # Python LSL recorder output
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| EDA streamer fails to start | Check COM port in Device Manager; confirm Shimmer is paired and powered |
| Polar HR timeout | Ensure chest strap is on and moist; runner will scan automatically |
| EEG stream not found | Check eego software is streaming; expect 2 streams for dual-amp setup |
| Shimmer battery blocked (< 25%) | Charge Shimmer before session |
| Wrong participant loaded | Answer `n` at the confirmation prompt; rerun with `--participant P00X` |

### Add a new participant

```powershell
python scripts/generate_participant_assignments.py \
    --participant-ids P001 P002 P003 \
    --sequences SEQ1 SEQ2 SEQ3
```

Or edit `config/participant_assignments.yaml` directly.
