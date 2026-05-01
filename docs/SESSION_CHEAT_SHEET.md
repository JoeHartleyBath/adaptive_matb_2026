# Adaptive MATB — Session Operator Cheat Sheet

> **This document is the primary reference for running a participant session.**
> Read it end-to-end before your first session. Keep it open throughout.

---

## Quick nav

- [A. Before participant arrives](#a-before-participant-arrives)
- [B. When participant arrives](#b-when-participant-arrives)
- [C. Running the session](#c-running-the-session)
- [D. Troubleshooting](#d-troubleshooting)

---

## A. Before Participant Arrives

### Timing guide

| Time before participant | Action |
|---|---|
| 45 min | Start hardware setup (PC on, eego open, LabRecorder open) |
| 30 min | Prepare the saline solution for the EEG net |
| 20 min | **Begin soaking the EEG net** in saline — no earlier, no later |
| 10 min | Joystick plugged in, Shimmer EDA charging or ready, Polar strap wet |
| 5 min  | Run a quick smoke test: double-click the session launcher, enter the PID, check the phase list appears, then Ctrl-C to abort |

> **Net soaking timing matters.** Soaking for less than 15 minutes gives poor electrode contact.
> Soaking for more than 30 minutes can cause the net to become too wet and drip.

---

### 25 Hz environment check

A 25 Hz electrical artifact (50 Hz mains half-harmonic) has been observed in several recordings.
It sits in the middle of the beta band and directly contaminates the MWL model's most-selected
features. It couples more strongly when electrode impedance is high.

**Do this once before fitting the cap, while the net is soaking:**

1. With the EEG amplifier on and eego streaming, open a fresh LabRecorder recording for ~30 s (no participant needed — just the amplifier idling).
2. Power everything on as it will be during the session: both monitors, the task PC, your laptop if it will be open, chargers plugged in.
3. After the session, run the sweep from the analysis PC to confirm the recording is clean:
   ```
   .\.venv\Scripts\python.exe scripts/_tmp_25hz_sweep_all_xdfs.py
   ```
   A `spike_p90 < 3.0` for that file means the environment is clean.
4. If a spike is found, turn off equipment one item at a time (laptop charger, second monitor, overhead light dimmer) until the spike disappears. Note the culprit and keep it off during the session.

> **The rest-baseline step also checks automatically.** If the 25 Hz spike is present in the
> participant's rest recording, calibrate_participant.py will print a `WARNING: 25 Hz SPIKE`
> banner in the terminal. Do not proceed past that point — re-record rest after fixing the source.

---

### Hardware startup checklist

Work through this list in order before the participant arrives.

- [ ] Ethernet cable plugged into lab PC (required for EEG — without it, signal is very noisy)
- [ ] ANTneuro eego software open; both amplifiers connected and battery level shown
- [ ] LabRecorder.exe open (`C:\LabRecorder\LabRecorder.exe`); RCS indicator shows "Listening"
- [ ] **Confirm both EEG data streams visible in LabRecorder** — click *Update* in LabRecorder's stream list; you must see **2 streams of type `EEG`** (one per amplifier, named `EE225-...-eego_laptop`) before proceeding. If only 1 or 0 appear: restart eego software, check USB cables.
- [ ] **25 Hz environment check** — do this before fitting the cap (see box below)
- [ ] EEG net placed in saline solution (see timing guide above)
- [ ] Joystick plugged into USB port and detected by Windows
- [ ] Polar H10 HR strap electrodes moistened with water
- [ ] Shimmer EDA unit powered on; electrode gel applied to Ag/AgCl electrodes
- [ ] Output data drive available (default `C:\data\adaptive_matb\` — check there is space)
- [ ] Participant folder confirmed in `config/participant_assignments.yaml`
- [ ] Session information sheet and consent form ready (paper copies)
- [ ] QuestionPro survey link ready in browser (see Section B)

---

## B. When Participant Arrives

### Greeting script

When the participant arrives, say:

> "Hi, thanks for coming in today. My name is [YOUR NAME]. This session will take about [DURATION] hours, including some breaks.
>
> Before we start, I'll ask you to read through a brief information sheet, and if you're happy to continue, sign a consent form. You're completely free to withdraw at any time without giving a reason.
>
> After that, I'll fit you with an EEG cap — that's a soft net with small sponge electrodes that sit on your scalp. It's not uncomfortable. While I'm fitting it, I'll explain what you'll be doing in the task."

---

### Informed consent

1. Give participant the information sheet — ask them to read it.
2. Answer any questions.
3. Provide consent form — participant signs.
4. Retain signed consent form in the participant folder (locked cabinet).

---

### Pre-session questionnaire (QuestionPro)

Open the following URL in the browser on the **lab PC** and hand the keyboard/mouse to the participant:

> **PRE-SESSION URL:** `[PRE_SESSION_QUESTIONNAIRE_URL — FILL IN BEFORE LIVE STUDY]`

Wait for participant to complete it. Retrieve keyboard when done.

---

### Head measurement and net size selection

1. Ask the participant to remove any hair accessories (clips, bobbles, headbands).
2. Use a soft tape measure. Place it:
   - Around the widest circumference of the head
   - Approximately 1 cm above the ears
   - Passing across the forehead just above the eyebrows
3. Record circumference to the nearest 0.5 cm.
4. Select net size from the table below:

| Head circumference | Net size |
|---|---|
| < 52 cm | XS |
| 52–54 cm | S |
| 54–56 cm | M |
| 56–58 cm | L |
| > 58 cm | XL |

> **Note:** Verify these ranges against your specific ANTneuro net model before the first participant session. The ranges above are common but the manufacturer's sizing chart takes precedence.

---

### EEG net fitting

1. Remove the net from saline. Shake off excess water gently — it should be damp, not dripping.
2. Locate the vertex electrode (CZ) — the single electrode at the top centre.
3. Hold the net open and lower onto the participant's head, placing CZ at the top of the skull (midpoint front-to-back and left-to-right).
4. Fasten the chin strap if your net has one — snug but not tight.
5. Check that the ear reference electrodes sit just in front of each ear, level with the ear canal.
6. Open eego software → Impedance check.
7. Aim for **< 30 kΩ per electrode**. > 50 kΩ is marginal; > 100 kΩ is a problem.
   - For any electrode > 50 kΩ: gently press down on the electrode and twist slightly.
   - If still high after 2 attempts: apply a small amount of electrode gel directly under that electrode using the blunt applicator syringe.
8. When impedance is acceptable across the scalp, proceed.

While fitting the net, explain the task to the participant:

> "You'll be looking at a screen showing a simulated aircraft monitoring display. There are four different tasks running at once — tracking a moving target with a joystick, watching instrument gauges and responding when they go out of range, monitoring lights that should stay on, and listening to communications.
>
> Some tasks will be harder than others depending on the conditions. You don't need any flying experience — we're measuring how you manage divided attention.
>
> For one of the two main blocks, the difficulty will adjust automatically based on signals from the EEG cap. In the other block it won't change. We won't tell you which is which until afterwards.
>
> There are no right or wrong answers and we're not testing your ability — we're studying the system."

---

## C. Running the Session

### Launch

Double-click the appropriate icon on the lab PC desktop:

| Session type | Icon to click first | Icon to click second |
|---|---|---|
| Main study participant (P001–P030) | **STUDY STAIRCASE** (no sensors needed) | **STUDY SESSION** (after sensors fitted) |
| Pilot run (PPILOT01–PPILOT10) | **PILOT STAIRCASE** (no sensors needed) | **PILOT SESSION** (after sensors fitted) |
| Dry-run / self-test (PDRY01–05, PSELF) | No icon — run directly: `python src/run_full_study_session.py --participant PDRY01` | (same, `--start-phase 3` to skip to full session) |

> **Two-icon workflow:** The staircase icon runs Practice + Staircase (Phases 1–2, ~15 min) **without any sensors**. While it runs, you can be setting up the EEG cap. Once the staircase window closes, fit the remaining sensors, then click the Full Session icon to continue from Phase 3.

> **If an icon is missing:** Navigate to `C:\adaptive_matb_2026\scripts\`, run `setup_desktop_shortcuts.ps1` to recreate all four shortcuts. Or right-click the `.bat` file and choose *Send to → Desktop (create shortcut)*.

When the **Staircase** launcher opens:
1. Type the participant number (e.g. `12` for P012).
2. Confirm the green summary and press Enter to launch.

When the **Full Session** launcher opens:
1. Type the same participant number.
2. Press Enter to accept Phase 3 (default start-after-staircase), or type a higher number if resuming.
3. Review the green summary and press Enter to launch.

---

### Phase-by-phase: what you do at each pause

The script pauses at specific points and waits for you to press Enter.

| Pause | Your action |
|---|---|
| **Session plan displayed** | Confirm the participant ID and condition order (e.g. "adaptation → control") are correct. Press Enter. |
| **Phase 2 (staircase) complete** | Check the console shows `d_final = [a number]` printed in white. This confirms staircase converged. The window will close. |
| **Staircase icon complete** | Finish fitting any remaining sensors (EDA, HR). When all sensors ready, click the **Full Session** icon and enter the same participant number. |
| **Before Phase 3 — rest baseline** | Say to participant: *"A cross will appear on the screen. Please sit quietly, keep your eyes on the cross, and try not to move or blink for 2 minutes."* Wait until they are still and comfortable. Press Enter. |
| **After Phase 4 — scenarios generated** | Console shows three filenames generated. No action needed. Press Enter. |
| **After calibration run 1** | The task window will show a questionnaire on screen — participant fills it in using the mouse. Wait for them to finish. Press Enter. |
| **After calibration run 2** | Same — wait for on-screen questionnaire to complete. Press Enter. |
| **After Phase 6 — model calibration** | Console shows model saved. Press Enter. |
| **Before adaptation condition — baseline refresh** | Say: *"Before the next task, please sit quietly with your eyes on the cross for about a minute."* Wait for participant. Press Enter. |
| **After condition 1** | Wait for on-screen questionnaire. Press Enter. |
| **Session complete** | Proceed to post-session questionnaire below. |

> **About the in-task questionnaire:** Workload rating sliders appear automatically on the task screen at the end of each run. The participant rates each dimension with the mouse. You do not need to do anything — just wait.

---

### Post-session questionnaire (QuestionPro)

When the session completes, open the following URL in the browser and hand keyboard/mouse to participant:

> **POST-SESSION URL:** `[POST_SESSION_QUESTIONNAIRE_URL — FILL IN BEFORE LIVE STUDY]`

After participant finishes:
- Thank them and provide debrief sheet.
- Answer any questions about the study.
- Record session completion in the participant log.

---

## D. Troubleshooting

### Resuming a session that stopped mid-way

Double-click the **Full Session** icon, enter the same participant number, then at the phase prompt type the number of the **last completed phase + 1**.

| If the session stopped after... | Resume at phase |
|---|---|
| Practice (familiarisation scenarios) | 2 |
| Staircase | Use **Full Session** icon (defaults to Phase 3) |
| Rest baseline (black cross screen) | 4 |
| Scenario generation (console messages) | 5 |
| Calibration run 1 or 2 | 6 |
| Model calibration (console output) | 7 |

---

### EEG (ANTneuro eego)

**eego shows "No amplifier connected"**
→ Check the USB cable between amp and PC. Try unplugging and re-plugging.
→ If still not connected, restart eego software.

**Many electrodes showing impedance > 100 kΩ**
→ The net may have dried out. Apply more saline solution using the spray bottle, wait 2 minutes, recheck.
→ Ensure the participant's hair is not blocking contact.

**One or two electrodes persistently high**
→ Apply electrode gel directly to those sites using the blunt syringe applicator.
→ If still > 100 kΩ after gel, note which electrodes and proceed — one or two bad sites are acceptable.

**EEG smoke test fails at session launch ("FAIL — found 1 stream, need 2")**
→ Check both amplifiers are on (battery lights visible) and USB cables are connected.
→ Restart eego software.
→ If still failing after 2 attempts: call developer.

**Session script prints EEG stream check failed before a calibration or condition run**
→ The script checks both EEG streams are live immediately before each task launches. If this check fails mid-session, the streams have dropped since the last task.
→ In eego software, check that both amplifiers show a live signal trace. If not:
  1. Close eego, unplug both USB amplifier cables, wait 5 s, re-plug, reopen eego.
  2. Confirm both amps show live traces in eego.
  3. Press **R** in the session script to retry the stream check — do not press C.
→ Do **not** press C (continue) — a run that launches without live EEG streams will record only task markers, not EEG data, and cannot be recovered.

**EEG data file is very short (< 2 min) despite task completing fully**
→ This means the EEG data streams dropped from LSL while LabRecorder was running. The task markers are recorded but the EEG time series is gone.
→ The root cause is usually: Ethernet adapter entered power-saving mode (cuts the TCP stream carrying 500 Hz × 66ch × 2 amps), or eego lost its USB connection briefly.
→ The affected XDF is **not recoverable** — the calibration or condition run must be repeated.
→ To prevent recurrence: on the EEG laptop, set the Ethernet adapter's Power Management to **"Do not allow the computer to turn off this device to save power"** (Device Manager → Network Adapters → adapter Properties → Power Management tab). Also set the Power Plan to **High Performance**.

---

### LabRecorder

**LabRecorder is not recording (no "Recording" indicator in status bar)**
→ Open `C:\LabRecorder\LabRecorder.exe` if it is closed.
→ Check the RCS status indicator in the LabRecorder window shows "Listening".
→ If LabRecorder crashed mid-session: reopen it, then resume from the correct phase (see resume table above).

**LabRecorder crashed during calibration runs**
→ Reopen LabRecorder, then resume the session at Phase 5 (re-run the calibration block that was interrupted).

---

### Polar H10 HR strap

**No HR data at session start / "HR stream not found"**
→ Press the strap more firmly against the participant's skin.
→ Re-wet the two electrode contacts on the strap with water.
→ Open Windows *Settings → Bluetooth* and confirm the Polar H10 is connected.
→ If disconnected: put the strap in pairing mode (press and hold the button for 3 seconds) and re-pair.

---

### Shimmer EDA

**EDA values out of range / "EDA stream error"**
→ Check that the electrode cables are firmly clipped to the Ag/AgCl electrodes on the participant's fingers.
→ Re-apply electrode gel to the electrode surfaces.
→ If Shimmer light is off: press the power button and allow 10 seconds to reconnect.

---

### Joystick

**"No joystick detected" at session start**
→ Unplug the USB cable and re-plug it. Wait 5 seconds.
→ If still not detected: try a different USB port, then relaunch.

---

### Script and software errors

**"WARNING: Not running inside the project venv"**
→ Close the terminal and relaunch using the desktop icon (the `.bat` file activates the venv automatically).

**"ERROR: Virtual environment not found"**
→ The `.venv` folder is missing. Call developer — `[DEVELOPER_CONTACT — FILL IN BEFORE LIVE STUDY]`.

**"Module not found" or any Python ImportError**
→ Call developer — `[DEVELOPER_CONTACT — FILL IN BEFORE LIVE STUDY]`.

**Session window closes unexpectedly / exits with "failed with exit code 1"**
→ Note the error message printed above the exit line.
→ Resume from the correct phase if data was being collected.
→ If error is unfamiliar, call developer before continuing.

---

### Emergency contact

Developer on call during study hours:

> **Name / Phone / Email:** `[DEVELOPER_CONTACT — FILL IN BEFORE LIVE STUDY]`

---

*Last updated: 2026-04-02*
