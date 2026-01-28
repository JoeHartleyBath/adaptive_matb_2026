# Lab Note: 2026-01-24 - Pilot Branch Setup

## Research Thread
**Goal:** Prepare a stable, reproducible branch for the pilot data collection starting Jan 27th.

## Key Changes
1.  **Branch Creation:**
    *   Created `pilot/v0` branch from `main` (commit `d079f65`).
    *   **Rule:** This branch is identifyingly "locked" for the pilot. Only critical fixes should be cherry-picked.

2.  **Scenario Setup:**
    *   **Familiarisation:** Reverted `pilot_familiarisation.txt` to use the exact event logic and timing of the vendor's `default.txt` to ensure stability (no "Show All" bugs).
    *   **Instructions:** updated `src/python/vendor/openmatb/includes/instructions/pilot_en/*.txt`:
        *   Added `<center>` tags to all HTML files to match the visual style of the French defaults (which relied on a rendering bug).
        *   Rewrote `2_sysmon.txt` to clarify F1-F6 keys (F1-F4 for scales, F5-F6 for lights).
    *   **Training/calibration:** Generated using `generate_pilot_scenarios.py` with validated difficulty levels (Low: 0.2, Mod: 0.55, High: 0.95).

3.  **Verification:**
    *   Manual Run: Validated `pilot_familiarisation.txt` runs in interactive mode using `main.py` directly.

## Pending Actions
*   **Audio Translation:** The familiarization scenario still uses the default French audio prompts (e.g., "Changez la fréquence..."). These need to be recorded in English and placed in `sound/` or integrated via the TTS plugin.
*   **Participant Checklist:** Needs final update to include the specific instruction for "Sound Check" before starting the familiarization block.
*   **Scenario Verification:** Manually test run the generated Training and calibration scenarios to confirm difficulty levels and timing.
*   **Full Pipeline Test:** Execute full pilot runs (Training sequence + calibration blocks) in interactive mode.
*   **NASA-TLX Verification:** Confirm the NASA-TLX questionnaire triggers correctly after calibration blocks and saves data properly.


## Technical Details
*   **Environment Mocking:** To run `main.py` directly from the vendor folder for quick tests, use:
    ```powershell
    $env:OPENMATB_REPO_COMMIT="mock_commit"
    $env:OPENMATB_SUBMODULE_COMMIT="mock_submodule"
    python main.py
    ```
