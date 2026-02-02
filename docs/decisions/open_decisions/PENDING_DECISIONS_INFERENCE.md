# Decisions Required for Inference Engine Implementation

We have implemented the **Preprocessing Layer** (LSL -> Filter -> CAR). 
Before building the **Inference Layer** (Features -> Classifier -> Output), we need to resolve the following `TBD`s from the `mwl_eeg_input_contract.md`.

## 1. Sampling Rate & Resampling
**Decision:** What is the `TBD_EFFECTIVE_FS_HZ`?
- **Proposal:** `250 Hz`.
- **Implication:** The LSL stream arrives at 500 Hz. We need to implement a **Downsampler** in the `EegPreprocessor`.
- **Reasoning:** 250 Hz is sufficient for up to 125 Hz signals (Nyquist), covering our 40 Hz bandpass with ample margin, while reducing feature extraction compute by 50%.

## 2. Channel Geometry
**Decision:** What is `TBD_CHANNELS_ORDERED_V0`?
- **Context:** We connect to an "eego" amplifier.
- **Proposal:** Use the standard 64-channel 10-20 layout provided by the eego sport.
- **Action:** We need to capture a sample xdf/stream info to list the exact channel labels (e.g., `Fp1`, `Fz` vs `FP1`, `FZ`).
- **Input Needed:** Please provide a list of channels or a sample file from the actual hardware if available. Otherwise, we will default to a generic 64-ch list.

## 3. Feature Definition guarantees
**Decision:** Define the bands for `EEGLogBandpowerFeatureVectorV0`.
- **Proposal:**
    - Theta: 4 - 8 Hz
    - Alpha: 8 - 12 Hz
    - Beta: 12 - 30 Hz
- **Method:** Welch's method with a 1-second Hanning window, 50% overlap.

## 4. Normalization Strategy (Cold Start Problem)
**Decision:** How do we compute Normalization Statistics (`μ`, `σ`) for real-time Z-scoring?
- **Problem:** Real-time inference starts at $t=0$, but robust stats require data.
- **Proposal:** 
    1.  **Calibration Block:** The protocol MUST include a dedicated block (e.g., "Relax with eyes open for 2 mins") *before* the adaptive task.
    2.  **Implementation:** The system computes `μ` and `σ` from this block, freezes them, and applies them to the subsequent task blocks.
- **Action:** We need to update `run_openmatb.py` to support a "Calibration Mode" or ensure the playlist includes a calibration scenario (e.g., `pilot_calibration.txt`).

## 5. Artifact Rejection Policy
**Decision:** Logic for `TBD_ARTIFACT_POLICY_V0`.
- **Proposal:** Simple Amplitude Threshold.
- **Rule:** If any channel in the window exceeds $\pm 100 \mu V$ (post-cleaning), mark window as `INVALID`.
- **Behavior:** The Classifier skips `INVALID` windows (holds previous output or outputs "Unknown").

---
**Next Implementation Steps:**
1. Update `EegPreprocessor` to include **Resampling** (500 -> 250 Hz).
2. Implement `src/python/features/bandpower.py` (Welch).
3. Implement `src/python/inference/normalizer.py` (Load/Save Calibration Stats).
