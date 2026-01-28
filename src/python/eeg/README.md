# Real-Time EEG Pipeline

This directory contains the Python modules for connecting to, buffering, and cleaning real-time EEG data via Lab Streaming Layer (LSL).

## Architecture

The pipeline is designed as a **Sidecar** process that runs alongside the main application. It consumes raw EEG from LSL, cleans it in real-time without future look-ahead (causal processing), and makes it available for inference.

### Core Modules

| File | Purpose |
|Data Flow| |
| `eeg_inlet.py` | **Connection & Buffering.** Connects to an LSL stream, maintains a Ring Buffer of the last $N$ seconds, and handles chunk pulling. |
| `eeg_preprocessor.py` | **Orchestrator.** Takes a raw chunk of data and passes it through the filter chain (Bandpass -> Notch -> CAR). |
| `eeg_filters.py` | **DSP Logic.** Wraps `scipy.signal` to implement **Stateful Filters**. It maintains the filter state (`zi`) between chunks to ensure signal continuity. |
| **Config** | |
| `eeg_preprocessing_config.py` | Defines signal parameters (e.g., 0.5-40Hz Bandpass, 50Hz Notch). |
| `eeg_stream_config.py` | Defines LSL connection parameters (stream name, type). |

---

## The Signal Processing Chain

The pipeline implements a standard EEG cleaning sequence. Because this is **Real-Time**, we must use **Causal Filters** (processing current and past samples only), unlike offline analysis which can use Acausal filters (looking ahead in time) to correct phase delay.

### 1. Bandpass Filter (0.5 - 40 Hz)
*   **Purpose:** Removes DC drift (<0.5Hz) and high-frequency muscle noise/line jitter (>40Hz).
*   **Implementation:** 4th-order Butterworth filter.
*   **Verification:**
    *   *Input:* DC Offset (Signal + 100uV). *Output:* Mean should be ~0.
    *   *Input:* High freq noise (100Hz). *Output:* Attenuated significantly.

### 2. Notch Filter (50 Hz)
*   **Purpose:** Removes mains electricity hum (European standard 50Hz).
*   **Implementation:** IIR Notch Filter with Q=30.
*   **Verification:**
    *   *Input:* Pure 50Hz sine wave. *Output:* Near-zero amplitude.
    *   *Input:* 10Hz signal. *Output:* Unchanged.

### 3. Common Average Reference (CAR)
*   **Purpose:** Removes noise common to all channels (e.g., movement artifacts, remote reference noise).
*   **Equation:** $V_{clean}(c, t) = V_{raw}(c, t) - \frac{1}{N} \sum_{i=1}^{N} V_{raw}(i, t)$
*   **Verification:**
    *   *Input:* Identical signal on all channels. *Output:* Perfect silence (0).

---

## How to Verify Implementation

We provide a synthetic verification script that generates known signals and tests the pipeline's output.

### Run the Verification Script
```bash
python src/python/verify_preprocessing.py
```
**Expected Output:**
*   **Drift Removal:** PASS (Mean offset reduced).
*   **Mains Hum:** PASS (50Hz power suppressed).
*   **Signal Preservation:** PASS (10Hz Alpha calibration).

### Manual Research / Deep Dive
If you want to verify the Digital Signal Processing (DSP) theory:

1.  **Stateful Filtering (`sosfilt` + `zi`)**:
    *   *Concept:* Applying a filter to chunks independently causes "transient artifacts" (huge spikes) at the start of each chunk because the filter assumes the signal started at 0.
    *   *Solution:* We save the filter's internal state (`zi`) after chunk $N$ and pass it as the initial state for chunk $N+1$.
    *   *Docs:* Read Scipy documentation for [`scipy.signal.sosfilt_zi`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt_zi.html).

2.  **Causal vs. Zero-Phase**:
    *   *Offline (EEGLAB/Matlab):* Uses `filtfilt` (forward-backward filtering) which has **Zero Phase Delay** but requires the whole file.
    *   *Real-Time:* Uses `sosfilt` (forward only). This introduces a small **Phase Delay**.
    *   *Research:* Verify that the time shifted by the filter (Group Delay) is acceptable for your BCI application (usually <100ms for these frequencies).

3.  **LSL Timing**:
    *   LSL provides timestamps synchronized to a shared clock. Our inlet captures these but does not retimestamp data. We assume low latency delivery.

## Usage Example

```python
from eeg import EegInlet, EegStreamConfig, EegPreprocessingConfig, EegPreprocessor

# 1. Config
stream_cfg = EegStreamConfig(stream_name="BioSemi", stream_type="EEG")
prep_cfg = EegPreprocessingConfig(notch_freq=50.0)

# 2. Setup
preprocessor = EegPreprocessor(prep_cfg)
inlet = EegInlet(stream_cfg, preprocessor=preprocessor)

# 3. Running Loop
inlet.connect()
while True:
    # Pulls data AND automatically applies filters
    data, timestamps = inlet.pull_chunk()
    if data.size > 0:
         # 'data' is now clean and ready for classification
         run_classifier(data)
```
