import numpy as np
import time
from eeg import EegPreprocessingConfig, EegPreprocessor

def verify_dsp_chain():
    """
    Test the EEG preprocessing steps on synthetic data.
    """
    print("Initializing verification...")
    
    # 1. Setup Config
    srate = 500.0
    config = EegPreprocessingConfig(
        bp_low_hz=1.0,
        bp_high_hz=40.0,
        notch_freq=50.0,
        notch_quality=30.0,
        apply_car=True,
        srate=srate
    )
    
    # 2. Setup Preprocessor
    n_channels = 4
    preprocessor = EegPreprocessor(config)
    preprocessor.initialize_filters(n_channels)
    
    # 3. Generate Synthetic Signal
    # 10 seconds of data
    duration = 10.0
    t = np.arange(0, duration, 1/srate)
    n_samples = len(t)
    
    # Signal Components
    clean_sig = np.sin(2 * np.pi * 10.0 * t)             # 10Hz Alpha (Keep)
    mains_hum = 0.5 * np.sin(2 * np.pi * 50.0 * t)       # 50Hz Hum (Notch)
    drift = 2.0 * np.sin(2 * np.pi * 0.1 * t) + 10.0     # 0.1Hz Drift + DC Offset (Highpass should kill DC)
    
    # Combine (for channel 0)
    raw_ch0 = clean_sig + mains_hum + drift
    
    # Make channels correlated (for CAR test)
    # ch1 is same but weaker
    raw_ch1 = raw_ch0 * 0.8
    # ch2 is just noise
    raw_ch2 = mains_hum + drift
    # ch3 is random
    raw_ch3 = np.random.randn(n_samples) * 0.2
    
    raw_data = np.vstack([raw_ch0, raw_ch1, raw_ch2, raw_ch3]) # (4, n_samples)
    
    # 4. Process in chunks (simulate streaming)
    chunk_size = 50  # 100ms chunks
    processed_chunks = []
    
    print(f"Processing {duration}s data in {chunk_size} sample chunks...")
    start_time = time.time()
    for i in range(0, n_samples, chunk_size):
        chunk = raw_data[:, i:i+chunk_size]
        # Pad last chunk if needed or slice safe
        if chunk.shape[1] > 0:
            proc_chunk = preprocessor.process(chunk)
            processed_chunks.append(proc_chunk)
        
    processed_data = np.hstack(processed_chunks)
    elapsed = time.time() - start_time
    
    print(f"Processed {duration}s of {n_channels}ch data in {elapsed:.4f}s")
    
    # 5. Analysis
    raw_mean_drift = np.mean(raw_data[0])
    proc_mean_drift = np.mean(processed_data[0])
    
    print(f"Ch0 Mean (Raw): {raw_mean_drift:.3f}")
    print(f"Ch0 Mean (Proc): {proc_mean_drift:.3f} (Should be closer to 0 due to high-pass)")
    
    if abs(proc_mean_drift) < abs(raw_mean_drift):
        print("PASS: Mean offset reduced (Drift removed).")
    else:
        print("FAIL: Mean offset not reduced.")

    # Check Mains Hum (Notch)
    # Simple FFT check on a small window
    from scipy.fft import fft, fftfreq
    
    def get_power_at_fs(sig, fs, target_freq):
        N = len(sig)
        yf = fft(sig)
        xf = fftfreq(N, 1/fs)
        idx = np.argmin(np.abs(xf - target_freq))
        return np.abs(yf[idx])
        
    raw_50hz = get_power_at_fs(raw_data[0], srate, 50.0)
    proc_50hz = get_power_at_fs(processed_data[0], srate, 50.0)
    
    print(f"50Hz Power (Raw): {raw_50hz:.1f}")
    print(f"50Hz Power (Proc): {proc_50hz:.1f}")
    
    if proc_50hz < raw_50hz * 0.1:
         print("PASS: 50Hz hum suppressed.")
    else:
         print("FAIL: 50Hz hum not suppressed significantly.")

    # Check Signal Preservation (10Hz)
    raw_10hz = get_power_at_fs(raw_data[0], srate, 10.0)
    proc_10hz = get_power_at_fs(processed_data[0], srate, 10.0)
    
    print(f"10Hz Power (Raw): {raw_10hz:.1f}")
    print(f"10Hz Power (Proc): {proc_10hz:.1f}")
    
    # CAR Effect Calculation:
    # Ch0=1.0, Ch1=0.8, Ch2=0, Ch3=0. Avg = 0.45.
    # Result Ch0 = 1.0 - 0.45 = 0.55.
    # We expect roughly 55% of original amplitude.
    if proc_10hz > raw_10hz * 0.5: 
        print("PASS: 10Hz signal preserved (accounting for CAR reduction).")
    else:
        print("FAIL: 10Hz signal attenuated too much.")
        
if __name__ == "__main__":
    verify_dsp_chain()
