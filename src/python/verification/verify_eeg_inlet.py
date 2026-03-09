"""
Verification script for the EEG Inlet.
Run this script to verifying that the EegInlet can connect to a (mock) stream
and buffer data correctly.

Usage:
  1. Open two terminals.
  2. Terminal 1: python src/python/scripts/mock_eeg_stream.py 

To run self-contained:
  python src/python/verify_eeg_inlet.py --mock
"""

import sys
import time
import argparse
import pylsl
import numpy as np
from threading import Thread

# Fix import path to find the src/ modules
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eeg.eeg_inlet import EegInlet
from eeg.eeg_stream_config import EegStreamConfig

def run_mock_outlet(name="eego", type="EEG", srate=500, n_ch=66):
    """Publishes random data to LSL."""
    print(f"Starting Mock LSL Outlet: {name} ({n_ch}ch @ {srate}Hz)...")
    info = pylsl.StreamInfo(name, type, n_ch, srate, 'float32', 'mock_source_id')
    outlet = pylsl.StreamOutlet(info)
    
    while True:
        # Push chunks of 10 samples
        chunk = np.random.randn(10, n_ch).tolist()
        outlet.push_chunk(chunk)
        time.sleep(10 / srate)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Start a background mock stream")
    args = parser.parse_args()

    if args.mock:
        t = Thread(target=run_mock_outlet, daemon=True)
        t.start()
        time.sleep(1) # Let outlet warm up

    # 1. Configuration
    config = EegStreamConfig(
        stream_name="eego",
        stream_type="EEG",
        expected_srate=500.0,
        expected_channels=[], # Skip channel name check for now
        buffer_duration_s=5.0
    )
    
    # 2. Initialization
    inlet = EegInlet(config)
    
    # 3. Connection
    print("Connecting...")
    if not inlet.connect(timeout=3.0):
        print("Failed to connect! (Did you start an outlet?)")
        sys.exit(1)
        
    # 4. Data Loop
    print("Starting data ingestion loop for 3 seconds...")
    start_t = time.time()
    chunks_pulled = 0
    
    while time.time() - start_t < 3.0:
        chunk, ts = inlet.pull_chunk()
        if chunk.size > 0:
            chunks_pulled += 1
            # Print status every ~1 sec (assuming 500hz, ~50 chunks/sec with default pull size?) 
            # Actually pull_chunk is fast.
            pass
        time.sleep(0.01)

    print(f"Finished. Pulled {chunks_pulled} chunks.")
    print(f"Total samples processed: {inlet.samples_processed}")
    
    # 5. Verify Window Retrieval
    print("Verifying window retrieval...")
    try:
        data, ts = inlet.get_window(duration_s=1.0)
        print(f"Retrieved 1.0s window: Shape {data.shape}")
        
        expected_samples = int(1.0 * 500.0)
        # Check shape (n_ch, n_samples)
        if data.shape[1] == expected_samples:
            print("SUCCESS: Window shape matches expected duration.")
        else:
            print(f"FAILURE: Expected {expected_samples} samples, got {data.shape[1]}")
            
        if data.shape[0] != inlet.n_channels:
             print(f"FAILURE: Expected {inlet.n_channels} channels, got {data.shape[0]}")
             
    except Exception as e:
        print(f"FAILURE: Exception during get_window: {e}")

    inlet.close()

if __name__ == "__main__":
    main()
