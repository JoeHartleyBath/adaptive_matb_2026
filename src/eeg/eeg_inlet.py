import numpy as np
import pylsl
import time
from typing import Tuple, Optional
from collections import deque
from .eeg_stream_config import EegStreamConfig
from .eeg_preprocessor import EegPreprocessor

class EegInlet:
    """
    Manages the LSL connection to an EEG stream and provides a ring buffer of data.
    
    If a preprocessor is provided, data in the buffer is PREPROCESSED (filtered/referenced).
    Otherwise, it is RAW.
    """

    def __init__(self, config: EegStreamConfig, preprocessor: Optional[EegPreprocessor] = None):
        self.config = config
        self.preprocessor = preprocessor
        
        self.inlet: Optional[pylsl.StreamInlet] = None
        self.stream_info: Optional[pylsl.StreamInfo] = None
        self.buffer: Optional[np.ndarray] = None
        self.timestamps: Optional[np.ndarray] = None
        self.n_channels = 0
        self.buffer_samples = 0
        self.write_pointer = 0
        
        # State tracking
        self.is_connected = False
        self.samples_processed = 0
        self.start_time = 0.0

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Attempt to resolve and connect to the LSL stream.
        
        Args:
            timeout: Max time in seconds to wait for resolution.
            
        Returns:
            True if connected successfully, False otherwise.
        """
        print(f"Looking for LSL stream (Name: {self.config.stream_name}, Type: {self.config.stream_type})...")
        
        # Build predicate for resolution
        pred = f"type='{self.config.stream_type}'"
        if self.config.stream_name:
            pred += f" and name='{self.config.stream_name}'"
        if self.config.source_id:
            pred += f" and source_id='{self.config.source_id}'"
            
        streams = pylsl.resolve_stream('type', self.config.stream_type)
        
        # Filter manually if predicate logic is complex or strictly name-based matching is preferred
        target_stream = None
        for stream in streams:
            if self.config.stream_name and stream.name() != self.config.stream_name:
                continue
            # Found a match
            target_stream = stream
            break
            
        if not target_stream:
            print("Stream not found.")
            return False
            
        self.inlet = pylsl.StreamInlet(target_stream)
        self.stream_info = self.inlet.info()
        
        # Verify srate
        srate = self.stream_info.nominal_srate()
        if srate != self.config.expected_srate:
             print(f"Warning: Stream srate ({srate}) does not match config ({self.config.expected_srate}).")
        
        # Verify channels
        self.n_channels = self.stream_info.channel_count()
        expected = self.config.expected_channels
        if expected and self.n_channels != len(expected):
            print(f"Warning: Channel count ({self.n_channels}) != config ({len(expected)}).")
                
        # Initialize Preprocessor if present
        if self.preprocessor:
            print(f"Initializing preprocessor for {self.n_channels} channels.")
            self.preprocessor.initialize_filters(self.n_channels)

        # Initialize Ring Buffer
        # Shape: (n_channels, n_samples)
        # We process column-major (channels x time) for vectorization efficiency later
        self.buffer_samples = int(self.config.buffer_duration_s * self.config.expected_srate)
        self.buffer = np.zeros((self.n_channels, self.buffer_samples), dtype=np.float32)
        self.timestamps = np.zeros(self.buffer_samples, dtype=np.float64)
        
        self.is_connected = True
        self.start_time = time.time()
        print(f"Connected to {self.stream_info.name()} ({self.n_channels} ch @ {srate} Hz)")
        return True

    def pull_chunk(self, max_samples: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pull available data from LSL, update signal buffer, and return the new chunk.
        
        Returns:
            chunk (np.ndarray): New data chunk (n_channels, n_samples).
            timestamps (np.ndarray): LSL timestamps for the chunk.
        """
        if not self.inlet:
            return np.array([]), np.array([])
            
        # Pull chunk (data is samples x channels from LSL)
        chunk, ts = self.inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
        
        if not ts:
            return np.array([]), np.array([])
            
        # Convert to numpy
        # chunk is list of lists [[ch1, ch2...], [ch1, ch2...]] -> shape (n_samples, n_channels)
        chunk_arr = np.array(chunk, dtype=np.float32).T # Transpose to (n_channels, n_samples)
        ts_arr = np.array(ts, dtype=np.float64)
        
        # Preprocessing Step (inplace or new array)
        if self.preprocessor:
            chunk_arr = self.preprocessor.process(chunk_arr)
        
        n_new = len(ts)
        
        # Update Ring Buffer
        # Case 1: New data fits without wrapping
        if self.write_pointer + n_new <= self.buffer_samples:
            self.buffer[:, self.write_pointer:self.write_pointer + n_new] = chunk_arr
            self.timestamps[self.write_pointer:self.write_pointer + n_new] = ts_arr
            self.write_pointer += n_new
            if self.write_pointer >= self.buffer_samples:
                self.write_pointer = 0
        else:
            # Case 2: Wrap around
            space_end = self.buffer_samples - self.write_pointer
            # Write end part
            self.buffer[:, self.write_pointer:] = chunk_arr[:, :space_end]
            self.timestamps[self.write_pointer:] = ts_arr[:space_end]
            # Write start part
            remaining = n_new - space_end
            self.buffer[:, :remaining] = chunk_arr[:, space_end:]
            self.timestamps[:remaining] = ts_arr[space_end:]
            self.write_pointer = remaining
            
        self.samples_processed += n_new
        return chunk_arr, ts_arr

    def get_window(self, duration_s: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the most recent N seconds of data from the ring buffer.
        
        Args:
            duration_s: Length of window in seconds.
            
        Returns:
            data (np.ndarray): (n_channels, n_samples)
            timestamps (np.ndarray): (n_samples,)
        """
        n_samples = int(duration_s * self.config.expected_srate)
        if n_samples > self.buffer_samples:
            raise ValueError("Requested window larger than buffer size")
            
        # Reconstruct linear buffer from ring buffer based on write_pointer
        # The newest data is at write_pointer - 1
        
        indices = np.arange(self.write_pointer - n_samples, self.write_pointer)
        # Allow negative indices to wrap automatically via numpy's modulo behavior
        # But explicit modulo is safer for robustness
        indices = indices % self.buffer_samples
        
        data_window = self.buffer[:, indices]
        ts_window = self.timestamps[indices]
        
        return data_window, ts_window

    def close(self):
        if self.inlet:
            self.inlet.close_stream()
        self.is_connected = False
