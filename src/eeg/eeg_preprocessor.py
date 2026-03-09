import numpy as np
from typing import Optional
from .eeg_preprocessing_config import EegPreprocessingConfig
from .eeg_filters import RealTimeFilter, design_bandpass, design_notch

class EegPreprocessor:
    """
    Orchestrates the preprocessing pipeline: Bandpass -> Notch -> CAR.
    """
    def __init__(self, config: EegPreprocessingConfig):
        self.config = config
        # State
        self.n_channels: Optional[int] = None
        self.bp_filter: Optional[RealTimeFilter] = None
        self.notch_filter: Optional[RealTimeFilter] = None
        
    def initialize_filters(self, n_channels: int):
        """
        Initialize the filter states for N channels.
        Must be called before processing if not initialized in constructor.
        """
        self.n_channels = n_channels
        
        # Design & Init Bandpass
        bp_sos = design_bandpass(
            self.config.bp_low_hz, 
            self.config.bp_high_hz, 
            self.config.srate, 
            self.config.bp_order
        )
        self.bp_filter = RealTimeFilter(bp_sos, n_channels)
        
        # Design & Init Notch
        notch_sos = design_notch(
            self.config.notch_freq, 
            self.config.notch_quality, 
            self.config.srate
        )
        self.notch_filter = RealTimeFilter(notch_sos, n_channels)
        
    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of EEG data (n_channels, n_samples).
        """
        if self.bp_filter is None or self.notch_filter is None:
            raise RuntimeError("Filters not initialized. Call initialize_filters(n_channels) first.")
            
        if chunk.size == 0:
            return chunk
            
        # 1. Bandpass
        data = self.bp_filter.process(chunk)
        
        # 2. Notch
        data = self.notch_filter.process(data)
        
        # 3. CAR - Common Average Reference
        if self.config.apply_car:
            # Metric: Average across channels for each sample
            # chunk shape: (n_channels, n_samples), axis 0 = channels
            common_avg = np.mean(data, axis=0) # shape (n_samples,)
            data = data - common_avg
            
        return data

    def reset(self):
        """Reset filter states."""
        if self.n_channels is not None:
            self.initialize_filters(self.n_channels)
