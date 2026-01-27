from dataclasses import dataclass

@dataclass
class EegPreprocessingConfig:
    """
    Configuration for EEG preprocessing steps.
    
    Attributes:
        bp_low_hz: High-pass corner frequency (default: 0.5 Hz).
        bp_high_hz: Low-pass corner frequency (default: 40.0 Hz).
        bp_order: Order of the Butterworth bandpass filter (default: 4).
        notch_freq: Center frequency for notch filter (default: 50.0 Hz).
        notch_quality: Quality factor (Q) for notch filter (default: 30.0).
        apply_car: Whether to apply Common Average Reference (default: True).
        srate: Expected sampling rate in Hz (default: 500.0).
    """
    bp_low_hz: float = 0.5
    bp_high_hz: float = 40.0
    bp_order: int = 4
    notch_freq: float = 50.0
    notch_quality: float = 30.0
    apply_car: bool = True
    srate: float = 500.0
