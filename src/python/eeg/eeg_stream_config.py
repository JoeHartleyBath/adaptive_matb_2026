from dataclasses import dataclass, field
from typing import List, Literal, Optional

@dataclass
class EegStreamConfig:
    """
    Configuration for the EEG LSL Inlet.
    
    Attributes:
        stream_name: The name of the LSL stream to look for (e.g., "eego").
                     If None, looks for any stream with stream_type="EEG".
        stream_type: The type of the LSL stream (default: "EEG").
        expected_srate: The expected sampling rate in Hz (e.g., 500.0).
        expected_channels: List of channel labels to verify against.
                           If empty, verification is skipped.
        mains_freq: Mains frequency for notch filtering (50 or 60 Hz).
        buffer_duration_s: Duration of the ring buffer in seconds (default: 10.0).
        source_id: Optional LSL source_id to filter by unique device ID.
    """
    stream_name: Optional[str] = "eego"
    stream_type: str = "EEG"
    expected_srate: float = 500.0
    expected_channels: List[str] = field(default_factory=list)
    mains_freq: Literal[50, 60] = 50
    buffer_duration_s: float = 10.0
    source_id: Optional[str] = None
