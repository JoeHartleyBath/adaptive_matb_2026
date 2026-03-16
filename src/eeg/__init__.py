try:
    from .eeg_inlet import EegInlet
    from .eeg_stream_config import EegStreamConfig
except ImportError:  # pylsl not installed — online-only, not needed for analysis
    EegInlet = None  # type: ignore[assignment,misc]
    EegStreamConfig = None  # type: ignore[assignment,misc]

from .eeg_preprocessing_config import EegPreprocessingConfig
from .eeg_preprocessor import EegPreprocessor
from .eeg_filters import RealTimeFilter
from .eeg_windower import WindowConfig, extract_windows, slice_block
from .extract_features import extract_features, load_all_features, FIXED_BANDS

__all__ = [
    "EegInlet",
    "EegStreamConfig",
    "EegPreprocessingConfig",
    "EegPreprocessor",
    "RealTimeFilter",
    "WindowConfig",
    "extract_windows",
    "slice_block",
    "extract_features",
    "load_all_features",
    "FIXED_BANDS",
]
