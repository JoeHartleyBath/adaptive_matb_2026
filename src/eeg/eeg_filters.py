import numpy as np
import scipy.signal

class RealTimeFilter:
    """
    A stateful wrapper for scipy.signal filters (SOS) to support streaming.
    """
    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = sos
        # Initialize state: (n_sections, n_channels, 2)
        # scipy.signal.sosfilt_zi() generates state for 1 channel. We broadcast.
        zi_1ch = scipy.signal.sosfilt_zi(self.sos)
        # Repeat for each channel: resulting shape (n_sections, n_channels, 2)
        self.zi = np.repeat(zi_1ch[:, np.newaxis, :], n_channels, axis=1)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Apply filter to chunk (n_channels, n_samples). Update state.
        """
        if chunk.shape[1] == 0:
            return chunk
            
        # sosfilt applies along last axis by default, which is samples
        # internal zi shape must match input channels
        filtered, self.zi = scipy.signal.sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        return filtered

def design_bandpass(low_hz: float, high_hz: float, srate: float, order: int = 4) -> np.ndarray:
    """Design a Butterworth bandpass filter (SOS format)."""
    sos = scipy.signal.butter(order, [low_hz, high_hz], btype='band', fs=srate, output='sos')
    return sos

def design_notch(freq: float, quality: float, srate: float) -> np.ndarray:
    """Design a notch filter (SOS format)."""
    b, a = scipy.signal.iirnotch(freq, quality, fs=srate)
    sos = scipy.signal.tf2sos(b, a)
    return sos
