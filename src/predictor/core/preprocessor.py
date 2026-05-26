import numpy as np
import scipy.signal as signal

class EEGPreprocessor:
    def __init__(self, target_srate=256.0):
        self.target_srate = target_srate
        self.lowcut = 8.0
        self.highcut = 32.0
        self.notch_freq = 50.0
        self.notch_quality = 30.0
        
    def process(self, data: np.ndarray, input_srate: float) -> np.ndarray:
        """
        Applies standard EEG preprocessing:
        1. Channel Selection (1:17)
        2. Common Average Reference (CAR)
        3. Resampling (to target_srate)
        4. Notch Filter (50Hz)
        5. Bandpass Filter (8-32Hz)
        
        Args:
            data: Raw LSL data (n_channels, n_samples).
                  Expects channels 1-17 to be EEG.
            input_srate: Sampling rate of the input data.
            
        Returns:
            np.ndarray: Processed data (16, new_samples)
        """
        
        # 1. Channel Selection
        # We assume the standard layout where 0 is trigger/timestamp and 1..16 are EEG
        # If data has enough channels, we slice.
        if data.shape[0] > 16:
            data = data[1:17, :]
        elif data.shape[0] == 16:
            print("Warning: Input data has exactly 16 channels, but expected more.")
        else:
            raise Exception("Not enough channels in the input data. Expected at least 16 EEG channels.")

        # 2. Common Average Reference
        data = data - data.mean(axis=0, keepdims=True)

        # 3. Resampling
        n_samples = data.shape[1]
        if n_samples == 0:
            return np.empty((data.shape[0], 0))
            
        if input_srate != self.target_srate:
            duration = n_samples / input_srate
            target_samples = int(duration * self.target_srate)
            # signal.resample uses FFT, assumed good for chunks
            data = signal.resample(data, target_samples, axis=1)
        
        # 4. Notch Filter
        # Note: filtfilt avoids phase shift but requires data length > padlen
        # We need to be careful with very short windows.
        b_notch, a_notch = signal.iirnotch(self.notch_freq, self.notch_quality, self.target_srate)
        data = signal.filtfilt(b_notch, a_notch, data, axis=-1)
        
        # 5. Bandpass Filter
        nyquist = 0.5 * self.target_srate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b_band, a_band = signal.butter(5, [low, high], btype='band')
        data = signal.filtfilt(b_band, a_band, data, axis=-1)
        
        return data
