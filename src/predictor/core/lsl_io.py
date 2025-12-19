import time
import threading
import numpy as np
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams, proc_clocksync
from collections import deque
from ...common.constants import LSLConfig

class DataHandler:
    def __init__(self, buffer_duration=60.0):
        self.buffer_duration = buffer_duration
        self.srate = 0
        self.n_channels = 0
        self.running = False
        self.inlet = None
        self.thread = None
        
        self._data_chunks = deque() 
        self._timestamps = deque()
        self._total_samples = 0
        
    def find_streams(self):
        return resolve_streams()
        
    def connect(self, stream_info):
        self.inlet = StreamInlet(stream_info, processing_flags=proc_clocksync)
        self.srate = self.inlet.info().nominal_srate()
        self.n_channels = self.inlet.info().channel_count()
        print(f"Connected to {self.inlet.info().name()} ({self.n_channels} ch @ {self.srate} Hz)")
        
        # Reset buffer
        self._data_chunks.clear()
        self._timestamps.clear()
        self._total_samples = 0
        
    def start(self):
        if not self.inlet:
            raise RuntimeError("Not connected to a stream")
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _poll_loop(self):
        while self.running:
            chunk, ts = self.inlet.pull_chunk(timeout=1.0)
            if ts:
                chunk = np.array(chunk) # (n_samples, n_channels)
                # We usually want (n_channels, n_samples) for processing
                chunk = chunk.T 
                
                self._data_chunks.append(chunk)
                self._timestamps.extend(ts)
                self._total_samples += chunk.shape[1]
                
                # Prune buffer if too long (rough estimate)
                # Keep max buffer_duration
                max_samples = int(self.buffer_duration * self.srate)
                while self._total_samples > max_samples * 1.2: # 20% hysteresis
                    removed = self._data_chunks.popleft()
                    r_len = removed.shape[1]
                    self._total_samples -= r_len
                    # remove corresp timestamps
                    for _ in range(r_len):
                        self._timestamps.popleft()
            else:
                time.sleep(0.01)
                
    def get_latest_window(self, duration_sec: float):
        """
        Returns (data, timestamps) for the last `duration_sec` seconds.
        data shape: (n_channels, n_samples)
        """
        if self.srate == 0 or self._total_samples == 0:
            return None, None
            
        required_samples = int(duration_sec * self.srate)
        if self._total_samples < required_samples:
            print("Not enough data yet, returning None")
            return None, None
        
        # Flatten buffer
        # In a high-perf scenario, we might optimize this. 
        # For ~500Hz EEG, copying 10s is trivial.
        
        full_data = np.concatenate(self._data_chunks, axis=1) # (ch, all_samples)
        
        if full_data.shape[1] > required_samples:
            full_ts = np.array(self._timestamps)
            return full_data[:, -required_samples:], full_ts[-required_samples:]
        else:
            return full_data, np.array(self._timestamps)

class PredictionBroadcaster:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stream_name = f"{LSLConfig.STREAM_PREFIX}_{model_name}"
        
        self.info = StreamInfo(
            name=self.stream_name,
            type=LSLConfig.CONTENT_TYPE,
            channel_count=LSLConfig.CHANNEL_COUNT,
            nominal_srate=LSLConfig.NOMINAL_SRATE,
            channel_format='float32',
            source_id=f"pred_{model_name}"
        )
        
        self.outlet = StreamOutlet(self.info)
        print(f"Created Outlet: {self.stream_name}")
        
    def push_prediction(self, probabilities: np.ndarray):
        """
        Push a sample. 
        probabilities: (5,) float array
        """
        self.outlet.push_sample(probabilities)
