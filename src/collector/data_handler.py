import os
import threading
import time
from collections import deque
from datetime import datetime

import mne
import numpy as np
from pylsl import StreamInlet, resolve_streams


class LSLClient:
    def __init__(self):
        self.inlet = None
        self.info = None
        self.lsl_offset = None
        self.running = False
        self._thread = None
        self._data_buffer = deque()
        self._timestamp_buffer = deque()

    def find_streams(self):
        return resolve_streams(wait_time=1.0)

    def connect(self, stream_info):
        self.inlet = StreamInlet(stream_info)
        self.info = self.inlet.info()
        self.lsl_offset = self.inlet.time_correction()

    def start_recording(self):
        if self.inlet is None:
            raise RuntimeError("Stream not connected")
        self.running = True
        self._data_buffer.clear()
        self._timestamp_buffer.clear()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop_recording(self):
        self.running = False
        if self._thread:
            self._thread.join()

    def _record_loop(self):
        while self.running:
            chunk, timestamps = self.inlet.pull_chunk(timeout=1.0)
            if timestamps:
                self._data_buffer.extend(chunk)
                self._timestamp_buffer.extend(timestamps)
            else:
                time.sleep(0.001)

    def get_data(self):
        data = list(self._data_buffer)
        timestamps = list(self._timestamp_buffer)
        self._data_buffer.clear()
        self._timestamp_buffer.clear()
        return np.array(data) if data else np.empty((0, 0)), np.array(timestamps)


class DataLogger:
    def __init__(self, save_dir="data_new"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.raw_data = []
        self.timestamps = []
        self.events = []
        self.info = None

    def set_stream_info(self, lsl_info):
        n_channels = lsl_info.channel_count()
        sfreq = lsl_info.nominal_srate()
        ch_names = []

        ch = lsl_info.desc().child("channels").child("channel")
        for _ in range(n_channels):
            name = ch.child_value("label")
            ch_names.append(name if name else f"EEG_{len(ch_names):03d}")
            ch = ch.next_sibling()

        self.info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    def add_data(self, data, timestamps):
        if len(data) > 0:
            self.raw_data.append(data)
            self.timestamps.append(timestamps)

    def add_event(self, timestamp, marker):
        self.events.append((timestamp, marker))

    def remove_last_event(self):
        if self.events:
            self.events.pop()

    def clear(self):
        self.raw_data.clear()
        self.timestamps.clear()
        self.events.clear()

    def save(self, subject_id, run_id):
        if not self.raw_data:
            return None

        full_data = np.concatenate(self.raw_data, axis=0).T
        full_times = np.concatenate(self.timestamps)

        raw = mne.io.RawArray(full_data, self.info, verbose=False)

        start_time = full_times[0]
        mne_events = []
        for ts, marker in self.events:
            rel_time = ts - start_time
            if rel_time < 0:
                continue
            sample_idx = int(rel_time * self.info["sfreq"])
            if sample_idx < full_data.shape[1]:
                mne_events.append([sample_idx, 0, marker])

        if mne_events:
            mne_events = np.array(mne_events)
            onset = mne_events[:, 0] / self.info["sfreq"]
            duration = np.zeros_like(onset)
            description = [str(m) for m in mne_events[:, 2]]
            annotations = mne.Annotations(onset=onset, duration=duration, description=description)
            raw.set_annotations(annotations)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f"{subject_id}_run{run_id}_{timestamp_str}_raw.fif")
        raw.save(filename, overwrite=True, verbose=False)
        return filename
