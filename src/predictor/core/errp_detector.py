import threading
import time
import numpy as np
import scipy.signal as signal
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams, local_clock, proc_clocksync
from ...common.constants import ErrPConfig


class ErrPPreprocessor:
    def __init__(self, target_srate: float = 256.0, tmin: float = -0.2, tmax: float = 0.8):
        self.target_srate = target_srate
        self.expected_samples = int(round((tmax - tmin) * target_srate)) + 1

    def process(self, data: np.ndarray, input_srate: float) -> np.ndarray:
        if data.shape[0] > 16:
            data = data[1:17, :]

        if data.shape[1] == 0:
            return np.empty((data.shape[0], 0))

        data = data - data.mean(axis=0, keepdims=True)

        data = signal.resample(data, self.expected_samples, axis=1)

        nyq = 0.5 * self.target_srate
        b, a = signal.butter(5, [1.0 / nyq, 10.0 / nyq], btype='band')
        data = signal.filtfilt(b, a, data, axis=-1)

        return data


FEEDBACK_MARKER_CORRECT = 20
FEEDBACK_MARKER_ERROR = 21
FEEDBACK_MARKERS = {FEEDBACK_MARKER_CORRECT, FEEDBACK_MARKER_ERROR}


class ErrPDetector:
    def __init__(self, classifier, window_start: float = -0.2, window_end: float = 0.8):
        self.classifier = classifier
        self.window_start = window_start
        self.window_end = window_end
        self.preprocessor = ErrPPreprocessor()

        self._eeg_inlet: StreamInlet | None = None
        self._marker_inlet: StreamInlet | None = None
        self._outlet: StreamOutlet | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        self._eeg_buffer = []
        self._eeg_ts_buffer = []
        self._srate = 0.0

    def start(self, eeg_stream_name: str | None = None, marker_stream_name: str | None = None):
        self._running = True
        self._setup_outlet()
        self._thread = threading.Thread(
            target=self._run,
            args=(eeg_stream_name, marker_stream_name),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _setup_outlet(self):
        info = StreamInfo(
            name="ErrP_Detector",
            type=ErrPConfig.CONTENT_TYPE,
            channel_count=ErrPConfig.CHANNEL_COUNT,
            nominal_srate=0.0,
            channel_format='float32',
            source_id="errp_detector_001",
        )
        self._outlet = StreamOutlet(info)
        print("ErrP Detector: outlet created")

    def _find_stream(self, name: str | None, stream_type: str) -> StreamInlet | None:
        for _ in range(10):
            if not self._running:
                return None
            streams = resolve_streams(wait_time=1.0)
            for s in streams:
                if name and s.name() == name:
                    return StreamInlet(s, processing_flags=proc_clocksync)
                if not name and s.type() == stream_type:
                    return StreamInlet(s, processing_flags=proc_clocksync)
        return None

    def _run(self, eeg_name: str | None, marker_name: str | None):
        print("ErrP Detector: looking for streams...")
        self._eeg_inlet = self._find_stream(eeg_name, "EEG")
        if not self._eeg_inlet:
            print("ErrP Detector: EEG stream not found")
            return
        self._srate = self._eeg_inlet.info().nominal_srate()
        print(f"ErrP Detector: connected to EEG ({self._srate} Hz)")

        self._marker_inlet = self._find_stream(marker_name, "Markers")
        if not self._marker_inlet:
            print("ErrP Detector: Marker stream not found")
            return
        print("ErrP Detector: connected to Markers")

        max_buffer = int(self._srate * 10)

        while self._running:
            chunk, ts = self._eeg_inlet.pull_chunk(timeout=0.0)
            if ts:
                self._eeg_buffer.extend(chunk)
                self._eeg_ts_buffer.extend(ts)
                while len(self._eeg_buffer) > max_buffer:
                    self._eeg_buffer.pop(0)
                    self._eeg_ts_buffer.pop(0)

            sample, marker_ts = self._marker_inlet.pull_sample(timeout=0.0)
            if sample and marker_ts:
                marker_val = int(round(sample[0]))
                if marker_val in FEEDBACK_MARKERS:
                    self._schedule_classification(marker_ts)

            time.sleep(0.005)

    def _schedule_classification(self, feedback_ts: float):
        delay = self.window_end + 0.05
        threading.Timer(delay, self._classify_window, args=(feedback_ts,)).start()

    def _classify_window(self, feedback_ts: float):
        if not self._eeg_ts_buffer:
            return

        ts_arr = np.array(self._eeg_ts_buffer)
        t_start = feedback_ts + self.window_start
        t_end = feedback_ts + self.window_end
        mask = (ts_arr >= t_start) & (ts_arr <= t_end)

        if mask.sum() < 10:
            print(f"ErrP Detector: not enough samples ({mask.sum()}) for window")
            return

        data = np.array(self._eeg_buffer)[mask].T

        try:
            processed = self.preprocessor.process(data, self._srate)
            probs = self.classifier.predict_proba(processed, self.preprocessor.target_srate)
            if self._outlet:
                self._outlet.push_sample(probs.tolist())
            print(f"ErrP: p_correct={probs[0]:.2f} p_error={probs[1]:.2f}")
        except Exception as e:
            print(f"ErrP classification error: {e}")
