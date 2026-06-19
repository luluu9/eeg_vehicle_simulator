from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from pylsl import local_clock
from .lsl_io import DataHandler, PredictionBroadcaster
from .classifiers import BaseClassifier
from .preprocessor import EEGPreprocessor
from collections import deque
import numpy as np

DEFAULT_VOTE_SIZE = 50
DEFAULT_INTERVAL = 0.02

class ClassifierState:
    def __init__(self, classifier: BaseClassifier, vote_size: int = DEFAULT_VOTE_SIZE):
        self.classifier = classifier
        self.broadcaster = PredictionBroadcaster(classifier.name, channel_count=classifier.output_size)
        # Default target to min or reasonable middle
        self.target_window = classifier.min_window
        self.active = True
        # History for visualization (simple list of last N predictions)
        self.history = []
        # Rolling buffer of recent argmax decisions for majority voting
        self.vote_buffer = deque(maxlen=vote_size)

    def set_vote_size(self, vote_size: int):
        self.vote_buffer = deque(self.vote_buffer, maxlen=vote_size)

class PredictorEngine(QObject):
    # Signals
    prediction_made = pyqtSignal(str, np.ndarray, float) # clf_name, probs, latency
    aggregate_made = pyqtSignal(str, np.ndarray) # clf_name, vote histogram
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.data_handler = DataHandler()
        self.preprocessor = EEGPreprocessor(target_srate=256.0)
        self.classifiers = {} # name -> ClassifierState
        self._vote_size = DEFAULT_VOTE_SIZE
        
        # Main Loop Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._interval = DEFAULT_INTERVAL
        
    def add_classifier(self, clf: BaseClassifier):
        if clf.name in self.classifiers:
            print(f"Classifier {clf.name} already exists.")
            return
            
        state = ClassifierState(clf, vote_size=self._vote_size)
        self.classifiers[clf.name] = state
        print(f"Added classifier: {clf.name}")
        
    def set_interval(self, seconds: float):
        self._interval = max(0.01, min(5.0, seconds))
        if self.timer.isActive():
            self.timer.start(int(self._interval * 1000))

    def set_vote_size(self, n: int):
        self._vote_size = max(1, int(n))
        for state in self.classifiers.values():
            state.set_vote_size(self._vote_size)
            
    def set_classifier_window(self, name: str, window_sec: float):
        if name in self.classifiers:
            c = self.classifiers[name]
            # Clamp
            w = max(c.classifier.min_window, min(c.classifier.max_window, window_sec))
            c.target_window = w
            
    def find_streams(self):
        return self.data_handler.find_streams()
        
    def start_stream_input(self, target_stream=None):
        if target_stream:
            self.data_handler.connect(target_stream)
            self.data_handler.start()
            self.timer.start(int(self._interval * 1000))
            return True
        else:
            self.error_occurred.emit("No valid EEG stream selected/found")
            return False
            
    def stop(self):
        self.timer.stop()
        self.data_handler.stop()
        
    def _tick(self):
        # 1. Check data availability
        # Max required from all classifiers
        max_req = 0
        for s in self.classifiers.values():
            if s.active:
                max_req = max(max_req, s.target_window)
        
        if max_req == 0:
            return
            
        # Get 5x data samples for preprocessing
        max_req *= 5
        data, timestamps = self.data_handler.get_latest_window(max_req)
        
        if data is None:
            return
            
        fs = self.data_handler.srate
        
        # 2. Central Preprocessing
        # Filter the entire retrieved window once
        processed_data = self.preprocessor.process(data, fs)
        proc_fs = self.preprocessor.target_srate
        n_proc_samples = processed_data.shape[1]
        
        # 3. Predict for each classifier
        for name, state in self.classifiers.items():
            if not state.active:
                continue
                
            needed_samples = int(state.target_window * proc_fs)
            if n_proc_samples < needed_samples:
                print(f"Not enough data for {name}, skipping")
                continue
                
            input_slice = processed_data[:, -needed_samples:]
            
            try:
                probs = state.classifier.predict_proba(input_slice, proc_fs)

                # Majority voting over the rolling buffer of recent decisions
                state.vote_buffer.append(int(np.argmax(probs)))
                output_size = state.classifier.output_size
                counts = np.bincount(state.vote_buffer, minlength=output_size).astype(np.float32)
                histogram = counts / counts.sum()

                # Broadcast the aggregated decision (normalized vote histogram)
                state.broadcaster.push_prediction(histogram)
                
                # Calculate latency (freshness of data)
                # timestamps corresponds to data. Since we slice processed_data, assume linear time.
                # The end of the processed window corresponds to the end of the raw window.
                # So we can just take the last timestamp from the raw data timestamps.
                # (Assuming get_latest_window returns timestamps corresponding to 'data')
                
                latency = 0.0
                if timestamps is not None and len(timestamps) > 0:
                    last_ts = timestamps[-1]
                    now = local_clock()
                    latency = now - last_ts
                
                self.prediction_made.emit(name, probs, latency)
                self.aggregate_made.emit(name, histogram)
                
            except Exception as e:
                print(f"Error in {name}: {e}")
