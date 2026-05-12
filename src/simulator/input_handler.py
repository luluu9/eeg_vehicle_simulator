import threading
import time
import numpy as np
from pylsl import resolve_streams, StreamInlet
from ..common.constants import LSLConfig, ErrPConfig

class MultiStreamMonitor:
    def __init__(self):
        self.streams = {} # name -> StreamInlet
        self.latest_data = {} # name -> (probs, timestamp)
        self.errp_streams = {}
        self.errp_data = {}
        self.running = False
        self.discovery_thread = None
        self.poll_thread = None
        self.lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self.discovery_thread.start()
        
        self.poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.poll_thread.start()
        
    def stop(self):
        self.running = False
        if self.discovery_thread: self.discovery_thread.join()
        if self.poll_thread: self.poll_thread.join()
        
    def _discovery_loop(self):
        while self.running:
            found_streams_info = resolve_streams(wait_time=1.0)
            found_mi_names = set()
            found_errp_names = set()
            
            for info in found_streams_info:
                name = info.name()
                if info.type() == LSLConfig.CONTENT_TYPE:
                    found_mi_names.add(name)
                    with self.lock:
                        if name not in self.streams:
                            print(f"Found MI stream: {name}")
                            inlet = StreamInlet(info)
                            self.streams[name] = inlet
                            self.latest_data[name] = (np.zeros(5), 0)
                elif info.type() == ErrPConfig.CONTENT_TYPE:
                    found_errp_names.add(name)
                    with self.lock:
                        if name not in self.errp_streams:
                            print(f"Found ErrP stream: {name}")
                            inlet = StreamInlet(info)
                            self.errp_streams[name] = inlet
                            self.errp_data[name] = (np.zeros(ErrPConfig.CHANNEL_COUNT), 0)
            
            with self.lock:
                for name in [n for n in self.streams if n not in found_mi_names]:
                    print(f"MI stream disappeared: {name}")
                    del self.streams[name]
                    self.latest_data.pop(name, None)
                for name in [n for n in self.errp_streams if n not in found_errp_names]:
                    print(f"ErrP stream disappeared: {name}")
                    del self.errp_streams[name]
                    self.errp_data.pop(name, None)
                            
            time.sleep(2.0)
            
    def _poll_loop(self):
        while self.running:
            with self.lock:
                mi_names = list(self.streams.keys())
                errp_names = list(self.errp_streams.keys())

            for name in mi_names:
                with self.lock:
                    inlet = self.streams.get(name)
                if inlet:
                    sample, ts = inlet.pull_sample(timeout=0.0)
                    if sample:
                        with self.lock:
                            self.latest_data[name] = (np.array(sample), ts)

            for name in errp_names:
                with self.lock:
                    inlet = self.errp_streams.get(name)
                if inlet:
                    sample, ts = inlet.pull_sample(timeout=0.0)
                    if sample:
                        with self.lock:
                            self.errp_data[name] = (np.array(sample), ts)
            
            time.sleep(0.01)
            
    def get_probabilities(self):
        with self.lock:
            return {k: v[0].copy() for k, v in self.latest_data.items()}

    def get_errp(self):
        with self.lock:
            return {k: v[0].copy() for k, v in self.errp_data.items()}
