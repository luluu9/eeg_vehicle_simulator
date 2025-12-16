import threading
import time
import numpy as np
from pylsl import resolve_streams, StreamInlet
from ..common.constants import LSLConfig

class MultiStreamMonitor:
    def __init__(self):
        self.streams = {} # name -> StreamInlet
        self.latest_data = {} # name -> (probs, timestamp)
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
            # Look for streams with type definition from config
            # Wait time 1s
            found_streams_info = resolve_streams(wait_time=1.0)
            found_stream_names = set()
            
            for info in found_streams_info:
                if info.type() == LSLConfig.CONTENT_TYPE: # "Probabilities"
                    name = info.name()
                    found_stream_names.add(name)
                    with self.lock:
                        if name not in self.streams:
                            print(f"Found new prediction stream: {name}")
                            inlet = StreamInlet(info)
                            self.streams[name] = inlet
                            self.latest_data[name] = (np.zeros(5), 0)
            
            # Destroy streams that no longer exist
            with self.lock:
                streams_to_remove = [name for name in self.streams if name not in found_stream_names]
                for name in streams_to_remove:
                    print(f"Stream disappeared: {name}")
                    del self.streams[name]
                    if name in self.latest_data:
                        del self.latest_data[name]
                            
            time.sleep(2.0)
            
    def _poll_loop(self):
        while self.running:
            # Iterate over a COPY of keys to avoid runtime error if discovery adds one
            with self.lock:
                names = list(self.streams.keys())
                
            for name in names:
                inlet = None
                with self.lock:
                    inlet = self.streams.get(name)
                    
                if inlet:
                    # pull_sample (not chunk, we just want latest)
                    # output streams are irregular (0 srate), so pull_sample(timeout) is fine
                    sample, ts = inlet.pull_sample(timeout=0.0)
                    if sample:
                        with self.lock:
                            self.latest_data[name] = (np.array(sample), ts)
            
            time.sleep(0.01) # 100Hz polling is plenty for control
            
    def get_probabilities(self):
        """Returns snapshot of {name: np.array([p1..p5])}"""
        with self.lock:
            # Return copies just to be safe vs race conditions
            # Only return data, ignore timestamp for this simple API
            return {k: v[0].copy() for k, v in self.latest_data.items()}
