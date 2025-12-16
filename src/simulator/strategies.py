from abc import ABC, abstractmethod
import numpy as np
from ..common.constants import LSLChannel

class ControlMapper:
    @staticmethod
    def map_class_to_action(class_idx):
        steer, gas, brake = 0.0, 0.0, 0.0
        
        # Mapping based on LSLChannel enum values:
        # RELAX=0, LEFT=1, RIGHT=2, BOTH=3, FEET=4
        if class_idx == LSLChannel.LEFT.value:
            steer = -0.5
            gas = 0.0
        elif class_idx == LSLChannel.RIGHT.value:
            steer = 0.5
            gas = 0.0
        elif class_idx == LSLChannel.BOTH.value:
            gas = 0.3 
        elif class_idx == LSLChannel.FEET.value:
            brake = 0.8 
            
        return np.array([steer, gas, brake], dtype=np.float32)

class BaseStrategy(ABC):
    def __init__(self, name):
        self._name = name
    
    @property
    def name(self):
        return self._name

    @abstractmethod
    def compute(self, all_probs, selected_stream):
        """
        all_probs: dict {name: np.array}
        selected_stream: str
        Returns: np.array([steer, gas, brake]) or None (if no decision/relax)
        """
        pass
    
    @abstractmethod
    def get_params(self):
        return {}
        
    @abstractmethod
    def adjust_param(self, key, delta):
        pass

class ThresholdStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Threshold")
        self.threshold = 0.70
        
    def compute(self, all_probs, selected_stream):
        if selected_stream not in all_probs:
            return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)
            
        probs = all_probs[selected_stream] # shape (5,)
        max_idx = np.argmax(probs)
        val = probs[max_idx]
        
        if val >= self.threshold:
            return ControlMapper.map_class_to_action(max_idx)
        else:
            return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)
            
    def get_params(self):
        return {"Threshold": self.threshold}
        
    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.threshold = max(0.0, min(1.0, self.threshold + delta))

class AccumulatorStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Accumulator")
        self.decay = 0.9
        self.threshold = 1.5
        # We need state PER STREAM, because user might switch stream
        self.buffers = {} 
        
    def compute(self, all_probs, selected_stream):
        if selected_stream not in all_probs:
            return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)
            
        probs = all_probs[selected_stream]
        
        if selected_stream not in self.buffers:
            self.buffers[selected_stream] = np.zeros(5)
            
        # Update buffer
        self.buffers[selected_stream] *= self.decay
        self.buffers[selected_stream] += probs
        
        # Check
        buf = self.buffers[selected_stream]
        max_idx = np.argmax(buf)
        val = buf[max_idx]
        
        if val >= self.threshold:
            return ControlMapper.map_class_to_action(max_idx)
        
        return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)

    def get_params(self):
        return {"Decay": self.decay, "Threshold": self.threshold}
        
    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.threshold = max(0.1, min(5.0, self.threshold + delta))
        elif key == "Decay":
            self.decay = max(0.5, min(0.99, self.decay + (delta * 0.1)))

class StrategyManager:
    def __init__(self):
        self.strategies = [ThresholdStrategy(), AccumulatorStrategy()]
        self.active_idx = 0
        self.selected_stream = None # Name of stream driving the car
        
    def get_active(self):
        return self.strategies[self.active_idx]
        
    def next_strategy(self):
        self.active_idx = (self.active_idx + 1) % len(self.strategies)
        
    def process(self, all_probs):
        # Auto-select first available stream if none selected
        if self.selected_stream is None or self.selected_stream not in all_probs:
            if all_probs:
                self.selected_stream = list(all_probs.keys())[0]
            else:
                return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)

        return self.strategies[self.active_idx].compute(all_probs, self.selected_stream)
