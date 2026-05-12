from abc import ABC, abstractmethod
import time
import numpy as np
from ..common.constants import LSLChannel, StudyClass
from ..rl_interface.environment_bridge import RLInterface


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
    def compute(self, all_probs, selected_stream, errp_data=None):
        """
        all_probs: dict {name: np.array}
        selected_stream: str
        errp_data: dict {name: np.array} or None (ErrP predictions)
        Returns: np.array([steer, gas, brake]) or None (if no decision/relax)
        """
        pass
    
    @abstractmethod
    def get_params(self):
        return {}
        
    @abstractmethod
    def adjust_param(self, key, delta):
        pass

    @abstractmethod
    def get_debug_info(self, selected_stream):
        """
        Returns a dict of debug info for the specific selected stream.
        """
        return {}

    def set_env(self, env):
        pass

class ThresholdStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Threshold")
        self.threshold = 0.70
        
    def compute(self, all_probs, selected_stream, errp_data=None):
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

    def get_debug_info(self, selected_stream):
        return {"Type": "Simple Threshold"}


class AccumulatorStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Accumulator")
        self.decay = 0.9
        self.threshold = 1.5
        # We need state PER STREAM, because user might switch stream
        self.buffers = {} 
        
    def compute(self, all_probs, selected_stream, errp_data=None):
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

    def get_debug_info(self, selected_stream):
        info = {}
        if selected_stream in self.buffers:
            buf = self.buffers[selected_stream]
            info["Buffer"] = np.round(buf, 2)
        return info


class RLInterfaceStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("RL_Interface")
        self.interfaces = {} # One per stream
        self.env = None
        
        self.smoothing_threshold = 0.7
        self.smoothing_window = 10
        self.min_duration = 300 # ms
        
        self.last_debug_state = {}

        
    def set_env(self, env):
        self.env = env
        
    def _get_interface(self, stream_name):
        if stream_name not in self.interfaces:
            self.interfaces[stream_name] = RLInterface(
                buffer_config={
                    'window_size': self.smoothing_window, 
                    'threshold': self.smoothing_threshold, 
                    'min_duration_ms': self.min_duration
                }
            )
        return self.interfaces[stream_name]

    def compute(self, all_probs, selected_stream, errp_data=None):
        if selected_stream not in all_probs:
            return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)
            
        interface = self._get_interface(selected_stream)
        raw_probs = all_probs[selected_stream]
        
        # Get Obs (Updates buffer internally)
        # We need env for lidar. If no env, lidar will be empty/default.
        obs = interface.get_rl_observation(raw_probs, self.env)
        
        # Use intention from buffer to drive
        intention = interface.buffer.get_intention()
        action = ControlMapper.map_class_to_action(intention.value)
        steer, gas, brake = action[0], action[1], action[2]
        
        # --- Safety Override (Heuristic Agent) ---
        lidar = obs[5:12]
        speed = obs[12] # Speed is last element (index 12: 5 probs + 7 lidar)
        
        dist_right = lidar[0] 
        dist_left = lidar[6]
        dist_center = lidar[3]
        min_dist = np.min(lidar)
        
        # Tuning (Assuming Max Range 150, Track Radius ~100 but Width ~6.6)
        # Center of track ~ 0.044 normalized.
        # Wall ~ 0.0.
        SAFE_DIST = 0.015  # ~2.2 units
        SLOW_DIST = 0.05   # ~7.5 units
        MAX_SPEED = 0.3    # ~30 units/s 
        
        override_status = "NONE"

        # 1. Speed Limit
        if speed > MAX_SPEED:
            gas = 0.0
            # brake = 0.1 # Gentle drag?
        
        # 2. Proximity Speed Scaling (Safety Bubble)
        # Closer to wall = Slower max speed / accel
        if min_dist < SLOW_DIST:
            speed_factor = min_dist / SLOW_DIST
            gas *= speed_factor
            
        # 3. Collision Prevention (Steering)
        # If steering Left (steer < 0) and Left is blocked
        if steer < -0.1 and dist_left < SAFE_DIST:
            steer = 0.0 
            override_status = "LEFT BLOCKED"
            
        # If steering Right (steer > 0) and Right is blocked
        if steer > 0.1 and dist_right < SAFE_DIST:
            steer = 0.0
            override_status = "RIGHT BLOCKED"
            
        # 4. Frontal Collision (Gas)
        if gas > 0 and dist_center < SAFE_DIST:
             gas = 0.0
             brake = 0.8
             override_status = "FRONT BLOCKED"
             
        # Capture Debug State
        self.last_debug_state[selected_stream] = {
            "Intention": intention.name,
            "Lidar (R)": f"{dist_right:.3f}",
            "Lidar (L)": f"{dist_left:.3f}",
            "Lidar (C)": f"{dist_center:.3f}",
            "Speed": f"{speed:.2f}",
            "Override": override_status
        }
        
        return np.array([steer, gas, brake], dtype=np.float32)


    def get_params(self):
        return {
            "Threshold": self.smoothing_threshold, 
        }
        
    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.smoothing_threshold = max(0.1, min(1.0, self.smoothing_threshold + delta))
            for name, iface in self.interfaces.items():
                iface.buffer.threshold = self.smoothing_threshold

    def get_debug_info(self, selected_stream):
        if selected_stream in self.last_debug_state:
            return self.last_debug_state[selected_stream]
        return {"Status": "Waiting for input"}



class StrategyManager:
    def __init__(self):
        self.strategies = [ThresholdStrategy(), AccumulatorStrategy(), RLInterfaceStrategy()]
        self.active_idx = 0
        self.selected_stream = None # Name of stream driving the car
    
    def set_env(self, env):
        for s in self.strategies:
            s.set_env(env)

        
    def get_active(self):
        return self.strategies[self.active_idx]
        
    def next_strategy(self):
        self.active_idx = (self.active_idx + 1) % len(self.strategies)
        
    def process(self, all_probs, errp_data=None):
        # Auto-select first available stream if none selected
        if self.selected_stream is None or self.selected_stream not in all_probs:
            if all_probs:
                self.selected_stream = list(all_probs.keys())[0]
            else:
                return ControlMapper.map_class_to_action(LSLChannel.RELAX.value)

        return self.strategies[self.active_idx].compute(all_probs, self.selected_stream, errp_data)


# --- Study Fusion Strategies (4-class MI + ErrP) ---

REST_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)

STUDY_ACTION_MAP = {
    StudyClass.REST.value: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    StudyClass.LEFT.value: np.array([-0.5, 0.3, 0.0], dtype=np.float32),
    StudyClass.RIGHT.value: np.array([0.5, 0.3, 0.0], dtype=np.float32),
    StudyClass.FORWARD.value: np.array([0.0, 0.3, 0.0], dtype=np.float32),
}


def study_action(class_idx: int) -> np.ndarray:
    return STUDY_ACTION_MAP.get(class_idx, REST_ACTION).copy()


def _get_errp_error_prob(errp_data: dict | None) -> float:
    if not errp_data:
        return 0.0
    for probs in errp_data.values():
        return float(probs[1]) if len(probs) >= 2 else 0.0
    return 0.0


class BaselineStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("Baseline")
        self.threshold = 0.50

    def compute(self, all_probs, selected_stream, errp_data=None):
        if selected_stream not in all_probs:
            return REST_ACTION.copy()
        probs = all_probs[selected_stream]
        max_idx = int(np.argmax(probs[:4]))
        if probs[max_idx] >= self.threshold:
            return study_action(max_idx)
        return REST_ACTION.copy()

    def get_params(self):
        return {"Threshold": self.threshold}

    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.threshold = max(0.0, min(1.0, self.threshold + delta))

    def get_debug_info(self, selected_stream):
        return {"Type": "Baseline (MI only)"}


class StopStrategy(BaseStrategy):
    COOLDOWN_DURATION = 1.0

    def __init__(self):
        super().__init__("STOP")
        self.threshold = 0.50
        self.errp_threshold = 0.50
        self._vetoed = False
        self._veto_time = 0.0
        self.correction_count = 0

    def compute(self, all_probs, selected_stream, errp_data=None):
        now = time.monotonic()

        error_prob = _get_errp_error_prob(errp_data)
        if not self._vetoed and error_prob >= self.errp_threshold:
            self._vetoed = True
            self._veto_time = now
            self.correction_count += 1

        if self._vetoed:
            if now - self._veto_time >= self.COOLDOWN_DURATION:
                self._vetoed = False
            return REST_ACTION.copy()

        if selected_stream not in all_probs:
            return REST_ACTION.copy()
        probs = all_probs[selected_stream]
        max_idx = int(np.argmax(probs[:4]))
        if probs[max_idx] >= self.threshold:
            return study_action(max_idx)
        return REST_ACTION.copy()

    def get_params(self):
        return {"Threshold": self.threshold, "ErrP_Thr": self.errp_threshold}

    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.threshold = max(0.0, min(1.0, self.threshold + delta))
        elif key == "ErrP_Thr":
            self.errp_threshold = max(0.0, min(1.0, self.errp_threshold + delta))

    def get_debug_info(self, selected_stream):
        return {
            "Type": "STOP (MI + ErrP veto)",
            "Vetoed": self._vetoed,
            "Corrections": self.correction_count,
        }

    def reset_state(self):
        self._vetoed = False
        self.correction_count = 0


class AutocorrectStrategy(BaseStrategy):
    CORRECTION_DURATION = 1.0

    def __init__(self):
        super().__init__("Autocorrect")
        self.threshold = 0.50
        self.errp_threshold = 0.50
        self._correcting = False
        self._correction_time = 0.0
        self._correction_action = REST_ACTION
        self.correction_count = 0
        self._last_probs: np.ndarray | None = None

    def compute(self, all_probs, selected_stream, errp_data=None):
        now = time.monotonic()

        error_prob = _get_errp_error_prob(errp_data)
        if not self._correcting and error_prob >= self.errp_threshold and self._last_probs is not None:
            sorted_idx = np.argsort(self._last_probs[:4])[::-1]
            second_best = int(sorted_idx[1])
            self._correction_action = study_action(second_best)
            self._correcting = True
            self._correction_time = now
            self.correction_count += 1

        if self._correcting:
            if now - self._correction_time >= self.CORRECTION_DURATION:
                self._correcting = False
            return self._correction_action.copy()

        if selected_stream not in all_probs:
            return REST_ACTION.copy()
        probs = all_probs[selected_stream]
        self._last_probs = probs.copy()
        max_idx = int(np.argmax(probs[:4]))
        if probs[max_idx] >= self.threshold:
            return study_action(max_idx)
        return REST_ACTION.copy()

    def get_params(self):
        return {"Threshold": self.threshold, "ErrP_Thr": self.errp_threshold}

    def adjust_param(self, key, delta):
        if key == "Threshold":
            self.threshold = max(0.0, min(1.0, self.threshold + delta))
        elif key == "ErrP_Thr":
            self.errp_threshold = max(0.0, min(1.0, self.errp_threshold + delta))

    def get_debug_info(self, selected_stream):
        return {
            "Type": "Autocorrect (MI + ErrP → 2nd best)",
            "Correcting": self._correcting,
            "Corrections": self.correction_count,
        }

    def reset_state(self):
        self._correcting = False
        self._last_probs = None
        self.correction_count = 0


STUDY_STRATEGIES = {
    "baseline": BaselineStrategy,
    "stop": StopStrategy,
    "autocorrect": AutocorrectStrategy,
}
