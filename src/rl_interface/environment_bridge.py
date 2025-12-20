
import numpy as np
import time
from .buffer import ProbabilisticBuffer
from .lidar import SimulatedLidar

class RLInterface:
    def __init__(self, 
                 buffer_config: dict = None,
                 lidar_config: dict = None):
        
        self.buffer = ProbabilisticBuffer(**(buffer_config or {}))
        self.lidar = SimulatedLidar(**(lidar_config or {}))
        
        # Latency monitoring
        self.last_process_time = 0
        
    def get_rl_observation(self, raw_probs: np.ndarray, env) -> np.ndarray:
        """
        Main entry point for the RL loop.
        
        Args:
            raw_probs: Raw classifier output (5 classes).
            env: The gymnasium environment instance (WheelchairRacing).
            
        Returns:
            observation: Concatenated vector of [Smoothed Probs (5), Lidar (7), Speed (1)]
        """
        t0 = time.perf_counter()
        
        # 1. Update Buffer
        smoothed_probs = self.buffer.update(raw_probs)
        
        # 2. Get Lidar
        lidar_dists = self.lidar.scan(env)
        
        # 3. Get Vehicle State (Speed)
        speed = 0.0
        if hasattr(env, 'unwrapped') and env.unwrapped.car:
             v = env.unwrapped.car.hull.linearVelocity
             speed = np.sqrt(v[0]**2 + v[1]**2)
             
        # Normalize Lidar (0..1) usually good for RL, but user asked for distances. 
        # Using raw for now, or scaled.
        # Let's scale Lidar by max_range to keep it 0-1ish
        lidar_scaled = lidar_dists / self.lidar.max_range
        
        # 4. Concatenate
        obs = np.concatenate([
            smoothed_probs,         # 5
            lidar_scaled,           # 7
            [speed / 100.0]         # 1 (Approx max speed scaling)
        ])
        
        self.last_process_time = (time.perf_counter() - t0) * 1000 # ms
        
        return obs.astype(np.float32)

    def get_latency_ms(self):
        return self.last_process_time
