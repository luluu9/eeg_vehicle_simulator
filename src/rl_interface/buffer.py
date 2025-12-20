import numpy as np
import time
from collections import deque
from enum import IntEnum

class Intention(IntEnum):
    RELAX = 0
    LEFT = 1
    RIGHT = 2
    BOTH_HANDS = 3
    BOTH_FEET = 4

class ProbabilisticBuffer:
    def __init__(self, 
                 num_classes: int = 5, 
                 window_size: int = 10, 
                 threshold: float = 0.7, 
                 min_duration_ms: float = 300,
                 smoothing_factor: float = 0.3):
        """
        Buffer to smooth flickery probabilities and stabilize state transitions.
        
        Args:
            num_classes: Number of classification classes. 
            window_size: Size of sliding window for voting/averaging.
            threshold: Probability threshold to consider a state change valid.
            min_duration_ms: Time in milliseconds the signal must be above threshold to trigger change.
            smoothing_factor: Alpha for EMA (0.0 to 1.0). Higher = more responsive, less smooth.
        """
        self.num_classes = num_classes
        self.threshold = threshold
        self.min_duration_s = min_duration_ms / 1000.0
        self.smoothing_factor = smoothing_factor
        
        # Buffers
        self.history = deque(maxlen=window_size)
        self.smoothed_probs = np.zeros(num_classes)
        
        # State Management
        self.current_intention = Intention.RELAX
        self.pending_intention = None
        self.pending_start_time = 0.0
        
        # For EMA initialization
        self.first_update = True

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Push new probabilities into the buffer and update internal state.
        
        Args:
            probabilities: array of shape (num_classes,) containing predicted probabilities.
            
        Returns:
            smoothed_probabilities: The current EMA smoothed probability vector.
        """
        if len(probabilities) != self.num_classes:
            # Handle mismatch silently or pad/clip? Stick to error ensures correctness.
            raise ValueError(f"Expected {self.num_classes} probabilities, got {len(probabilities)}")
            
        self.history.append(probabilities)
        
        # Update EMA
        if self.first_update:
            self.smoothed_probs = probabilities
            self.first_update = False
        else:
            self.smoothed_probs = (self.smoothing_factor * probabilities) + \
                                  ((1 - self.smoothing_factor) * self.smoothed_probs)
        
        # Normalize just in case (though EMA of normalized is normalized)
        total = np.sum(self.smoothed_probs)
        if total > 0:
            self.smoothed_probs /= total
            
        self._update_state_machine()
        
        return self.smoothed_probs

    def _update_state_machine(self):
        """
        Checks thresholds and timers to potentially switch the active intention.
        """
        # Find strong candidate
        max_idx = np.argmax(self.smoothed_probs)
        max_prob = self.smoothed_probs[max_idx]
        
        if max_prob >= self.threshold:
            candidate = Intention(max_idx)
            
            if candidate != self.current_intention:
                # We have a candidate for change
                if candidate == self.pending_intention:
                    # Check duration
                    elapsed = time.time() - self.pending_start_time
                    if elapsed >= self.min_duration_s:
                        # Commit change
                        self.current_intention = candidate
                        self.pending_intention = None
                else:
                    # New candidate, start timer
                    self.pending_intention = candidate
                    self.pending_start_time = time.time()
            else:
                # Candidate is same as current, reset pending
                self.pending_intention = None
        else:
            # Signal too weak, reset pending transition (optional: could drift to Relax?)
            # Requirement: "Only change ... if ... exceeds threshold".
            # So if signal is weak, we hold current state? 
            # OR does Relax (class 0) become the default if everything is low?
            # Typically classifiers output Relax prob high if nothing else matches.
            # So we assume if "Relax" is high, it will trigger the logic above to switch to Relax.
            # If ALL are low (high entropy), we just reset pending.
            self.pending_intention = None

    def get_intention(self) -> Intention:
        """Returns the current stable intention."""
        return self.current_intention
        
    def get_smoothed_probabilities(self) -> np.ndarray:
        """Returns the current smoothed probability vector."""
        return self.smoothed_probs

    def reset(self):
        self.history.clear()
        self.smoothed_probs = np.zeros(self.num_classes)
        self.first_update = True
        self.current_intention = Intention.RELAX
        self.pending_intention = None
