import time
from collections import deque

import numpy as np

from .strategies import REST_ACTION
from ..common.constants import StudyClass

SLOW_GAS = 0.15
SLOW_STEER = 0.3
DEFAULT_HISTORY_SECONDS = 0.8
DEFAULT_FPS = 50
DEFAULT_ERRP_THRESHOLD = 0.5
DEFAULT_ERRP_ACTION_DELAY = 0.0

CONTINUOUS_ACTION_MAP = {
    StudyClass.REST.value: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    StudyClass.LEFT.value: np.array([-SLOW_STEER, SLOW_GAS, 0.0], dtype=np.float32),
    StudyClass.RIGHT.value: np.array([SLOW_STEER, SLOW_GAS, 0.0], dtype=np.float32),
    StudyClass.FORWARD.value: np.array([0.0, SLOW_GAS, 0.0], dtype=np.float32),
}


def continuous_action(class_idx: int) -> np.ndarray:
    return CONTINUOUS_ACTION_MAP.get(class_idx, REST_ACTION).copy()


def inverse_action(action: np.ndarray) -> np.ndarray:
    return np.array([-action[0], -action[1], 0.0], dtype=np.float32)


class MovementController:
    NORMAL = "NORMAL"
    REVERSING = "REVERSING"

    def __init__(self, strategy_name: str = "baseline",
                 errp_threshold: float = DEFAULT_ERRP_THRESHOLD,
                 errp_action_delay: float = DEFAULT_ERRP_ACTION_DELAY,
                 history_seconds: float = DEFAULT_HISTORY_SECONDS,
                 fps: int = DEFAULT_FPS, clock=time.monotonic):
        self.strategy_name = strategy_name
        self.errp_threshold = errp_threshold
        self.errp_action_delay = errp_action_delay
        self.history = deque(maxlen=max(1, int(round(history_seconds * fps))))
        self._clock = clock
        self.state = self.NORMAL
        self._reverse_queue: list[np.ndarray] = []
        self._pending_snapshot: list[np.ndarray] | None = None
        self._pending_time = 0.0
        self.correction_count = 0

    @property
    def is_reversing(self) -> bool:
        return self.state == self.REVERSING

    def step(self, dominant_class: int, errp_prob: float = 0.0) -> np.ndarray:
        now = self._clock()

        if self.state == self.REVERSING:
            if self._reverse_queue:
                return inverse_action(self._reverse_queue.pop())
            self.state = self.NORMAL
            self._pending_snapshot = None

        if (self.strategy_name == "errp" and self._pending_snapshot is None
                and errp_prob >= self.errp_threshold):
            self._pending_snapshot = list(self.history)
            self._pending_time = now + self.errp_action_delay
            self.correction_count += 1

        if self._pending_snapshot is not None and now >= self._pending_time:
            self._reverse_queue = self._pending_snapshot
            self._pending_snapshot = None
            self.history.clear()
            self.state = self.REVERSING
            if self._reverse_queue:
                return inverse_action(self._reverse_queue.pop())
            self.state = self.NORMAL

        action = continuous_action(dominant_class)
        self.history.append(action.copy())
        return action
