import math
import time
from collections import deque

import numpy as np

from .strategies import REST_ACTION
from ..common.constants import StudyClass

SLOW_GAS = 0.03
SLOW_STEER = 0.1
BRAKE_STRENGTH = 0.8
DEFAULT_HISTORY_SECONDS = 0.8
DEFAULT_BRAKE_SECONDS = 0.3
DEFAULT_FPS = 50
DEFAULT_ERRP_THRESHOLD = 0.5
DEFAULT_ERRP_ACTION_DELAY = 0.0
DEFAULT_CRUISE_SPEED = 4.0
NOMINAL_TURN_RATE = math.radians(21.0)
STOP_SPEED_EPS = 0.1

CONTINUOUS_ACTION_MAP = {
    StudyClass.REST.value: np.array([0.0, 0.0, 0.0], dtype=np.float32),
    StudyClass.LEFT.value: np.array([-SLOW_STEER, 0.0, 0.0], dtype=np.float32),
    StudyClass.RIGHT.value: np.array([SLOW_STEER, 0.0, 0.0], dtype=np.float32),
    StudyClass.FORWARD.value: np.array([0.0, SLOW_GAS, 0.0], dtype=np.float32),
}

BRAKE_ACTION = np.array([0.0, 0.0, BRAKE_STRENGTH], dtype=np.float32)


def continuous_action(class_idx: int) -> np.ndarray:
    return CONTINUOUS_ACTION_MAP.get(class_idx, REST_ACTION).copy()


def inverse_action(action: np.ndarray) -> np.ndarray:
    return np.array([-action[0], -action[1], 0.0], dtype=np.float32)


class MovementController:
    NORMAL = "NORMAL"
    ERRP_STOPPING = "ERRP_STOPPING"
    REVERSING = "REVERSING"
    COOLDOWN = "COOLDOWN"
    BRAKING = "BRAKING"

    def __init__(self, strategy_name: str = "baseline",
                 errp_threshold: float = DEFAULT_ERRP_THRESHOLD,
                 errp_action_delay: float = DEFAULT_ERRP_ACTION_DELAY,
                 history_seconds: float = DEFAULT_HISTORY_SECONDS,
                 brake_seconds: float = DEFAULT_BRAKE_SECONDS,
                 cruise_speed: float = DEFAULT_CRUISE_SPEED,
                 fps: int = DEFAULT_FPS, clock=time.monotonic):
        self.strategy_name = strategy_name
        self.errp_threshold = errp_threshold
        self.errp_action_delay = errp_action_delay
        self.cruise_speed = cruise_speed
        self.history = deque(maxlen=max(1, int(round(history_seconds * fps))))
        self._brake_total = max(1, int(round(brake_seconds * fps)))
        self._brake_frames = 0
        self._last_class = StudyClass.REST.value
        self._clock = clock
        self.state = self.NORMAL
        self._reverse_queue: list[np.ndarray] = []
        self._cooldown_until = 0.0
        self.correction_count = 0

    @property
    def is_reversing(self) -> bool:
        return self.state == self.REVERSING

    @property
    def is_correcting(self) -> bool:
        return self.state in (self.ERRP_STOPPING, self.REVERSING, self.COOLDOWN)

    def step(self, dominant_class: int, errp_prob: float = 0.0,
             speed: float = 0.0) -> np.ndarray:
        now = self._clock()

        if (self.strategy_name == "errp" and not self.is_correcting
                and errp_prob >= self.errp_threshold):
            self._reverse_queue = list(self.history)
            self.history.clear()
            self._last_class = StudyClass.REST.value
            self.correction_count += 1
            self.state = self.ERRP_STOPPING

        if self.state == self.ERRP_STOPPING:
            if speed > STOP_SPEED_EPS:
                return BRAKE_ACTION.copy()
            self.state = self.REVERSING

        if self.state == self.REVERSING:
            if self._reverse_queue:
                return inverse_action(self._reverse_queue.pop())
            self._cooldown_until = now + self.errp_action_delay
            self.state = self.COOLDOWN

        if self.state == self.COOLDOWN:
            if now < self._cooldown_until:
                return REST_ACTION.copy()
            self.state = self.NORMAL

        if self.state == self.BRAKING:
            if self._brake_frames > 0:
                self._brake_frames -= 1
                return BRAKE_ACTION.copy()
            self.state = self.NORMAL

        if dominant_class != self._last_class and self._last_class != StudyClass.REST.value:
            self._last_class = dominant_class
            self.history.clear()
            self.state = self.BRAKING
            self._brake_frames = self._brake_total - 1
            return BRAKE_ACTION.copy()

        self._last_class = dominant_class
        action = continuous_action(dominant_class)
        if action[1] > 0.0 and speed >= self.cruise_speed:
            action[1] = 0.0
        self.history.append(action.copy())
        return action
