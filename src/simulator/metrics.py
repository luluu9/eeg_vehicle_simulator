import math
import time
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrialResult:
    goal_type: str
    completed: bool
    completion_time: float
    path_length: float
    optimal_path_length: float
    correction_count: int

    @property
    def path_efficiency(self) -> float:
        if self.path_length <= 0:
            return 1.0 if self.optimal_path_length == 0 else 0.0
        return min(1.0, self.optimal_path_length / self.path_length)


class MetricsCollector:
    def __init__(self):
        self._positions: list[tuple[float, float]] = []
        self._start_time: float = 0.0
        self._correction_count: int = 0
        self._trials: list[TrialResult] = []

    def start_trial(self):
        self._positions.clear()
        self._start_time = time.monotonic()
        self._correction_count = 0

    def record_position(self, x: float, y: float):
        self._positions.append((x, y))

    def record_correction(self):
        self._correction_count += 1

    def end_trial(self, goal_type: str, completed: bool, optimal_length: float) -> TrialResult:
        elapsed = time.monotonic() - self._start_time
        result = TrialResult(
            goal_type=goal_type,
            completed=completed,
            completion_time=elapsed,
            path_length=self._compute_path_length(),
            optimal_path_length=optimal_length,
            correction_count=self._correction_count,
        )
        self._trials.append(result)
        return result

    def _compute_path_length(self) -> float:
        if len(self._positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._positions)):
            dx = self._positions[i][0] - self._positions[i - 1][0]
            dy = self._positions[i][1] - self._positions[i - 1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    @property
    def trials(self) -> list[TrialResult]:
        return list(self._trials)

    def summary(self) -> dict:
        if not self._trials:
            return {}
        completed = [t for t in self._trials if t.completed]
        return {
            "total_trials": len(self._trials),
            "completed_trials": len(completed),
            "avg_completion_time": float(np.mean([t.completion_time for t in completed])) if completed else 0.0,
            "avg_path_efficiency": float(np.mean([t.path_efficiency for t in completed])) if completed else 0.0,
            "total_corrections": sum(t.correction_count for t in self._trials),
        }

    @staticmethod
    def compute_itr(n_classes: int, accuracy: float, avg_command_time: float) -> float:
        if accuracy <= 0 or accuracy >= 1 or avg_command_time <= 0 or n_classes < 2:
            return 0.0
        p = accuracy
        n = n_classes
        bits = math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
        return max(0.0, bits * (60.0 / avg_command_time))
