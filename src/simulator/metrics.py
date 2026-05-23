import math
import time
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Decision:
    timestamp: float
    x: float
    y: float
    angle: float
    action: list[float]
    mi_class: int
    errp_error_prob: float


@dataclass
class TrialResult:
    goal_type: str
    completed: bool
    goal_completion_pct: float
    completion_time: float
    optimal_time: float
    correction_count: int
    expected_class: int | None = None
    road_adherence: float | None = None
    decisions: list[Decision] = field(default_factory=list)

    @property
    def normalized_time(self) -> float | None:
        if not self.completed or self.optimal_time <= 0:
            return None
        return self.completion_time / self.optimal_time

    @property
    def accuracy(self) -> float | None:
        if self.expected_class is None or not self.decisions:
            return None
        correct = sum(1 for d in self.decisions if d.mi_class == self.expected_class)
        return correct / len(self.decisions)


class MetricsCollector:
    def __init__(self):
        self._positions: list[tuple[float, float]] = []
        self._start_time: float = 0.0
        self._correction_count: int = 0
        self._trials: list[TrialResult] = []
        self._decisions: list[Decision] = []
        self._on_road_steps: int = 0
        self._total_steps: int = 0

    def start_trial(self):
        self._positions.clear()
        self._decisions.clear()
        self._start_time = time.monotonic()
        self._correction_count = 0
        self._on_road_steps = 0
        self._total_steps = 0

    def record_position(self, x: float, y: float):
        self._positions.append((x, y))

    def record_decision(self, x: float, y: float, angle: float,
                        action: np.ndarray, mi_class: int,
                        errp_error_prob: float = 0.0,
                        on_road: bool | None = None):
        self._decisions.append(Decision(
            timestamp=time.monotonic() - self._start_time,
            x=x, y=y, angle=angle,
            action=action.tolist(),
            mi_class=mi_class,
            errp_error_prob=errp_error_prob,
        ))
        if on_road is not None:
            self._total_steps += 1
            if on_road:
                self._on_road_steps += 1

    def record_correction(self):
        self._correction_count += 1

    def end_trial(self, goal_type: str, completed: bool,
                  goal_completion_pct: float, optimal_time: float,
                  expected_class: int | None = None) -> TrialResult:
        elapsed = time.monotonic() - self._start_time
        road_adherence = None
        if self._total_steps > 0:
            road_adherence = self._on_road_steps / self._total_steps

        result = TrialResult(
            goal_type=goal_type,
            completed=completed,
            goal_completion_pct=min(goal_completion_pct, 1.0),
            completion_time=elapsed,
            optimal_time=optimal_time,
            correction_count=self._correction_count,
            expected_class=expected_class,
            road_adherence=road_adherence,
            decisions=list(self._decisions),
        )
        self._trials.append(result)
        return result

    @property
    def trials(self) -> list[TrialResult]:
        return list(self._trials)

    def summary(self) -> dict:
        if not self._trials:
            return {}
        completed = [t for t in self._trials if t.completed]
        norm_times = [t.normalized_time for t in completed if t.normalized_time is not None]
        road_vals = [t.road_adherence for t in self._trials if t.road_adherence is not None]

        itr_trials = [t for t in self._trials if t.expected_class is not None and t.decisions]
        itr = None
        if itr_trials:
            total_correct = sum(
                sum(1 for d in t.decisions if d.mi_class == t.expected_class)
                for t in itr_trials
            )
            total_decisions = sum(len(t.decisions) for t in itr_trials)
            accuracy = total_correct / total_decisions if total_decisions > 0 else 0.0
            total_time_min = sum(t.completion_time for t in itr_trials) / 60.0
            avg_command_time = (total_time_min / total_decisions) * 60.0 if total_decisions > 0 else 0.0
            itr = self.compute_itr(4, accuracy, avg_command_time)

        return {
            "total_trials": len(self._trials),
            "completed_trials": len(completed),
            "avg_goal_completion_pct": float(np.mean([t.goal_completion_pct for t in self._trials])),
            "avg_normalized_time": float(np.mean(norm_times)) if norm_times else None,
            "avg_road_adherence": float(np.mean(road_vals)) if road_vals else None,
            "total_corrections": sum(t.correction_count for t in self._trials),
            "itr_bits_per_min": itr,
            "trials": [
                {
                    "goal_type": t.goal_type,
                    "completed": t.completed,
                    "goal_completion_pct": t.goal_completion_pct,
                    "completion_time": t.completion_time,
                    "optimal_time": t.optimal_time,
                    "correction_count": t.correction_count,
                    "road_adherence": t.road_adherence,
                    "decisions": [
                        {
                            "timestamp": d.timestamp,
                            "x": d.x,
                            "y": d.y,
                            "angle": d.angle,
                            "action": d.action,
                            "mi_class": d.mi_class,
                            "errp_error_prob": d.errp_error_prob,
                        }
                        for d in t.decisions
                    ],
                }
                for t in self._trials
            ],
        }

    @staticmethod
    def compute_itr(n_classes: int, accuracy: float, avg_command_time: float) -> float:
        if accuracy <= 0 or accuracy >= 1 or avg_command_time <= 0 or n_classes < 2:
            return 0.0
        p = accuracy
        n = n_classes
        bits = math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
        return max(0.0, bits * (60.0 / avg_command_time))
