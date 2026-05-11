import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List


class TaskType(Enum):
    REST = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    FORWARD = auto()


TASK_LABELS = {
    TaskType.REST: "Rest",
    TaskType.LEFT_HAND: "Left Hand",
    TaskType.RIGHT_HAND: "Right Hand",
    TaskType.FORWARD: "Forward",
}


@dataclass
class CollectorConfig:
    idle_duration: float = 1.5
    cue_duration: float = 1.5
    imagery_duration: float = 3.5
    feedback_duration: float = 1.5

    n_runs: int = 10
    trials_per_run: int = 24
    break_after_run: int = 5

    error_rate: float = 0.3
    sampling_rate: int = 2048

    markers: Dict[TaskType, int] = field(default=None)
    feedback_markers: Dict[TaskType, int] = field(default=None)
    marker_correct: int = 20
    marker_wrong: int = 21

    def __post_init__(self):
        if self.markers is None:
            self.markers = {
                TaskType.REST: 1,
                TaskType.LEFT_HAND: 2,
                TaskType.RIGHT_HAND: 3,
                TaskType.FORWARD: 4,
            }
        if self.feedback_markers is None:
            self.feedback_markers = {k: v + 10 for k, v in self.markers.items()}

    @property
    def tasks(self) -> List[TaskType]:
        return [TaskType.LEFT_HAND, TaskType.RIGHT_HAND, TaskType.FORWARD, TaskType.REST]

    @property
    def n_classes(self) -> int:
        return len(self.tasks)

    @property
    def trials_per_class_per_run(self) -> int:
        return self.trials_per_run // self.n_classes

    @property
    def total_trials(self) -> int:
        return self.n_runs * self.trials_per_run

    @property
    def trial_duration(self) -> float:
        return self.idle_duration + self.cue_duration + self.imagery_duration + self.feedback_duration

    def get_marker(self, task: TaskType) -> int:
        return self.markers[task]

    def get_feedback_marker(self, task: TaskType) -> int:
        return self.feedback_markers[task]

    def generate_trial_sequence(self) -> List[TaskType]:
        tasks = []
        for task in self.tasks:
            tasks.extend([task] * self.trials_per_class_per_run)
        random.shuffle(tasks)
        return tasks
