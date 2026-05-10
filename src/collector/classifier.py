import random
from .config import TaskType


MOVEMENT_TASKS = [TaskType.LEFT_HAND, TaskType.RIGHT_HAND, TaskType.FORWARD]


class MockClassifier:
    def __init__(self, error_rate: float = 0.3):
        self.error_rate = error_rate

    def predict(self, true_label: TaskType) -> TaskType:
        if random.random() >= self.error_rate:
            return true_label

        if true_label in MOVEMENT_TASKS:
            wrong_choices = [t for t in MOVEMENT_TASKS if t != true_label]
        else:
            wrong_choices = [t for t in TaskType if t != true_label]

        return random.choice(wrong_choices)
