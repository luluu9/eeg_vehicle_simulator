from collections import Counter

from src.collector.classifier import MockClassifier
from src.collector.config import TaskType


class TestMockClassifier:
    def test_correct_rate_approximate(self):
        clf = MockClassifier(error_rate=0.3)
        n = 3000
        correct = sum(1 for _ in range(n) if clf.predict(TaskType.LEFT_HAND) == TaskType.LEFT_HAND)
        assert 0.60 < correct / n < 0.80

    def test_wrong_prediction_never_equals_true(self):
        clf = MockClassifier(error_rate=1.0)
        for task in TaskType:
            for _ in range(100):
                assert clf.predict(task) != task

    def test_movement_task_wrong_excludes_rest(self):
        clf = MockClassifier(error_rate=1.0)
        for task in [TaskType.LEFT_HAND, TaskType.RIGHT_HAND, TaskType.FORWARD]:
            for _ in range(200):
                pred = clf.predict(task)
                assert pred != TaskType.REST
                assert pred != task

    def test_rest_wrong_can_be_any_movement(self):
        clf = MockClassifier(error_rate=1.0)
        predictions = Counter(clf.predict(TaskType.REST) for _ in range(600))
        assert TaskType.REST not in predictions
        for t in [TaskType.LEFT_HAND, TaskType.RIGHT_HAND, TaskType.FORWARD]:
            assert predictions[t] > 0

    def test_zero_error_rate_always_correct(self):
        clf = MockClassifier(error_rate=0.0)
        for task in TaskType:
            for _ in range(50):
                assert clf.predict(task) == task
