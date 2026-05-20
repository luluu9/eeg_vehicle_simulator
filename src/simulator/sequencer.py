import random
from dataclasses import dataclass, field


TASKS = ["A", "B"]
STRATEGIES = ["baseline", "stop", "autocorrect"]
ERRP_STRATEGIES = {"stop", "autocorrect"}


@dataclass
class Experiment:
    task_id: int
    task: str
    strategy: str
    status: str = "pending"
    result: dict | None = None


class ExperimentSequencer:
    def __init__(self):
        self._experiments = self._generate_sequence()

    def _generate_sequence(self) -> list[Experiment]:
        task_a = [Experiment(0, "A", s) for s in random.sample(STRATEGIES, len(STRATEGIES))]
        task_b = [Experiment(0, "B", s) for s in random.sample(STRATEGIES, len(STRATEGIES))]
        seq = task_a + task_b
        for i, exp in enumerate(seq):
            exp.task_id = i + 1
        return seq

    @property
    def current(self) -> Experiment | None:
        for exp in self._experiments:
            if exp.status == "pending":
                return exp
        return None

    def advance(self, status: str, result: dict | None = None):
        exp = self.current
        if exp is not None:
            exp.status = status
            exp.result = result

    def override_current(self, task: str, strategy: str):
        exp = self.current
        if exp is not None:
            exp.task = task
            exp.strategy = strategy

    @property
    def all_done(self) -> bool:
        return self.current is None
