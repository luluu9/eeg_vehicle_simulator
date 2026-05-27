from collections import Counter

from src.collector.config import CollectorConfig, TaskType


class TestCollectorConfig:
    def test_default_config_values(self):
        config = CollectorConfig()
        assert config.idle_duration == 1.5
        assert config.cue_duration == 1.5
        assert config.imagery_duration == 3.5
        assert config.feedback_duration == 1.5
        assert config.n_runs == 10
        assert config.trials_per_run == 24
        assert config.error_rate == 0.3
        assert config.break_every_n_runs == 2

    def test_total_trials(self):
        config = CollectorConfig()
        assert config.total_trials == 240

    def test_trial_duration(self):
        config = CollectorConfig()
        assert config.trial_duration == 8.0

    def test_four_classes(self):
        config = CollectorConfig()
        assert config.n_classes == 4
        assert set(config.tasks) == {
            TaskType.LEFT_HAND,
            TaskType.RIGHT_HAND,
            TaskType.FORWARD,
            TaskType.REST,
        }

    def test_trials_per_class_per_run_balanced(self):
        config = CollectorConfig()
        assert config.trials_per_class_per_run == 6
        assert config.trials_per_class_per_run * config.n_classes == config.trials_per_run

    def test_markers_complete(self):
        config = CollectorConfig()
        for task in config.tasks:
            assert task in config.markers
            assert task in config.feedback_markers
            assert config.get_feedback_marker(task) == config.get_marker(task) + 10

    def test_marker_values_unique(self):
        config = CollectorConfig()
        task_markers = list(config.markers.values())
        feedback_markers = list(config.feedback_markers.values())
        all_markers = task_markers + feedback_markers + [config.marker_correct, config.marker_wrong]
        assert len(all_markers) == len(set(all_markers))

    def test_generate_trial_sequence_length(self):
        config = CollectorConfig()
        seq = config.generate_trial_sequence()
        assert len(seq) == config.trials_per_run

    def test_generate_trial_sequence_balanced(self):
        config = CollectorConfig()
        seq = config.generate_trial_sequence()
        counts = Counter(seq)
        for task in config.tasks:
            assert counts[task] == 6

    def test_generate_trial_sequence_randomized(self):
        config = CollectorConfig()
        sequences = [tuple(config.generate_trial_sequence()) for _ in range(20)]
        assert len(set(sequences)) > 1
