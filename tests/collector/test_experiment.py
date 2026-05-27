import sys
from collections import Counter
from unittest.mock import MagicMock

import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from src.collector.config import CollectorConfig, TaskType
from src.collector.experiment import ExperimentSession, ExperimentState, COUNTDOWN_FROM


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def session(qapp):
    config = CollectorConfig(n_runs=2, trials_per_run=8, break_after_run=1)
    s = ExperimentSession(config)
    yield s
    s.stop()


class TestExperimentState:
    def test_state_transitions_order(self, session):
        states = []
        session.state_changed.connect(lambda s: states.append(s))
        session.current_task = TaskType.LEFT_HAND

        session._enter_idle()
        session._on_timeout()  # IDLE -> CUE
        session._on_timeout()  # CUE -> IMAGERY
        session._on_timeout()  # IMAGERY -> FEEDBACK

        assert states == [
            ExperimentState.IDLE,
            ExperimentState.CUE,
            ExperimentState.IMAGERY,
            ExperimentState.FEEDBACK,
        ]

    def test_timing_idle(self, session):
        session._enter_idle()
        assert session._timer.interval() == 1500

    def test_timing_cue(self, session):
        session.current_task = TaskType.LEFT_HAND
        session._enter_cue()
        assert session._timer.interval() == 1500

    def test_timing_imagery(self, session):
        session._enter_imagery()
        assert session._timer.interval() == 3500

    def test_timing_feedback(self, session):
        session.current_task = TaskType.LEFT_HAND
        session._enter_feedback()
        assert session._timer.interval() == 1500


class TestExperimentTrials:
    def test_trial_sequence_balanced(self, session):
        session.trial_sequence = session.config.generate_trial_sequence()
        counts = Counter(session.trial_sequence)
        for task in session.config.tasks:
            assert counts[task] == session.config.trials_per_class_per_run

    def test_total_trials_across_runs(self, qapp):
        config = CollectorConfig(n_runs=3, trials_per_run=8, break_after_run=10)
        s = ExperimentSession(config)
        finished = []
        s.finished.connect(lambda: finished.append(True))

        s.start()
        # Simulate all timeouts (extra iterations for countdown between runs)
        for _ in range(3 * 8 * 4 + 3 * COUNTDOWN_FROM):
            if s.state == ExperimentState.FINISHED:
                break
            s._on_timeout()

        assert len(finished) == 1
        assert s._total_trials_done == 24
        s.stop()


class TestExperimentBreak:
    def test_break_after_configured_run(self, qapp):
        config = CollectorConfig(n_runs=4, trials_per_run=4, break_after_run=2)
        s = ExperimentSession(config)
        breaks = []
        s.break_requested.connect(lambda: breaks.append(True))

        s.start()
        # Run through 2 runs (extra iterations for countdown)
        for _ in range(2 * 4 * 4 + 2 * COUNTDOWN_FROM):
            if s.state == ExperimentState.BREAK:
                break
            s._on_timeout()

        assert len(breaks) == 1
        assert s.state == ExperimentState.BREAK
        s.stop()

    def test_resume_from_break_continues(self, qapp):
        config = CollectorConfig(n_runs=4, trials_per_run=4, break_after_run=2)
        s = ExperimentSession(config)

        s.start()
        for _ in range(2 * 4 * 4 + 2 * COUNTDOWN_FROM):
            if s.state == ExperimentState.BREAK:
                break
            s._on_timeout()

        assert s.state == ExperimentState.BREAK
        s.resume_from_break()
        assert s.state == ExperimentState.COUNTDOWN
        s.stop()


class TestExperimentMarkers:
    def test_cue_emits_task_marker(self, qapp):
        config = CollectorConfig()
        logger = MagicMock()
        lsl = MagicMock()
        lsl.lsl_offset = 0.0
        lsl.get_data.return_value = (MagicMock(__len__=lambda s: 0), [])

        s = ExperimentSession(config, lsl_client=lsl, data_logger=logger)
        s.current_task = TaskType.FORWARD
        s._enter_cue()

        logger.add_event.assert_called()
        marker = logger.add_event.call_args[0][1]
        assert marker == config.get_marker(TaskType.FORWARD)
        s.stop()

    def test_feedback_emits_prediction_and_quality_markers(self, qapp):
        config = CollectorConfig()
        logger = MagicMock()
        lsl = MagicMock()
        lsl.lsl_offset = 0.0
        lsl.get_data.return_value = (MagicMock(__len__=lambda s: 0), [])

        s = ExperimentSession(config, lsl_client=lsl, data_logger=logger)
        s.current_task = TaskType.LEFT_HAND
        s._enter_feedback()

        assert logger.add_event.call_count == 2
        markers = [call[0][1] for call in logger.add_event.call_args_list]
        # First: feedback marker (prediction class), second: quality (20 or 21)
        assert markers[1] in (config.marker_correct, config.marker_wrong)
        s.stop()
