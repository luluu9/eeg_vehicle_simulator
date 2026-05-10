import sys

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.collector.config import TaskType
from src.collector.stimulus import (
    ACTION_MAP,
    CUE_SYMBOLS,
    WheelchairStimulus,
    draw_cue_overlay,
    draw_feedback_border,
    frame_to_pixmap,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class TestActionMap:
    def test_all_tasks_have_actions(self):
        for task in TaskType:
            assert task in ACTION_MAP
            assert ACTION_MAP[task].shape == (3,)

    def test_rest_is_no_action(self):
        np.testing.assert_array_equal(ACTION_MAP[TaskType.REST], [0, 0, 0])

    def test_forward_has_gas_no_steer(self):
        action = ACTION_MAP[TaskType.FORWARD]
        assert action[0] == 0.0  # no steer
        assert action[1] > 0.0   # gas
        assert action[2] == 0.0  # no brake

    def test_left_steers_negative(self):
        assert ACTION_MAP[TaskType.LEFT_HAND][0] < 0

    def test_right_steers_positive(self):
        assert ACTION_MAP[TaskType.RIGHT_HAND][0] > 0


class TestCueSymbols:
    def test_all_tasks_have_symbols(self):
        for task in TaskType:
            assert task in CUE_SYMBOLS
            assert len(CUE_SYMBOLS[task]) > 0


class TestWheelchairStimulus:
    def test_renders_rgb_frame(self):
        stim = WheelchairStimulus()
        frame = stim.get_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        stim.close()

    def test_step_returns_new_frame(self):
        stim = WheelchairStimulus()
        action = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        frame = stim.step(action)
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3
        stim.close()

    def test_reset_does_not_crash(self):
        stim = WheelchairStimulus()
        stim.reset()
        frame = stim.get_frame()
        assert frame is not None
        stim.close()


class TestFrameConversion:
    def test_frame_to_pixmap(self, qapp):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        pixmap = frame_to_pixmap(frame)
        assert not pixmap.isNull()
        assert pixmap.width() == 200
        assert pixmap.height() == 100

    def test_cue_overlay_returns_pixmap(self, qapp):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        pixmap = frame_to_pixmap(frame)
        result = draw_cue_overlay(pixmap, TaskType.LEFT_HAND)
        assert not result.isNull()
        assert result.size() == pixmap.size()

    def test_feedback_border_returns_pixmap(self, qapp):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        pixmap = frame_to_pixmap(frame)
        result_correct = draw_feedback_border(pixmap, True)
        result_wrong = draw_feedback_border(pixmap, False)
        assert not result_correct.isNull()
        assert not result_wrong.isNull()
