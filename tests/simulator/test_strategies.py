import time
import numpy as np
import pytest
from src.simulator.strategies import (
    BaselineStrategy, StopStrategy, AutocorrectStrategy,
    study_action, STUDY_STRATEGIES, REST_ACTION,
)
from src.common.constants import StudyClass


class TestStudyAction:
    def test_rest_is_zero(self):
        a = study_action(StudyClass.REST.value)
        np.testing.assert_array_equal(a, [0, 0, 0])

    def test_left_steers_negative(self):
        a = study_action(StudyClass.LEFT.value)
        assert a[0] < 0

    def test_right_steers_positive(self):
        a = study_action(StudyClass.RIGHT.value)
        assert a[0] > 0

    def test_forward_has_gas(self):
        a = study_action(StudyClass.FORWARD.value)
        assert a[0] == 0 and a[1] > 0

    def test_returns_copy(self):
        a1 = study_action(0)
        a2 = study_action(0)
        a1[0] = 999
        assert a2[0] != 999


class TestBaselineStrategy:
    def _probs(self, dominant_idx, value=0.8):
        p = np.array([0.05, 0.05, 0.05, 0.05])
        p[dominant_idx] = value
        return p

    def test_selects_dominant_class(self):
        s = BaselineStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        action = s.compute(probs, "stream")
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_rest_when_below_threshold(self):
        s = BaselineStrategy()
        probs = {"stream": np.array([0.25, 0.25, 0.25, 0.25])}
        action = s.compute(probs, "stream")
        np.testing.assert_array_equal(action, REST_ACTION)

    def test_rest_when_no_stream(self):
        s = BaselineStrategy()
        action = s.compute({}, "missing")
        np.testing.assert_array_equal(action, REST_ACTION)

    def test_ignores_errp(self):
        s = BaselineStrategy()
        probs = {"stream": self._probs(StudyClass.LEFT.value)}
        errp = {"errp": np.array([0.0, 1.0])}
        action = s.compute(probs, "stream", errp)
        expected = study_action(StudyClass.LEFT.value)
        np.testing.assert_array_equal(action, expected)


class TestStopStrategy:
    def _probs(self, dominant_idx, value=0.8):
        p = np.array([0.05, 0.05, 0.05, 0.05])
        p[dominant_idx] = value
        return p

    def test_normal_operation(self):
        s = StopStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        action = s.compute(probs, "stream")
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_veto_on_errp(self):
        s = StopStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        action = s.compute(probs, "stream", errp)
        np.testing.assert_array_equal(action, REST_ACTION)

    def test_veto_increments_correction(self):
        s = StopStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        assert s.correction_count == 1

    def test_veto_stays_during_cooldown(self):
        s = StopStrategy()
        s.COOLDOWN_DURATION = 10.0
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        action = s.compute(probs, "stream")
        np.testing.assert_array_equal(action, REST_ACTION)

    def test_veto_clears_after_cooldown(self):
        s = StopStrategy()
        s.COOLDOWN_DURATION = 0.0
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        time.sleep(0.01)
        action = s.compute(probs, "stream")
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_no_double_veto_during_cooldown(self):
        s = StopStrategy()
        s.COOLDOWN_DURATION = 10.0
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        s.compute(probs, "stream", errp)
        assert s.correction_count == 1

    def test_reset_state(self):
        s = StopStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        s.reset_state()
        assert s.correction_count == 0
        assert not s._vetoed


class TestAutocorrectStrategy:
    def _probs(self, dominant_idx, second_idx, dom_val=0.7, sec_val=0.2):
        p = np.array([0.025, 0.025, 0.025, 0.025])
        p[dominant_idx] = dom_val
        p[second_idx] = sec_val
        return p

    def test_normal_operation(self):
        s = AutocorrectStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        action = s.compute(probs, "stream")
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_autocorrect_uses_second_best(self):
        s = AutocorrectStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        s.compute(probs, "stream")
        errp = {"errp": np.array([0.2, 0.8])}
        action = s.compute(probs, "stream", errp)
        expected = study_action(StudyClass.LEFT.value)
        np.testing.assert_array_equal(action, expected)

    def test_autocorrect_increments_correction(self):
        s = AutocorrectStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        s.compute(probs, "stream")
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        assert s.correction_count == 1

    def test_no_correction_without_prior_probs(self):
        s = AutocorrectStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        errp = {"errp": np.array([0.2, 0.8])}
        action = s.compute(probs, "stream", errp)
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_correction_clears_after_duration(self):
        s = AutocorrectStrategy()
        s.CORRECTION_DURATION = 0.0
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        s.compute(probs, "stream")
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        time.sleep(0.01)
        action = s.compute(probs, "stream")
        expected = study_action(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(action, expected)

    def test_reset_state(self):
        s = AutocorrectStrategy()
        probs = {"stream": self._probs(StudyClass.FORWARD.value, StudyClass.LEFT.value)}
        s.compute(probs, "stream")
        errp = {"errp": np.array([0.2, 0.8])}
        s.compute(probs, "stream", errp)
        s.reset_state()
        assert s.correction_count == 0
        assert not s._correcting


class TestStudyStrategies:
    def test_all_registered(self):
        assert "baseline" in STUDY_STRATEGIES
        assert "stop" in STUDY_STRATEGIES
        assert "autocorrect" in STUDY_STRATEGIES

    def test_instantiation(self):
        for name, cls in STUDY_STRATEGIES.items():
            s = cls()
            assert s.name
