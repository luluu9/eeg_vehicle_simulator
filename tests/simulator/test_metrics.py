import math
import time
import pytest
from src.simulator.metrics import MetricsCollector, TrialResult


class TestTrialResult:
    def test_path_efficiency_normal(self):
        r = TrialResult("move", True, 10.0, 20.0, 10.0, 0)
        assert r.path_efficiency == 0.5

    def test_path_efficiency_optimal(self):
        r = TrialResult("move", True, 10.0, 10.0, 10.0, 0)
        assert r.path_efficiency == 1.0

    def test_path_efficiency_capped_at_one(self):
        r = TrialResult("move", True, 10.0, 5.0, 10.0, 0)
        assert r.path_efficiency == 1.0

    def test_path_efficiency_zero_path(self):
        r = TrialResult("rest", True, 5.0, 0.0, 0.0, 0)
        assert r.path_efficiency == 1.0

    def test_path_efficiency_zero_actual_nonzero_optimal(self):
        r = TrialResult("move", False, 0.0, 0.0, 5.0, 0)
        assert r.path_efficiency == 0.0


class TestMetricsCollector:
    def test_path_length_simple(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_position(0, 0)
        mc.record_position(3, 4)
        result = mc.end_trial("move", True, 5.0)
        assert abs(result.path_length - 5.0) < 0.01

    def test_path_length_multiple_segments(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_position(0, 0)
        mc.record_position(1, 0)
        mc.record_position(1, 1)
        result = mc.end_trial("move", True, 2.0)
        assert abs(result.path_length - 2.0) < 0.01

    def test_correction_count(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_correction()
        mc.record_correction()
        mc.record_correction()
        result = mc.end_trial("move", True, 0)
        assert result.correction_count == 3

    def test_trials_accumulate(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("a", True, 1.0)
        mc.start_trial()
        mc.end_trial("b", False, 2.0)
        assert len(mc.trials) == 2

    def test_summary_empty(self):
        mc = MetricsCollector()
        assert mc.summary() == {}

    def test_summary_with_trials(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_position(0, 0)
        mc.record_position(5, 0)
        mc.end_trial("move", True, 5.0)
        mc.start_trial()
        mc.end_trial("rest", False, 0)
        s = mc.summary()
        assert s["total_trials"] == 2
        assert s["completed_trials"] == 1
        assert s["total_corrections"] == 0

    def test_start_trial_resets_state(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_position(0, 0)
        mc.record_correction()
        mc.end_trial("a", True, 0)
        mc.start_trial()
        result = mc.end_trial("b", True, 0)
        assert result.correction_count == 0
        assert result.path_length == 0.0


class TestITR:
    def test_itr_basic(self):
        itr = MetricsCollector.compute_itr(4, 0.8, 5.0)
        assert itr > 0

    def test_itr_zero_accuracy(self):
        assert MetricsCollector.compute_itr(4, 0.0, 5.0) == 0.0

    def test_itr_perfect_accuracy(self):
        assert MetricsCollector.compute_itr(4, 1.0, 5.0) == 0.0

    def test_itr_zero_time(self):
        assert MetricsCollector.compute_itr(4, 0.8, 0.0) == 0.0

    def test_itr_increases_with_accuracy(self):
        itr_low = MetricsCollector.compute_itr(4, 0.4, 5.0)
        itr_high = MetricsCollector.compute_itr(4, 0.9, 5.0)
        assert itr_high > itr_low

    def test_itr_increases_with_speed(self):
        itr_slow = MetricsCollector.compute_itr(4, 0.8, 10.0)
        itr_fast = MetricsCollector.compute_itr(4, 0.8, 5.0)
        assert itr_fast > itr_slow
