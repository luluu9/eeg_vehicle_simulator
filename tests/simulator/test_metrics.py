import math
import time
import numpy as np
import pytest
from src.simulator.metrics import MetricsCollector, TrialResult, Decision


class TestDecision:
    def test_fields_stored(self):
        d = Decision(
            timestamp=1.5, x=10.0, y=20.0, angle=0.5,
            action=[0.0, 0.3, 0.0], mi_class=3, errp_error_prob=0.8,
        )
        assert d.timestamp == 1.5
        assert d.x == 10.0
        assert d.y == 20.0
        assert d.angle == 0.5
        assert d.action == [0.0, 0.3, 0.0]
        assert d.mi_class == 3
        assert d.errp_error_prob == 0.8

    def test_errp_zero_by_default_construction(self):
        d = Decision(timestamp=0, x=0, y=0, angle=0, action=[0, 0, 0], mi_class=0, errp_error_prob=0.0)
        assert d.errp_error_prob == 0.0


class TestTrialResult:
    def test_normalized_time_completed(self):
        r = TrialResult("move", True, 1.0, 10.0, 5.0, 0)
        assert r.normalized_time == 2.0

    def test_normalized_time_exact_optimal(self):
        r = TrialResult("move", True, 1.0, 5.0, 5.0, 0)
        assert r.normalized_time == 1.0

    def test_normalized_time_not_completed(self):
        r = TrialResult("move", False, 0.5, 10.0, 5.0, 0)
        assert r.normalized_time is None

    def test_normalized_time_zero_optimal(self):
        r = TrialResult("rest", True, 1.0, 5.0, 0.0, 0)
        assert r.normalized_time is None

    def test_normalized_time_negative_optimal(self):
        r = TrialResult("move", True, 1.0, 5.0, -1.0, 0)
        assert r.normalized_time is None

    def test_goal_completion_pct_stored(self):
        r = TrialResult("move", False, 0.73, 10.0, 5.0, 0)
        assert r.goal_completion_pct == 0.73

    def test_road_adherence_none_by_default(self):
        r = TrialResult("move", True, 1.0, 5.0, 5.0, 0)
        assert r.road_adherence is None

    def test_road_adherence_stored(self):
        r = TrialResult("trajectory", True, 1.0, 30.0, 20.0, 0, road_adherence=0.92)
        assert r.road_adherence == 0.92

    def test_decisions_default_empty(self):
        r = TrialResult("move", True, 1.0, 5.0, 5.0, 0)
        assert r.decisions == []


class TestMetricsCollector:
    def test_correction_count(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_correction()
        mc.record_correction()
        mc.record_correction()
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert result.correction_count == 3

    def test_trials_accumulate(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("a", True, 1.0, 5.0)
        mc.start_trial()
        mc.end_trial("b", False, 0.5, 5.0)
        assert len(mc.trials) == 2
        assert mc.trials[0].goal_type == "a"
        assert mc.trials[1].goal_type == "b"

    def test_trials_returns_copy(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("a", True, 1.0, 5.0)
        trials = mc.trials
        trials.clear()
        assert len(mc.trials) == 1

    def test_start_trial_resets_corrections(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_correction()
        mc.end_trial("a", True, 1.0, 5.0)
        mc.start_trial()
        result = mc.end_trial("b", True, 1.0, 5.0)
        assert result.correction_count == 0

    def test_start_trial_resets_decisions(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0)
        mc.end_trial("a", True, 1.0, 5.0)
        mc.start_trial()
        result = mc.end_trial("b", True, 1.0, 5.0)
        assert len(result.decisions) == 0

    def test_start_trial_resets_road_adherence(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=False)
        mc.end_trial("a", True, 1.0, 5.0)
        mc.start_trial()
        result = mc.end_trial("b", True, 1.0, 5.0)
        assert result.road_adherence is None

    def test_goal_completion_capped_at_one(self):
        mc = MetricsCollector()
        mc.start_trial()
        result = mc.end_trial("move", True, 1.5, 5.0)
        assert result.goal_completion_pct == 1.0

    def test_goal_completion_partial(self):
        mc = MetricsCollector()
        mc.start_trial()
        result = mc.end_trial("move", False, 0.6, 5.0)
        assert abs(result.goal_completion_pct - 0.6) < 1e-9

    def test_goal_completion_zero(self):
        mc = MetricsCollector()
        mc.start_trial()
        result = mc.end_trial("move", False, 0.0, 5.0)
        assert result.goal_completion_pct == 0.0


class TestRecordDecision:
    def test_basic_decision_fields(self):
        mc = MetricsCollector()
        mc.start_trial()
        action = np.array([0.5, 0.3, 0.0])
        mc.record_decision(1.0, 2.0, 0.5, action, 2, errp_error_prob=0.7)
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.x == 1.0
        assert d.y == 2.0
        assert d.angle == 0.5
        assert d.action == [0.5, 0.3, 0.0]
        assert d.mi_class == 2
        assert d.errp_error_prob == 0.7

    def test_errp_defaults_to_zero(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0)
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert result.decisions[0].errp_error_prob == 0.0

    def test_multiple_decisions_order_preserved(self):
        mc = MetricsCollector()
        mc.start_trial()
        for i in range(5):
            mc.record_decision(float(i), 0, 0, np.zeros(3), i % 4, errp_error_prob=i * 0.1)
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert len(result.decisions) == 5
        for i, d in enumerate(result.decisions):
            assert d.x == float(i)
            assert d.mi_class == i % 4
            assert abs(d.errp_error_prob - i * 0.1) < 1e-9

    def test_timestamps_monotonically_increase(self):
        mc = MetricsCollector()
        mc.start_trial()
        for _ in range(3):
            mc.record_decision(0, 0, 0, np.zeros(3), 0)
            time.sleep(0.01)
        result = mc.end_trial("move", True, 1.0, 5.0)
        timestamps = [d.timestamp for d in result.decisions]
        assert timestamps == sorted(timestamps)
        assert timestamps[0] >= 0
        assert timestamps[-1] > timestamps[0]

    def test_action_is_list_not_ndarray(self):
        mc = MetricsCollector()
        mc.start_trial()
        action = np.array([-0.5, 0.3, 0.0])
        mc.record_decision(0, 0, 0, action, 1)
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert isinstance(result.decisions[0].action, list)
        assert result.decisions[0].action == [-0.5, 0.3, 0.0]

    def test_action_mutation_safety(self):
        mc = MetricsCollector()
        mc.start_trial()
        action = np.array([1.0, 2.0, 3.0])
        mc.record_decision(0, 0, 0, action, 0)
        action[0] = 999.0
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert result.decisions[0].action[0] == 1.0


class TestRoadAdherence:
    def test_all_on_road(self):
        mc = MetricsCollector()
        mc.start_trial()
        for _ in range(10):
            mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=True)
        result = mc.end_trial("trajectory", True, 1.0, 10.0)
        assert result.road_adherence == 1.0

    def test_all_off_road(self):
        mc = MetricsCollector()
        mc.start_trial()
        for _ in range(10):
            mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=False)
        result = mc.end_trial("trajectory", False, 0.3, 10.0)
        assert result.road_adherence == 0.0

    def test_mixed(self):
        mc = MetricsCollector()
        mc.start_trial()
        action = np.zeros(3)
        mc.record_decision(0, 0, 0, action, 0, on_road=True)
        mc.record_decision(0, 0, 0, action, 0, on_road=True)
        mc.record_decision(0, 0, 0, action, 0, on_road=False)
        mc.record_decision(0, 0, 0, action, 0, on_road=True)
        result = mc.end_trial("trajectory", True, 1.0, 10.0)
        assert abs(result.road_adherence - 0.75) < 0.01

    def test_none_on_road_means_no_road_adherence(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0)
        mc.record_decision(0, 0, 0, np.zeros(3), 0)
        result = mc.end_trial("move", True, 1.0, 5.0)
        assert result.road_adherence is None

    def test_partial_none_only_counts_explicit(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0)
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=True)
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=False)
        result = mc.end_trial("trajectory", True, 1.0, 10.0)
        assert abs(result.road_adherence - 0.5) < 0.01


class TestSummary:
    def test_empty(self):
        mc = MetricsCollector()
        assert mc.summary() == {}

    def test_all_completed(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("move", True, 1.0, 5.0)
        mc.start_trial()
        mc.end_trial("move", True, 0.8, 10.0)
        s = mc.summary()
        assert s["total_trials"] == 2
        assert s["completed_trials"] == 2
        assert abs(s["avg_goal_completion_pct"] - 0.9) < 0.01

    def test_none_completed(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("move", False, 0.3, 5.0)
        mc.start_trial()
        mc.end_trial("move", False, 0.5, 5.0)
        s = mc.summary()
        assert s["completed_trials"] == 0
        assert s["avg_normalized_time"] is None
        assert abs(s["avg_goal_completion_pct"] - 0.4) < 0.01

    def test_road_adherence_in_summary(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=True)
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=True)
        mc.end_trial("trajectory", True, 1.0, 10.0)
        mc.start_trial()
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=False)
        mc.record_decision(0, 0, 0, np.zeros(3), 0, on_road=False)
        mc.end_trial("trajectory", False, 0.5, 10.0)
        s = mc.summary()
        assert abs(s["avg_road_adherence"] - 0.5) < 0.01

    def test_road_adherence_none_when_no_road_trials(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.end_trial("move", True, 1.0, 5.0)
        s = mc.summary()
        assert s["avg_road_adherence"] is None

    def test_corrections_summed_across_trials(self):
        mc = MetricsCollector()
        mc.start_trial()
        mc.record_correction()
        mc.record_correction()
        mc.end_trial("a", True, 1.0, 5.0)
        mc.start_trial()
        mc.record_correction()
        mc.end_trial("b", True, 1.0, 5.0)
        s = mc.summary()
        assert s["total_corrections"] == 3


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

    def test_itr_two_classes(self):
        itr = MetricsCollector.compute_itr(2, 0.8, 5.0)
        assert itr > 0

    def test_itr_one_class_invalid(self):
        assert MetricsCollector.compute_itr(1, 0.8, 5.0) == 0.0


class TestTrialAccuracy:
    def test_all_correct(self):
        decisions = [Decision(0, 0, 0, 0, [0, 0, 0], mi_class=1, errp_error_prob=0.0) for _ in range(10)]
        r = TrialResult("turn_left", True, 1.0, 10.0, 5.0, 0, expected_class=1, decisions=decisions)
        assert r.accuracy == 1.0

    def test_all_wrong(self):
        decisions = [Decision(0, 0, 0, 0, [0, 0, 0], mi_class=2, errp_error_prob=0.0) for _ in range(10)]
        r = TrialResult("turn_left", True, 1.0, 10.0, 5.0, 0, expected_class=1, decisions=decisions)
        assert r.accuracy == 0.0

    def test_half_correct(self):
        decisions = [
            Decision(0, 0, 0, 0, [0, 0, 0], mi_class=1, errp_error_prob=0.0),
            Decision(0, 0, 0, 0, [0, 0, 0], mi_class=2, errp_error_prob=0.0),
            Decision(0, 0, 0, 0, [0, 0, 0], mi_class=1, errp_error_prob=0.0),
            Decision(0, 0, 0, 0, [0, 0, 0], mi_class=0, errp_error_prob=0.0),
        ]
        r = TrialResult("turn_left", True, 1.0, 10.0, 5.0, 0, expected_class=1, decisions=decisions)
        assert r.accuracy == 0.5

    def test_no_expected_class(self):
        decisions = [Decision(0, 0, 0, 0, [0, 0, 0], mi_class=1, errp_error_prob=0.0)]
        r = TrialResult("trajectory", True, 1.0, 10.0, 5.0, 0, decisions=decisions)
        assert r.accuracy is None

    def test_no_decisions(self):
        r = TrialResult("turn_left", True, 1.0, 10.0, 5.0, 0, expected_class=1)
        assert r.accuracy is None


class TestITRInSummary:
    def _make_decisions(self, mc, mi_class: int, count: int) -> None:
        action = np.zeros(3)
        for _ in range(count):
            mc.record_decision(0, 0, 0, action, mi_class)

    def test_itr_computed_for_task_a(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=1, count=8)
        self._make_decisions(mc, mi_class=2, count=2)
        time.sleep(0.02)
        result = mc.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        assert result.accuracy == 0.8
        assert result.completion_time > 0
        s = mc.summary()
        assert s["itr_bits_per_min"] is not None
        assert s["itr_bits_per_min"] > 0

    def test_itr_none_for_task_b_only(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=3, count=10)
        mc.end_trial("trajectory", True, 1.0, 10.0)
        s = mc.summary()
        assert s["itr_bits_per_min"] is None

    def test_itr_mixed_tasks(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=1, count=10)
        mc.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        mc.start_trial()
        self._make_decisions(mc, mi_class=3, count=10)
        mc.end_trial("trajectory", True, 1.0, 10.0)
        s = mc.summary()
        # ITR only uses the task_a trial (expected_class set)
        # accuracy = 1.0 -> compute_itr returns 0 (perfect accuracy edge case)
        assert s["itr_bits_per_min"] is not None

    def test_itr_perfect_accuracy_gives_zero(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=1, count=10)
        mc.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        s = mc.summary()
        assert s["itr_bits_per_min"] == 0.0

    def test_itr_zero_accuracy_gives_zero(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=2, count=10)
        mc.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        s = mc.summary()
        assert s["itr_bits_per_min"] == 0.0

    def test_itr_higher_accuracy_yields_higher_itr(self):
        mc_low = MetricsCollector()
        mc_low.start_trial()
        self._make_decisions(mc_low, mi_class=1, count=6)
        self._make_decisions(mc_low, mi_class=2, count=4)
        time.sleep(0.05)
        mc_low.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        s_low = mc_low.summary()

        mc_high = MetricsCollector()
        mc_high.start_trial()
        self._make_decisions(mc_high, mi_class=1, count=9)
        self._make_decisions(mc_high, mi_class=2, count=1)
        time.sleep(0.05)
        mc_high.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)
        s_high = mc_high.summary()

        assert s_low["itr_bits_per_min"] > 0
        assert s_high["itr_bits_per_min"] > s_low["itr_bits_per_min"]

    def test_itr_multiple_trials_aggregated(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=1, count=8)
        self._make_decisions(mc, mi_class=0, count=2)
        time.sleep(0.01)
        mc.end_trial("turn_left", True, 1.0, 5.0, expected_class=1)

        mc.start_trial()
        self._make_decisions(mc, mi_class=0, count=7)
        self._make_decisions(mc, mi_class=1, count=3)
        time.sleep(0.01)
        mc.end_trial("rest", True, 1.0, 5.0, expected_class=0)

        s = mc.summary()
        assert s["itr_bits_per_min"] is not None
        # 15 correct out of 20 = 75% accuracy
        assert s["itr_bits_per_min"] > 0

    def test_itr_uses_wolpaw_formula(self):
        mc = MetricsCollector()
        mc.start_trial()
        self._make_decisions(mc, mi_class=3, count=7)
        self._make_decisions(mc, mi_class=1, count=3)
        time.sleep(0.1)
        result = mc.end_trial("forward", True, 1.0, 5.0, expected_class=3)

        accuracy = 0.7
        n = 4
        expected_bits = math.log2(n) + accuracy * math.log2(accuracy) + (1 - accuracy) * math.log2((1 - accuracy) / (n - 1))
        avg_time = result.completion_time / 10
        expected_itr = expected_bits * (60.0 / avg_time)

        s = mc.summary()
        assert abs(s["itr_bits_per_min"] - expected_itr) < 0.01
