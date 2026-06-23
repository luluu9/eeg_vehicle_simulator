import numpy as np

from src.simulator.controller import (
    MovementController, continuous_action, inverse_action,
    SLOW_GAS, SLOW_STEER, BRAKE_STRENGTH,
)
from src.common.constants import StudyClass


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class TestContinuousAction:
    def test_rest_is_zero(self):
        np.testing.assert_array_equal(continuous_action(StudyClass.REST.value), [0, 0, 0])

    def test_forward_has_slow_gas(self):
        a = continuous_action(StudyClass.FORWARD.value)
        assert a[0] == 0 and a[1] == SLOW_GAS

    def test_left_steers_negative(self):
        assert continuous_action(StudyClass.LEFT.value)[0] == -SLOW_STEER

    def test_right_steers_positive(self):
        assert continuous_action(StudyClass.RIGHT.value)[0] == SLOW_STEER

    def test_returns_copy(self):
        a = continuous_action(StudyClass.FORWARD.value)
        a[1] = 999
        assert continuous_action(StudyClass.FORWARD.value)[1] == SLOW_GAS


class TestInverseAction:
    def test_negates_steer_and_gas(self):
        inv = inverse_action(np.array([0.3, 0.15, 0.0]))
        np.testing.assert_allclose(inv, [-0.3, -0.15, 0.0], atol=1e-6)

    def test_clears_brake(self):
        assert inverse_action(np.array([0.0, 0.0, 0.8]))[2] == 0.0


class TestBaselineController:
    def test_forward_moves_forward(self):
        c = MovementController(strategy_name="baseline")
        a = c.step(StudyClass.FORWARD.value)
        assert a[1] == SLOW_GAS

    def test_records_history(self):
        c = MovementController(strategy_name="baseline", history_seconds=0.8, fps=50)
        for _ in range(10):
            c.step(StudyClass.FORWARD.value)
        assert len(c.history) == 10

    def test_history_capped(self):
        c = MovementController(strategy_name="baseline", history_seconds=0.2, fps=50)
        for _ in range(100):
            c.step(StudyClass.FORWARD.value)
        assert len(c.history) == 10

    def test_baseline_ignores_errp(self):
        c = MovementController(strategy_name="baseline")
        for _ in range(5):
            c.step(StudyClass.FORWARD.value)
        a = c.step(StudyClass.FORWARD.value, errp_prob=0.99)
        assert not c.is_reversing
        assert a[1] == SLOW_GAS


class TestErrpReversal:
    def test_reversal_replays_inverse(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.0,
                               history_seconds=0.1, fps=50, clock=clock)
        for _ in range(5):
            c.step(StudyClass.FORWARD.value)
        assert len(c.history) == 5

        a = c.step(StudyClass.REST.value, errp_prob=0.9)
        assert c.is_reversing
        np.testing.assert_array_equal(a, inverse_action(continuous_action(StudyClass.FORWARD.value)))

    def test_reversal_consumes_all_history(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.0,
                               history_seconds=0.1, fps=50, clock=clock)
        for _ in range(5):
            c.step(StudyClass.FORWARD.value)
        actions = [c.step(StudyClass.REST.value, errp_prob=0.9)]
        while c.is_reversing:
            actions.append(c.step(StudyClass.REST.value))
        reversal_actions = [a for a in actions if a[1] < 0]
        assert len(reversal_actions) == 5

    def test_errp_ignored_during_reversal(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.0,
                               history_seconds=0.2, fps=50, clock=clock)
        for _ in range(10):
            c.step(StudyClass.FORWARD.value)
        c.step(StudyClass.REST.value, errp_prob=0.9)
        assert c.correction_count == 1
        c.step(StudyClass.REST.value, errp_prob=0.9)
        assert c.correction_count == 1
        assert c.is_reversing

    def test_brakes_to_stop_before_reversing(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.0,
                               history_seconds=0.1, fps=50, clock=clock)
        for _ in range(5):
            c.step(StudyClass.FORWARD.value)
        a = c.step(StudyClass.REST.value, errp_prob=0.9, speed=10.0)
        assert a[2] == BRAKE_STRENGTH
        assert not c.is_reversing
        a = c.step(StudyClass.REST.value, speed=0.0)
        assert c.is_reversing
        assert a[1] < 0

    def test_reverse_starts_immediately_on_detection(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.1,
                               history_seconds=1.0, fps=50, clock=clock)
        for _ in range(5):
            c.step(StudyClass.FORWARD.value)
        a = c.step(StudyClass.REST.value, errp_prob=0.9)
        assert c.is_reversing
        assert a[1] < 0
        assert c.correction_count == 1

    def test_action_delay_postpones_resume_after_reversal(self):
        clock = FakeClock()
        c = MovementController(strategy_name="errp", errp_action_delay=0.1,
                               history_seconds=0.04, fps=50, clock=clock)
        for _ in range(2):
            c.step(StudyClass.FORWARD.value)
        c.step(StudyClass.REST.value, errp_prob=0.9)
        while c.is_reversing:
            c.step(StudyClass.REST.value)
        a = c.step(StudyClass.FORWARD.value)
        np.testing.assert_array_equal(a, [0, 0, 0])
        clock.t = 0.11
        a = c.step(StudyClass.FORWARD.value)
        assert a[1] == SLOW_GAS


class TestBrakingOnDirectionChange:
    def test_no_brake_on_initial_move(self):
        c = MovementController(strategy_name="baseline")
        a = c.step(StudyClass.FORWARD.value)
        assert a[2] == 0.0 and a[1] == SLOW_GAS

    def test_brake_when_changing_direction(self):
        c = MovementController(strategy_name="baseline", brake_seconds=0.1, fps=50)
        c.step(StudyClass.FORWARD.value)
        a = c.step(StudyClass.LEFT.value)
        assert a[2] == BRAKE_STRENGTH
        assert a[0] == 0.0 and a[1] == 0.0

    def test_brake_lasts_configured_frames(self):
        c = MovementController(strategy_name="baseline", brake_seconds=0.1, fps=50)
        c.step(StudyClass.FORWARD.value)
        brake_steps = 0
        for _ in range(10):
            a = c.step(StudyClass.LEFT.value)
            if a[2] == BRAKE_STRENGTH:
                brake_steps += 1
            else:
                break
        assert brake_steps == 5

    def test_resumes_new_direction_after_brake(self):
        c = MovementController(strategy_name="baseline", brake_seconds=0.1, fps=50)
        c.step(StudyClass.FORWARD.value)
        last = None
        for _ in range(10):
            last = c.step(StudyClass.LEFT.value)
        assert last[0] == -SLOW_STEER

    def test_no_brake_when_returning_to_rest_then_move(self):
        c = MovementController(strategy_name="baseline", brake_seconds=0.1, fps=50)
        c.step(StudyClass.REST.value)
        a = c.step(StudyClass.FORWARD.value)
        assert a[2] == 0.0 and a[1] == SLOW_GAS


class TestCruiseGovernor:
    def test_gas_applied_below_cruise(self):
        c = MovementController(strategy_name="baseline", cruise_speed=4.0)
        a = c.step(StudyClass.FORWARD.value, speed=1.0)
        assert a[1] == SLOW_GAS

    def test_gas_cut_at_cruise(self):
        c = MovementController(strategy_name="baseline", cruise_speed=4.0)
        a = c.step(StudyClass.FORWARD.value, speed=4.0)
        assert a[1] == 0.0

    def test_steer_unaffected_by_speed(self):
        c = MovementController(strategy_name="baseline", cruise_speed=4.0)
        a = c.step(StudyClass.LEFT.value, speed=10.0)
        assert a[0] == -SLOW_STEER


