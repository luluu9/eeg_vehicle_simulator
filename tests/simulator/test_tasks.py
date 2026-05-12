import math
import pytest
from src.simulator.tasks import (
    GoalType, Goal, GoalChecker, WheelchairState,
    TrajectoryTask, Waypoint, create_default_trajectory,
    TASK_A_GOALS, _normalize_angle,
)


class TestNormalizeAngle:
    def test_zero(self):
        assert _normalize_angle(0) == 0

    def test_positive_wrap(self):
        assert abs(_normalize_angle(3 * math.pi) - math.pi) < 0.01

    def test_negative_wrap(self):
        assert abs(_normalize_angle(-3 * math.pi) - (-math.pi)) < 0.01


class TestGoalChecker:
    def _state(self, x=0, y=0, angle=0, speed=0):
        return WheelchairState(x, y, angle, speed)

    def test_turn_left_completed(self):
        start = self._state(angle=0)
        current = self._state(angle=math.radians(90))
        goal = Goal(GoalType.TURN_LEFT, 90, 15)
        assert GoalChecker.check(goal, start, current, 5.0)

    def test_turn_left_within_tolerance(self):
        start = self._state(angle=0)
        current = self._state(angle=math.radians(80))
        goal = Goal(GoalType.TURN_LEFT, 90, 15)
        assert GoalChecker.check(goal, start, current, 5.0)

    def test_turn_left_not_completed(self):
        start = self._state(angle=0)
        current = self._state(angle=math.radians(30))
        goal = Goal(GoalType.TURN_LEFT, 90, 15)
        assert not GoalChecker.check(goal, start, current, 5.0)

    def test_turn_right_completed(self):
        start = self._state(angle=0)
        current = self._state(angle=math.radians(-90))
        goal = Goal(GoalType.TURN_RIGHT, 90, 15)
        assert GoalChecker.check(goal, start, current, 5.0)

    def test_move_forward_completed(self):
        start = self._state(x=0, y=0)
        current = self._state(x=3, y=4)
        goal = Goal(GoalType.MOVE_FORWARD, 5, 0.5)
        assert GoalChecker.check(goal, start, current, 5.0)

    def test_move_forward_not_enough(self):
        start = self._state(x=0, y=0)
        current = self._state(x=1, y=1)
        goal = Goal(GoalType.MOVE_FORWARD, 5, 0.5)
        assert not GoalChecker.check(goal, start, current, 5.0)

    def test_rest_completed(self):
        start = self._state(x=0, y=0)
        current = self._state(x=0.1, y=0.1)
        goal = Goal(GoalType.REST, 5, 0.3)
        assert GoalChecker.check(goal, start, current, 6.0)

    def test_rest_not_enough_time(self):
        start = self._state(x=0, y=0)
        current = self._state(x=0, y=0)
        goal = Goal(GoalType.REST, 5, 0.3)
        assert not GoalChecker.check(goal, start, current, 3.0)

    def test_rest_moved_too_far(self):
        start = self._state(x=0, y=0)
        current = self._state(x=2, y=2)
        goal = Goal(GoalType.REST, 5, 0.3)
        assert not GoalChecker.check(goal, start, current, 10.0)


class TestOptimalPathLength:
    def test_forward(self):
        goal = Goal(GoalType.MOVE_FORWARD, 5, 0.5)
        assert GoalChecker.optimal_path_length(goal) == 5.0

    def test_rest(self):
        goal = Goal(GoalType.REST, 5, 0.3)
        assert GoalChecker.optimal_path_length(goal) == 0.0

    def test_turn(self):
        goal = Goal(GoalType.TURN_LEFT, 90, 15)
        length = GoalChecker.optimal_path_length(goal)
        assert length > 0


class TestTaskAGoals:
    def test_has_four_goals(self):
        assert len(TASK_A_GOALS) == 4

    def test_all_have_timeout(self):
        for goal in TASK_A_GOALS:
            assert goal.timeout > 0


class TestTrajectoryTask:
    def test_check_waypoint(self):
        wps = [Waypoint(0, 5), Waypoint(0, 10)]
        task = TrajectoryTask(wps, waypoint_radius=2.0)
        assert task.check_waypoint(0, 4.5)
        assert task.current_idx == 1
        assert not task.completed

    def test_completed(self):
        wps = [Waypoint(0, 5)]
        task = TrajectoryTask(wps, waypoint_radius=2.0)
        task.check_waypoint(0, 5)
        assert task.completed

    def test_progress(self):
        wps = [Waypoint(0, 5), Waypoint(0, 10), Waypoint(0, 15), Waypoint(0, 20)]
        task = TrajectoryTask(wps, waypoint_radius=2.0)
        assert task.progress == 0.0
        task.check_waypoint(0, 5)
        assert task.progress == 0.25
        task.check_waypoint(0, 10)
        assert task.progress == 0.5

    def test_deviation(self):
        wps = [Waypoint(0, 10)]
        task = TrajectoryTask(wps, waypoint_radius=2.0)
        dev = task.get_deviation(3, 6)
        assert dev == 5.0

    def test_total_path_length(self):
        wps = [Waypoint(0, 0), Waypoint(0, 10), Waypoint(10, 10)]
        task = TrajectoryTask(wps)
        assert abs(task.total_path_length - 20.0) < 0.01

    def test_reset(self):
        wps = [Waypoint(0, 5), Waypoint(0, 10)]
        task = TrajectoryTask(wps, waypoint_radius=2.0)
        task.check_waypoint(0, 5)
        task.reset()
        assert task.current_idx == 0
        assert not task.completed


class TestDefaultTrajectory:
    def test_creates_waypoints(self):
        t = create_default_trajectory()
        assert len(t.waypoints) > 0

    def test_forms_closed_loop(self):
        t = create_default_trajectory()
        first = t.waypoints[0]
        last = t.waypoints[-1]
        dist = math.sqrt((last.x - 0) ** 2 + (last.y - 0) ** 2)
        assert dist < 10  # ends near origin
