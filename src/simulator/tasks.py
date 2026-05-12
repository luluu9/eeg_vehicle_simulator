import math
from enum import Enum
from dataclasses import dataclass


class GoalType(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    REST = "rest"


@dataclass
class Goal:
    goal_type: GoalType
    target_value: float  # degrees for turn, meters for move, seconds for rest
    tolerance: float     # degrees or meters
    timeout: float = 60.0


@dataclass
class WheelchairState:
    x: float
    y: float
    angle: float  # radians
    speed: float


@dataclass
class Waypoint:
    x: float
    y: float


def get_wheelchair_state(env) -> WheelchairState:
    car = env.unwrapped.car
    pos = car.hull.position
    vel = car.hull.linearVelocity
    return WheelchairState(
        x=float(pos[0]),
        y=float(pos[1]),
        angle=float(car.hull.angle),
        speed=float(vel.length),
    )


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class GoalChecker:
    @staticmethod
    def check(goal: Goal, start: WheelchairState, current: WheelchairState, elapsed: float) -> bool:
        if goal.goal_type == GoalType.TURN_LEFT:
            delta = _normalize_angle(current.angle - start.angle)
            target_rad = math.radians(goal.target_value)
            return abs(delta - target_rad) <= math.radians(goal.tolerance)

        elif goal.goal_type == GoalType.TURN_RIGHT:
            delta = _normalize_angle(current.angle - start.angle)
            target_rad = -math.radians(goal.target_value)
            return abs(delta - target_rad) <= math.radians(goal.tolerance)

        elif goal.goal_type == GoalType.MOVE_FORWARD:
            dx = current.x - start.x
            dy = current.y - start.y
            dist = math.sqrt(dx * dx + dy * dy)
            return dist >= goal.target_value - goal.tolerance

        elif goal.goal_type == GoalType.REST:
            dx = current.x - start.x
            dy = current.y - start.y
            dist = math.sqrt(dx * dx + dy * dy)
            return dist <= goal.tolerance and elapsed >= goal.target_value

        return False

    @staticmethod
    def optimal_path_length(goal: Goal) -> float:
        if goal.goal_type == GoalType.MOVE_FORWARD:
            return goal.target_value
        if goal.goal_type in (GoalType.TURN_LEFT, GoalType.TURN_RIGHT):
            return 0.5 * math.radians(goal.target_value)
        return 0.0


TASK_A_GOALS = [
    Goal(GoalType.TURN_LEFT, target_value=90, tolerance=15, timeout=60),
    Goal(GoalType.MOVE_FORWARD, target_value=5, tolerance=0.5, timeout=60),
    Goal(GoalType.REST, target_value=5, tolerance=0.3, timeout=15),
    Goal(GoalType.TURN_RIGHT, target_value=90, tolerance=15, timeout=60),
]


class TrajectoryTask:
    TIME_LIMIT = 300.0

    def __init__(self, waypoints: list[Waypoint], waypoint_radius: float = 2.0):
        self.waypoints = waypoints
        self.waypoint_radius = waypoint_radius
        self.current_idx = 0

    def check_waypoint(self, x: float, y: float) -> bool:
        if self.current_idx >= len(self.waypoints):
            return False
        wp = self.waypoints[self.current_idx]
        dist = math.sqrt((x - wp.x) ** 2 + (y - wp.y) ** 2)
        if dist <= self.waypoint_radius:
            self.current_idx += 1
            return True
        return False

    @property
    def completed(self) -> bool:
        return self.current_idx >= len(self.waypoints)

    @property
    def progress(self) -> float:
        return self.current_idx / len(self.waypoints) if self.waypoints else 1.0

    def get_deviation(self, x: float, y: float) -> float:
        if self.current_idx >= len(self.waypoints):
            return 0.0
        wp = self.waypoints[self.current_idx]
        return math.sqrt((x - wp.x) ** 2 + (y - wp.y) ** 2)

    @property
    def total_path_length(self) -> float:
        total = 0.0
        for i in range(1, len(self.waypoints)):
            dx = self.waypoints[i].x - self.waypoints[i - 1].x
            dy = self.waypoints[i].y - self.waypoints[i - 1].y
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def reset(self):
        self.current_idx = 0


def create_default_trajectory() -> TrajectoryTask:
    spacing = 5.0
    side = 30.0
    n = int(side / spacing)
    waypoints = []
    for i in range(1, n + 1):
        waypoints.append(Waypoint(0, i * spacing))
    for i in range(1, n + 1):
        waypoints.append(Waypoint(-i * spacing, side))
    for i in range(1, n + 1):
        waypoints.append(Waypoint(-side, side - i * spacing))
    for i in range(1, n + 1):
        waypoints.append(Waypoint(-side + i * spacing, 0))
    return TrajectoryTask(waypoints)
