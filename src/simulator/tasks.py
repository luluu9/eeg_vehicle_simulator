import math
from enum import Enum
from dataclasses import dataclass

from gymnasium.envs.box2d.wheelchair_dynamics import WHEELCHAIR_LENGTH, WHEELCHAIR_WIDTH


class GoalType(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    REST = "rest"


@dataclass
class Goal:
    goal_type: GoalType
    target_value: float  # degrees for turn, meters for move, seconds for rest
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
            return delta >= target_rad

        elif goal.goal_type == GoalType.TURN_RIGHT:
            delta = _normalize_angle(current.angle - start.angle)
            target_rad = -math.radians(goal.target_value)
            return delta <= target_rad

        elif goal.goal_type == GoalType.MOVE_FORWARD:
            dx = current.x - start.x
            dy = current.y - start.y
            dist = math.sqrt(dx * dx + dy * dy)
            return dist >= goal.target_value

        elif goal.goal_type == GoalType.REST:
            return elapsed >= goal.target_value

        return False

    @staticmethod
    def optimal_path_length(goal: Goal) -> float:
        if goal.goal_type == GoalType.MOVE_FORWARD:
            return goal.target_value
        if goal.goal_type in (GoalType.TURN_LEFT, GoalType.TURN_RIGHT):
            return 0.5 * math.radians(goal.target_value)
        return 0.0

    @staticmethod
    def completion_pct(goal: Goal, start: WheelchairState, current: WheelchairState, elapsed: float) -> float:
        if goal.goal_type == GoalType.TURN_LEFT:
            delta = _normalize_angle(current.angle - start.angle)
            return min(1.0, max(0.0, delta / math.radians(goal.target_value)))
        elif goal.goal_type == GoalType.TURN_RIGHT:
            delta = _normalize_angle(current.angle - start.angle)
            return min(1.0, max(0.0, -delta / math.radians(goal.target_value)))
        elif goal.goal_type == GoalType.MOVE_FORWARD:
            dx = current.x - start.x
            dy = current.y - start.y
            dist = math.sqrt(dx * dx + dy * dy)
            return min(1.0, dist / goal.target_value)
        elif goal.goal_type == GoalType.REST:
            return min(1.0, elapsed / goal.target_value) if goal.target_value > 0 else 1.0
        return 0.0

    @staticmethod
    def optimal_time(goal: Goal) -> float:
        from .controller import DEFAULT_CRUISE_SPEED, NOMINAL_TURN_RATE
        if goal.goal_type == GoalType.MOVE_FORWARD:
            return goal.target_value / DEFAULT_CRUISE_SPEED
        if goal.goal_type in (GoalType.TURN_LEFT, GoalType.TURN_RIGHT):
            return math.radians(goal.target_value) / NOMINAL_TURN_RATE
        if goal.goal_type == GoalType.REST:
            return goal.target_value
        return goal.timeout


TASK_A_GOALS = [
    Goal(GoalType.TURN_LEFT, target_value=90, timeout=60),
    Goal(GoalType.MOVE_FORWARD, target_value=5 * WHEELCHAIR_LENGTH, timeout=60),
    Goal(GoalType.REST, target_value=15, timeout=60),
    Goal(GoalType.TURN_RIGHT, target_value=90, timeout=60),
]


class TrajectoryTask:
    TIME_LIMIT = 300.0

    ROAD_HALF_WIDTH = 2 * WHEELCHAIR_WIDTH

    def __init__(self, waypoints: list[Waypoint], waypoint_radius: float = 2.0):
        self.waypoints = waypoints
        self.waypoint_radius = waypoint_radius
        self.current_idx = 0
        self._cumulative_lengths = self._compute_cumulative_lengths()
        self._max_progress: float = 0.0

    def check_waypoint(self, x: float, y: float) -> bool:
        if self.current_idx >= len(self.waypoints):
            return False
        finish = self.waypoints[-1]
        if math.sqrt((x - finish.x) ** 2 + (y - finish.y) ** 2) <= self.waypoint_radius:
            self.current_idx = len(self.waypoints)
            return True
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
        if self._cumulative_lengths:
            return self._cumulative_lengths[-1]
        return 0.0

    def _compute_cumulative_lengths(self) -> list[float]:
        lengths = [0.0]
        for i in range(1, len(self.waypoints)):
            dx = self.waypoints[i].x - self.waypoints[i - 1].x
            dy = self.waypoints[i].y - self.waypoints[i - 1].y
            lengths.append(lengths[-1] + math.sqrt(dx * dx + dy * dy))
        return lengths

    def arc_length_progress(self, x: float, y: float) -> float:
        best_s = 0.0
        min_dist_sq = float('inf')
        for i in range(len(self.waypoints) - 1):
            ax, ay = self.waypoints[i].x, self.waypoints[i].y
            bx, by = self.waypoints[i + 1].x, self.waypoints[i + 1].y
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
            px, py = ax + t * dx, ay + t * dy
            dist_sq = (x - px) ** 2 + (y - py) ** 2
            seg_len = math.sqrt(seg_len_sq)
            s = self._cumulative_lengths[i] + t * seg_len
            if dist_sq < min_dist_sq - 1e-9:
                min_dist_sq = dist_sq
                best_s = s
            elif abs(dist_sq - min_dist_sq) < 1e-9 and s > best_s:
                best_s = s
        progress = best_s / self.total_path_length if self.total_path_length > 0 else 1.0
        self._max_progress = max(self._max_progress, progress)
        return self._max_progress

    def is_on_road(self, x: float, y: float) -> bool:
        min_dist = float('inf')
        for i in range(len(self.waypoints) - 1):
            ax, ay = self.waypoints[i].x, self.waypoints[i].y
            bx, by = self.waypoints[i + 1].x, self.waypoints[i + 1].y
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-9:
                continue
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
            px, py = ax + t * dx, ay + t * dy
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if dist < min_dist:
                min_dist = dist
        return min_dist <= self.ROAD_HALF_WIDTH

    @property
    def optimal_time(self) -> float:
        from .controller import DEFAULT_CRUISE_SPEED
        if DEFAULT_CRUISE_SPEED <= 0:
            return self.TIME_LIMIT
        return self.total_path_length / DEFAULT_CRUISE_SPEED

    def reset(self):
        self.current_idx = 0
        self._max_progress = 0.0


def create_default_trajectory() -> TrajectoryTask:
    step = WHEELCHAIR_LENGTH  # 4.0 Box2D units = 1 m real-world
    n = 5                     # 5 waypoints per segment = 5 m per segment

    end_forward = n * step    # 20.0  (end of segment 1)
    end_right   = n * step    # 20.0  (width of segment 2)

    waypoints = [Waypoint(0.0, 0.0)]  # start at car spawn position (immediately cleared)
    # Segment 1: 5 m forward (north, +y) - requires FORWARD MI
    for i in range(1, n + 1):
        waypoints.append(Waypoint(0.0, i * step))
    # Segment 2: 5 m to the right (east, +x) - requires RIGHT MI then FORWARD
    for i in range(1, n + 1):
        waypoints.append(Waypoint(i * step, end_forward))
    # Segment 3: 5 m to the left (turn left from east → face north, +y) - requires LEFT MI then FORWARD
    for i in range(1, n + 1):
        waypoints.append(Waypoint(end_right, end_forward + i * step))
    return TrajectoryTask(waypoints)
