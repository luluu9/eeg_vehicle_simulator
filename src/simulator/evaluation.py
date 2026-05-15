import math
import time
import numpy as np
import gymnasium as gym

try:
    import pygame
except ImportError:
    pygame = None

from .input_handler import MultiStreamMonitor
from .strategies import STUDY_STRATEGIES, REST_ACTION
from .metrics import MetricsCollector
from .tasks import (
    TASK_A_GOALS, GoalChecker, GoalType, TrajectoryTask, Waypoint,
    get_wheelchair_state, create_default_trajectory, _normalize_angle,
)


pygame_flip_original = None

_CUE_RADIUS = 60
_CUE_CX = 960
_CUE_CY = 810
_COLOR_CUE = (255, 255, 0)
_COLOR_REST = (255, 255, 0)
_COLOR_OUTLINE = (0, 100, 255)


def _patch_pygame_flip():
    global pygame_flip_original
    if pygame is not None and pygame_flip_original is None:
        pygame_flip_original = pygame.display.flip
        pygame.display.flip = lambda: None


def _flip():
    if pygame_flip_original is not None:
        pygame_flip_original()


def _draw_compass_arrow(screen, goal, start_state, current_state):
    cx, cy, r = _CUE_CX, _CUE_CY, _CUE_RADIUS
    delta = _normalize_angle(current_state.angle - start_state.angle)
    if goal.goal_type == GoalType.TURN_LEFT:
        target = math.radians(goal.target_value)
    else:
        target = -math.radians(goal.target_value)
    remaining = target - delta
    angle = -remaining

    pygame.draw.circle(screen, _COLOR_OUTLINE, (cx, cy), r, 2)

    shaft_len = r * 1.0
    tip_x = cx + shaft_len * math.sin(angle)
    tip_y = cy - shaft_len * math.cos(angle)
    pygame.draw.line(screen, _COLOR_CUE, (cx, cy), (int(tip_x), int(tip_y)), 3)

    head_len = r * 0.3
    spread = 0.45
    lx = tip_x - head_len * math.sin(angle - spread)
    ly = tip_y + head_len * math.cos(angle - spread)
    rx = tip_x - head_len * math.sin(angle + spread)
    ry = tip_y + head_len * math.cos(angle + spread)
    pygame.draw.polygon(screen, _COLOR_CUE, [
        (int(tip_x), int(tip_y)), (int(lx), int(ly)), (int(rx), int(ry)),
    ])


def _draw_rest_circle(screen):
    cx, cy, r = _CUE_CX, _CUE_CY, _CUE_RADIUS
    pygame.draw.circle(screen, _COLOR_REST, (cx, cy), r, 3)
    pygame.draw.circle(screen, _COLOR_REST, (cx, cy), r // 3, 2)


def _draw_forward_line(screen, goal, start_state, current_state):
    cx, cy, r = _CUE_CX, _CUE_CY, _CUE_RADIUS
    dx = current_state.x - start_state.x
    dy = current_state.y - start_state.y
    dist = math.sqrt(dx * dx + dy * dy)
    progress = min(dist / goal.target_value, 1.0)

    pygame.draw.circle(screen, _COLOR_OUTLINE, (cx, cy), r, 2)

    goal_y = int(cy - r * 0.7)
    hw = int(math.sqrt(max(r * r - (goal_y - cy) ** 2, 0)))
    pygame.draw.line(screen, _COLOR_CUE, (cx - hw, goal_y), (cx + hw, goal_y), 3)

    start_y = cy + int(r * 0.5)
    dot_y = int(start_y - (start_y - goal_y) * progress)
    pygame.draw.circle(screen, _COLOR_CUE, (cx, dot_y), 6)


class EvaluationSession:
    def __init__(self, strategy_name: str, task_name: str):
        self.strategy_name = strategy_name
        self.task_name = task_name.upper()
        self.metrics = MetricsCollector()
        self.monitor = MultiStreamMonitor()

    def run(self) -> dict:
        _patch_pygame_flip()
        self.monitor.start()

        env = gym.make("WheelchairRacing-v0", render_mode="human", max_episode_steps=10_000_000)
        strategy = STUDY_STRATEGIES[self.strategy_name]()
        env.reset()
        screen = pygame.display.get_surface()

        try:
            if self.task_name == "A":
                self._run_task_a(env, strategy, screen)
            elif self.task_name == "B":
                self._run_task_b(env, strategy, screen)
        finally:
            self.monitor.stop()
            env.close()
            pygame.quit()

        return self.metrics.summary()

    def _run_task_a(self, env, strategy, screen):
        clock = pygame.time.Clock()

        for i, goal in enumerate(TASK_A_GOALS):
            env.reset()
            start_state = get_wheelchair_state(env)
            self.metrics.start_trial()
            elapsed = 0.0
            completed = False
            running = True

            while running:
                dt = clock.tick(60) / 1000.0
                elapsed += dt

                if not self._handle_events():
                    return

                mi_probs = self.monitor.get_probabilities()
                errp_data = self.monitor.get_errp()
                stream = self._pick_stream(mi_probs)
                action = strategy.compute(mi_probs, stream, errp_data) if stream else REST_ACTION.copy()

                env.step(action)
                state = get_wheelchair_state(env)
                self.metrics.record_position(state.x, state.y)

                if hasattr(strategy, 'correction_count'):
                    while self.metrics._correction_count < strategy.correction_count:
                        self.metrics.record_correction()

                if GoalChecker.check(goal, start_state, state, elapsed):
                    completed = True
                    running = False
                if elapsed >= goal.timeout:
                    running = False

                self._render_goal_cue(screen, goal, i, elapsed, start_state, state)
                _flip()

            optimal = GoalChecker.optimal_path_length(goal)
            self.metrics.end_trial(goal.goal_type.value, completed, optimal)

    def _run_task_b(self, env, strategy, screen):
        clock = pygame.time.Clock()
        trajectory = create_default_trajectory()
        env.reset()
        self.metrics.start_trial()
        elapsed = 0.0

        while elapsed < trajectory.TIME_LIMIT and not trajectory.completed:
            dt = clock.tick(60) / 1000.0
            elapsed += dt

            if not self._handle_events():
                break

            mi_probs = self.monitor.get_probabilities()
            errp_data = self.monitor.get_errp()
            stream = self._pick_stream(mi_probs)
            action = strategy.compute(mi_probs, stream, errp_data) if stream else REST_ACTION.copy()

            env.step(action)
            state = get_wheelchair_state(env)
            self.metrics.record_position(state.x, state.y)
            trajectory.check_waypoint(state.x, state.y)

            if hasattr(strategy, 'correction_count'):
                while self.metrics._correction_count < strategy.correction_count:
                    self.metrics.record_correction()

            self._render_trajectory_overlay(screen, trajectory, elapsed)
            _flip()

        self.metrics.end_trial("trajectory", trajectory.completed, trajectory.total_path_length)

    def _pick_stream(self, mi_probs: dict) -> str | None:
        if not mi_probs:
            return None
        return next(iter(mi_probs))

    @staticmethod
    def _handle_events() -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    @staticmethod
    def _render_goal_cue(screen, goal, trial_idx, elapsed, start_state, current_state):
        if goal.goal_type in (GoalType.TURN_LEFT, GoalType.TURN_RIGHT):
            _draw_compass_arrow(screen, goal, start_state, current_state)
        elif goal.goal_type == GoalType.REST:
            _draw_rest_circle(screen)
        elif goal.goal_type == GoalType.MOVE_FORWARD:
            _draw_forward_line(screen, goal, start_state, current_state)

        font = pygame.font.SysFont("Arial", 24)
        text = font.render(
            f"{trial_idx + 1}/{len(TASK_A_GOALS)}  |  {elapsed:.0f}s",
            True, (180, 180, 180),
        )
        screen.blit(text, (20, 20))

    @staticmethod
    def _render_trajectory_overlay(screen, trajectory, elapsed):
        font = pygame.font.SysFont("Arial", 32)
        lines = [
            f"Trajectory:  {trajectory.progress:.0%}  ({trajectory.current_idx}/{len(trajectory.waypoints)})",
            f"Time: {elapsed:.1f}s / {trajectory.TIME_LIMIT:.0f}s",
        ]
        y = 20
        for line in lines:
            surf = font.render(line, True, (255, 255, 255))
            screen.blit(surf, (20, y))
            y += 40
