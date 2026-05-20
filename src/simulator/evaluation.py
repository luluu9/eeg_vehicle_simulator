import math
import random
import time
import numpy as np
import gymnasium as gym
import pygame

from .input_handler import MultiStreamMonitor
from .strategies import STUDY_STRATEGIES, REST_ACTION
from .metrics import MetricsCollector
from .tasks import (
    TASK_A_GOALS, GoalChecker, GoalType, TrajectoryTask, Waypoint,
    get_wheelchair_state, create_default_trajectory, _normalize_angle,
)
from gymnasium.envs.box2d.wheelchair_dynamics import WHEELCHAIR_WIDTH


pygame_flip_original = None

_CUE_RADIUS = 60
_CUE_CX = 960
_CUE_CY = 810
_COLOR_CUE = (255, 255, 0)
_COLOR_REST = (255, 255, 0)
_COLOR_OUTLINE = (0, 100, 255)

_WORLD_ZOOM = 8 * 6.0
_WINDOW_W = 1920
_WINDOW_H = 1080


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
    pygame.draw.circle(screen, _COLOR_REST, (cx, cy), r // 2, 5)


def _world_to_screen(wx, wy, car_x, car_y, car_angle):
    cam_angle = -car_angle
    v = pygame.math.Vector2(wx - car_x, wy - car_y).rotate_rad(cam_angle)
    sx = v[0] * _WORLD_ZOOM + _WINDOW_W / 2
    sy = v[1] * _WORLD_ZOOM + _WINDOW_H / 4
    return int(sx), _WINDOW_H - int(sy)


def _draw_forward_line(screen, goal, start_state, current_state):
    radius_world = goal.target_value
    radius_px = int(radius_world * _WORLD_ZOOM)
    center = _world_to_screen(
        start_state.x, start_state.y,
        current_state.x, current_state.y, current_state.angle,
    )
    pygame.draw.circle(screen, _COLOR_CUE, center, radius_px, 3)


_PATH_COLOR = (80, 80, 80)
_PATH_EDGE_COLOR = (140, 140, 140)
_PATH_WIDTH = 2 * WHEELCHAIR_WIDTH


def _world_to_surf(wx, wy, zoom, translation, angle):
    v = pygame.math.Vector2(wx, wy).rotate_rad(angle)
    return (v[0] * zoom + translation[0], v[1] * zoom + translation[1])


def _draw_path_on_surf(surf, trajectory, zoom, translation, angle):
    wps = trajectory.waypoints
    if len(wps) < 2:
        return

    half_w = _PATH_WIDTH * zoom / 2

    for i in range(len(wps) - 1):
        p1 = _world_to_surf(wps[i].x, wps[i].y, zoom, translation, angle)
        p2 = _world_to_surf(wps[i + 1].x, wps[i + 1].y, zoom, translation, angle)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            continue
        nx = -dy / length * half_w
        ny = dx / length * half_w

        quad = [
            (p1[0] + nx, p1[1] + ny),
            (p2[0] + nx, p2[1] + ny),
            (p2[0] - nx, p2[1] - ny),
            (p1[0] - nx, p1[1] - ny),
        ]

        color = _PATH_COLOR if i < trajectory.current_idx else _PATH_EDGE_COLOR
        pygame.draw.polygon(surf, color, quad)


def _install_path_renderer(env, trajectory):
    car = env.unwrapped.car
    original_draw = car.draw.__func__ if hasattr(car.draw, '__func__') else None

    def patched_draw(self, surf, zoom, trans, angle, draw_particles=True):
        _draw_path_on_surf(surf, trajectory, zoom, trans, angle)
        if original_draw:
            original_draw(self, surf, zoom, trans, angle, draw_particles)
        else:
            type(car).draw(self, surf, zoom, trans, angle, draw_particles)

    import types
    car.draw = types.MethodType(patched_draw, car)


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
        goals = list(TASK_A_GOALS)
        random.shuffle(goals)

        for i, goal in enumerate(goals):
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
        _install_path_renderer(env, trajectory)
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
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(
            f"{trajectory.progress:.0%}  |  {elapsed:.0f}s",
            True, (180, 180, 180),
        )
        screen.blit(text, (20, 20))
