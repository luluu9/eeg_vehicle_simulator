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
from gymnasium.envs.box2d.wheelchair_racing import (
    WHEELCHAIR_CENTER_X_RATIO, WHEELCHAIR_CENTER_Y_RATIO,
    WINDOW_W, WINDOW_H, ZOOM, SCALE,
)


pygame_flip_original = None

_CUE_RADIUS = 60
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
    sw, sh = screen.get_size()
    cx, cy = int(sw * WHEELCHAIR_CENTER_X_RATIO), int(sh * WHEELCHAIR_CENTER_Y_RATIO)
    r = max(20, int(_CUE_RADIUS * sh / WINDOW_H))
    delta = _normalize_angle(current_state.angle - start_state.angle)
    if goal.goal_type == GoalType.TURN_LEFT:
        target = math.radians(goal.target_value)
    else:
        target = -math.radians(goal.target_value)
    remaining = target - delta
    angle = -remaining

    pygame.draw.circle(screen, _COLOR_OUTLINE, (cx, cy), r, 2)

    shaft_len = r * 1.0

    ref_tip_x, ref_tip_y = cx, cy - shaft_len
    ref_base_x, ref_base_y = cx, cy - shaft_len + shaft_len * 0.2
    pygame.draw.line(screen, (0, 200, 0),
                     (int(ref_base_x), int(ref_base_y)),
                     (int(ref_tip_x), int(ref_tip_y)), 3)

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
    sw, sh = screen.get_size()
    cx, cy = int(sw * WHEELCHAIR_CENTER_X_RATIO), int(sh * WHEELCHAIR_CENTER_Y_RATIO)
    r = max(20, int(_CUE_RADIUS * sh / WINDOW_H))
    pygame.draw.circle(screen, _COLOR_REST, (cx, cy), r // 2, 5)


def _world_to_screen(wx, wy, car_x, car_y, car_angle, screen):
    sw, sh = screen.get_size()
    zoom_x = ZOOM * SCALE * sw / WINDOW_W
    zoom_y = ZOOM * SCALE * sh / WINDOW_H
    cam_angle = -car_angle
    v = pygame.math.Vector2(wx - car_x, wy - car_y).rotate_rad(cam_angle)
    sx = v[0] * zoom_x + sw / 2
    sy_pre = v[1] * zoom_y + sh / 4
    return int(sx), sh - int(sy_pre)


def _draw_forward_line(screen, goal, start_state, current_state):
    sw, sh = screen.get_size()
    zoom_x = ZOOM * SCALE * sw / WINDOW_W
    radius_px = int(goal.target_value * zoom_x)
    center = _world_to_screen(
        start_state.x, start_state.y,
        current_state.x, current_state.y, current_state.angle, screen,
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

# ── Intermission ─────────────────────────────────────────────────────────────

_COUNTDOWN_FROM = 5
_COUNTDOWN_COLOR = (0, 100, 255)
_TEXT_COLOR = (255, 255, 255)
_BAR_COLOR = (0, 0, 0, 160)


def _draw_top_bar(screen: pygame.Surface, text: str):
    sw, sh = screen.get_size()
    font = pygame.font.SysFont("Arial", max(18, sh // 30))
    rendered = font.render(text, True, _TEXT_COLOR)
    bar_h = rendered.get_height() + 20
    bar = pygame.Surface((sw, bar_h), pygame.SRCALPHA)
    bar.fill(_BAR_COLOR)
    screen.blit(bar, (0, 0))
    screen.blit(rendered, (sw // 2 - rendered.get_width() // 2, 10))


def _draw_countdown_digit(screen: pygame.Surface, digit: int):
    sw, sh = screen.get_size()
    cx = int(sw * WHEELCHAIR_CENTER_X_RATIO)
    cy = int(sh * WHEELCHAIR_CENTER_Y_RATIO)
    font = pygame.font.SysFont("Arial", max(48, sh // 8), bold=True)
    rendered = font.render(str(digit), True, _COUNTDOWN_COLOR)
    rendered.set_alpha(140)
    screen.blit(rendered, (cx - rendered.get_width() // 2, cy - rendered.get_height() // 2))


def run_intermission(env, screen: pygame.Surface, experiment):
    _patch_pygame_flip()
    env.step(REST_ACTION.copy())

    info_text = f"Task {experiment.task}  ·  {experiment.strategy.capitalize()}  (#{experiment.task_id})"

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False

        _draw_top_bar(screen, f"{info_text}    Press SPACE to begin")
        _flip()
        pygame.time.wait(16)


class EvaluationSession:
    def __init__(self, strategy_name: str, task_name: str,
                 mi_channel: str, errp_channel: str | None, subject_id: str):
        self.strategy_name = strategy_name
        self.task_name = task_name.upper()
        self.mi_channel = mi_channel
        self.errp_channel = errp_channel
        self.subject_id = subject_id
        self.metrics = MetricsCollector()
        self.monitor = MultiStreamMonitor()

    def run(self, env=None, screen=None) -> dict:
        _patch_pygame_flip()
        self.monitor.start()

        owned_env = env is None
        if owned_env:
            env = gym.make("WheelchairRacing-v0", render_mode="human", max_episode_steps=10_000_000)
            env.reset()
            screen = pygame.display.get_surface()

        strategy = STUDY_STRATEGIES[self.strategy_name]()

        try:
            if self.task_name == "A":
                self._run_task_a(env, strategy, screen)
            elif self.task_name == "B":
                self._run_task_b(env, strategy, screen)
        finally:
            self.monitor.stop()
            if owned_env:
                env.close()
                pygame.quit()

        result = self.metrics.summary()
        result.update({
            "subject_id": self.subject_id,
            "task": self.task_name,
            "strategy": self.strategy_name,
            "timestamp": time.strftime("%Y_%m_%d_%H:%M:%S"),
        })
        return result

    def _run_task_a(self, env, strategy, screen):
        clock = pygame.time.Clock()
        goals = list(TASK_A_GOALS)
        random.shuffle(goals)

        for i, goal in enumerate(goals):
            env.reset()
            start_state = get_wheelchair_state(env)
            if not self._goal_countdown(env, screen, goal, i, start_state):
                return
            self.metrics.start_trial()
            elapsed = 0.0
            completed = False
            running = True
            clock.tick()  # discard time accumulated during countdown

            while running:
                dt = clock.tick(60) / 1000.0
                elapsed += dt

                if not self._handle_events():
                    return

                mi_probs = self.monitor.get_probabilities()
                errp_data = self._pick_errp(self.monitor.get_errp())
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
            errp_data = self._pick_errp(self.monitor.get_errp())
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
        if self.mi_channel in mi_probs:
            return self.mi_channel
        return None

    def _pick_errp(self, errp_data: dict) -> dict:
        if self.errp_channel and self.errp_channel in errp_data:
            return {self.errp_channel: errp_data[self.errp_channel]}
        return {}

    @staticmethod
    def _goal_countdown(env, screen, goal, trial_idx: int, start_state) -> bool:
        for count in range(_COUNTDOWN_FROM, 0, -1):
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return False
                env.step(REST_ACTION.copy())
                state = get_wheelchair_state(env)
                if goal.goal_type in (GoalType.TURN_LEFT, GoalType.TURN_RIGHT):
                    _draw_compass_arrow(screen, goal, start_state, state)
                elif goal.goal_type == GoalType.REST:
                    _draw_rest_circle(screen)
                elif goal.goal_type == GoalType.MOVE_FORWARD:
                    _draw_forward_line(screen, goal, start_state, state)
                _draw_countdown_digit(screen, count)
                _flip()
                pygame.time.wait(16)
        return True

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
