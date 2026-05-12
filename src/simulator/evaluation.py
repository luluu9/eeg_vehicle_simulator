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
    TASK_A_GOALS, GoalChecker, TrajectoryTask, Waypoint,
    get_wheelchair_state, create_default_trajectory,
)


pygame_flip_original = None


def _patch_pygame_flip():
    global pygame_flip_original
    if pygame is not None and pygame_flip_original is None:
        pygame_flip_original = pygame.display.flip
        pygame.display.flip = lambda: None


def _flip():
    if pygame_flip_original is not None:
        pygame_flip_original()


GOAL_LABELS = {
    "turn_left": "TURN LEFT 90°",
    "turn_right": "TURN RIGHT 90°",
    "move_forward": "MOVE FORWARD 5m",
    "rest": "REST (stay still)",
}


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

                self._render_overlay(screen, goal, i, elapsed)
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
    def _render_overlay(screen, goal, trial_idx, elapsed):
        font = pygame.font.SysFont("Arial", 32)
        label = GOAL_LABELS.get(goal.goal_type.value, goal.goal_type.value)
        lines = [
            f"Trial {trial_idx + 1}/{len(TASK_A_GOALS)}:  {label}",
            f"Time: {elapsed:.1f}s / {goal.timeout:.0f}s",
        ]
        y = 20
        for line in lines:
            surf = font.render(line, True, (255, 255, 255))
            screen.blit(surf, (20, y))
            y += 40

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
