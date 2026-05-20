import json
import os
import sys
import time
import gymnasium as gym
import pygame

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.simulator.evaluation import EvaluationSession, run_intermission
from src.simulator.input_handler import MultiStreamMonitor
from src.simulator.launch_window import LaunchWindow
from src.simulator.sequencer import ExperimentSequencer


def _results_path(subject_id: str, experiment) -> str:
    ts = time.strftime("%Y_%m_%d_%H-%M-%S")
    filename = f"{experiment.task_id}_{experiment.task}_{experiment.strategy}_{ts}.json"
    folder = os.path.join("results", subject_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def main():
    app = QApplication(sys.argv)
    monitor = MultiStreamMonitor()
    monitor.start()
    sequencer = ExperimentSequencer()

    subject_id = ""
    mi_stream = ""
    errp_stream = ""

    while not sequencer.all_done:
        window = LaunchWindow(monitor, sequencer, subject_id, mi_stream, errp_stream)
        res = window.exec_and_get()
        if res is None:
            break
        subject_id, mi_stream, errp_stream = res

        exp = sequencer.current
        if exp is None:
            break

        env = gym.make("WheelchairRacing-v0", render_mode="human", max_episode_steps=10_000_000)
        env.reset()
        screen = pygame.display.get_surface()

        run_intermission(env, screen, exp)

        session = EvaluationSession(
            strategy_name=exp.strategy,
            task_name=exp.task,
            mi_channel=mi_stream,
            errp_channel=errp_stream,
            subject_id=subject_id,
        )
        result = session.run(env, screen)

        env.close()
        pygame.quit()

        path = _results_path(subject_id, exp)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {path}")

        sequencer.advance("completed", result)

    monitor.stop()

    if sequencer.all_done:
        QMessageBox.information(None, "Done", "All experiments completed.")


if __name__ == "__main__":
    main()
