import numpy as np
import gymnasium as gym
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout

from .config import TaskType


ACTION_MAP = {
    TaskType.LEFT_HAND: np.array([-0.5, 0.3, 0.0], dtype=np.float32),
    TaskType.RIGHT_HAND: np.array([0.5, 0.3, 0.0], dtype=np.float32),
    TaskType.FORWARD: np.array([0.0, 0.3, 0.0], dtype=np.float32),
    TaskType.REST: np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

CUE_SYMBOLS = {
    TaskType.LEFT_HAND: "←",
    TaskType.RIGHT_HAND: "→",
    TaskType.FORWARD: "↑",
    TaskType.REST: "○",
}


class WheelchairStimulus:
    WARMUP_STEPS = 50

    def __init__(self):
        self.env = gym.make("WheelchairRacing-v0", render_mode="rgb_array")
        self.env.reset()
        self._warmup()
        self._current_frame = self.env.render()

    def _warmup(self):
        noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(self.WARMUP_STEPS):
            self.env.step(noop)

    def get_frame(self) -> np.ndarray:
        return self._current_frame

    def step(self, action: np.ndarray) -> np.ndarray:
        _, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.env.reset()
        self._current_frame = self.env.render()
        return self._current_frame

    def reset(self):
        self.env.reset()
        self._warmup()
        self._current_frame = self.env.render()

    def close(self):
        self.env.close()


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


def draw_cue_overlay(pixmap: QPixmap, task: TaskType) -> QPixmap:
    result = pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    symbol = CUE_SYMBOLS[task]
    font = QFont("Arial", 72, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QPen(QColor(255, 255, 0), 3))

    rect = result.rect()
    painter.drawText(rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, symbol)
    painter.end()
    return result


def draw_feedback_border(pixmap: QPixmap, is_correct: bool) -> QPixmap:
    result = pixmap.copy()
    painter = QPainter(result)
    color = QColor(0, 200, 0) if is_correct else QColor(200, 0, 0)
    pen = QPen(color, 12)
    painter.setPen(pen)
    painter.drawRect(result.rect().adjusted(6, 6, -6, -6))
    painter.end()
    return result


class StimulusWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainBot Stimulus")
        self.setStyleSheet("background-color: black;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._stimulus = WheelchairStimulus()
        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_action = None
        self._anim_is_correct = None
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._blink_step)
        self._blink_visible = True
        self._blink_task = None

    def show_idle(self):
        self._stop_animations()
        frame = self._stimulus.get_frame()
        self._set_pixmap(frame_to_pixmap(frame))

    def show_cue(self, task: TaskType):
        self._stop_animations()
        frame = self._stimulus.get_frame()
        pixmap = draw_cue_overlay(frame_to_pixmap(frame), task)
        self._set_pixmap(pixmap)

    def show_imagery(self, task: TaskType):
        self._stop_animations()
        frame = self._stimulus.get_frame()
        pixmap = draw_cue_overlay(frame_to_pixmap(frame), task)
        self._set_pixmap(pixmap)

    def show_feedback(self, predicted_task: TaskType, is_correct: bool):
        self._stop_animations()
        self._anim_action = ACTION_MAP[predicted_task]
        self._anim_is_correct = is_correct
        self._anim_timer.start(33)

    def reset_env(self):
        self._stimulus.reset()

    def close_env(self):
        self._stimulus.close()

    def _anim_step(self):
        if self._anim_action is None:
            return
        frame = self._stimulus.step(self._anim_action)
        pixmap = draw_feedback_border(frame_to_pixmap(frame), self._anim_is_correct)
        self._set_pixmap(pixmap)

    def _blink_step(self):
        self._blink_visible = not self._blink_visible
        frame = self._stimulus.get_frame()
        if self._blink_visible and self._blink_task is not None:
            pixmap = draw_cue_overlay(frame_to_pixmap(frame), self._blink_task)
        else:
            pixmap = frame_to_pixmap(frame)
        self._set_pixmap(pixmap)

    def _stop_animations(self):
        self._anim_timer.stop()
        self._blink_timer.stop()
        self._anim_action = None
        self._blink_task = None

    def _set_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)

    def closeEvent(self, event):
        self._stop_animations()
        self.close_env()
        super().closeEvent(event)
