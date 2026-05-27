import numpy as np
import gymnasium as gym
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QColor, QFont, QPen, QBrush
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

WHEELCHAIR_CENTER_X_RATIO = 0.5
WHEELCHAIR_CENTER_Y_RATIO = 0.75

class WheelchairStimulus:
    def __init__(self):
        self.env = gym.make("WheelchairRacing-v0", render_mode="rgb_array")
        self.env.reset()
        self._current_frame = self.env.render()

    def get_frame(self) -> np.ndarray:
        return self._current_frame

    def step(self, action: np.ndarray) -> np.ndarray:
        self.env.step(action)
        self._current_frame = self.env.render()
        return self._current_frame

    def reset(self):
        self.env.reset()
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

    cx = int(result.width() * WHEELCHAIR_CENTER_X_RATIO)
    cy = int(result.height() * WHEELCHAIR_CENTER_Y_RATIO)

    cross_size = 20
    pen = QPen(QColor(0, 100, 255), 3)
    painter.setPen(pen)
    painter.drawLine(cx - cross_size, cy, cx + cross_size, cy)
    painter.drawLine(cx, cy - cross_size, cx, cy + cross_size)

    symbol = CUE_SYMBOLS[task]
    symbol_size = 90
    font = QFont("Arial", symbol_size, QFont.Weight.Bold)
    painter.setFont(font)

    path = QPainterPath()
    path.addText(0, 0, font, symbol)
    br = path.boundingRect()
    path.translate(cx - br.center().x(), cy - br.center().y())
    painter.setPen(QPen(QColor(255, 255, 0), 3))
    painter.setBrush(QBrush(QColor(255, 255, 0)))
    painter.drawPath(path)

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


def draw_countdown_overlay(pixmap: QPixmap, seconds: int) -> QPixmap:
    result = pixmap.copy()
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx = int(result.width() * WHEELCHAIR_CENTER_X_RATIO)
    cy = int(result.height() * WHEELCHAIR_CENTER_Y_RATIO)

    font = QFont("Arial", 120, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QPen(QColor(0, 100, 255)))

    path = QPainterPath()
    path.addText(0, 0, font, str(seconds))
    br = path.boundingRect()
    path.translate(cx - br.center().x(), cy - br.center().y())
    painter.setBrush(QBrush(QColor(0, 100, 255, 180)))
    painter.drawPath(path)

    painter.end()
    return result


NOOP_ACTION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SIM_INTERVAL_MS = 50


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

        self._action = NOOP_ACTION
        self._overlay_task = None
        self._feedback_border = None
        self._countdown_value = None

        self._sim_timer = QTimer()
        self._sim_timer.timeout.connect(self._sim_step)
        self._sim_timer.start(SIM_INTERVAL_MS)

    def show_idle(self):
        self._action = NOOP_ACTION
        self._overlay_task = None
        self._feedback_border = None
        self._countdown_value = None

    def show_cue(self, task: TaskType):
        self._action = NOOP_ACTION
        self._overlay_task = task
        self._feedback_border = None
        self._countdown_value = None

    def show_imagery(self, task: TaskType):
        self._action = NOOP_ACTION
        self._overlay_task = task
        self._feedback_border = None
        self._countdown_value = None

    def show_feedback(self, predicted_task: TaskType, is_correct: bool):
        self._action = ACTION_MAP[predicted_task]
        self._overlay_task = None
        self._feedback_border = is_correct
        self._countdown_value = None

    def show_countdown(self, seconds: int):
        self._action = NOOP_ACTION
        self._overlay_task = None
        self._feedback_border = None
        self._countdown_value = seconds

    def reset_env(self):
        self._stimulus.reset()

    def close_env(self):
        self._sim_timer.stop()
        self._stimulus.close()

    def _sim_step(self):
        frame = self._stimulus.step(self._action)
        pixmap = frame_to_pixmap(frame)
        if self._overlay_task is not None:
            pixmap = draw_cue_overlay(pixmap, self._overlay_task)
        if self._feedback_border is not None:
            pixmap = draw_feedback_border(pixmap, self._feedback_border)
        if self._countdown_value is not None:
            pixmap = draw_countdown_overlay(pixmap, self._countdown_value)
        self._set_pixmap(pixmap)

    def _set_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._label.setPixmap(scaled)

    def closeEvent(self, event):
        self.close_env()
        super().closeEvent(event)
