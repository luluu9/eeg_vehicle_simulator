from enum import Enum, auto

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from pylsl import local_clock

from .classifier import MockClassifier
from .config import TASK_LABELS, CollectorConfig, TaskType


COUNTDOWN_FROM = 5


class ExperimentState(Enum):
    IDLE = auto()
    WAITING = auto()
    COUNTDOWN = auto()
    CUE = auto()
    IMAGERY = auto()
    FEEDBACK = auto()
    BREAK = auto()
    FINISHED = auto()


class ExperimentSession(QObject):
    state_changed = pyqtSignal(ExperimentState)
    task_changed = pyqtSignal(TaskType)
    feedback_ready = pyqtSignal(TaskType, bool)  # predicted_task, is_correct
    progress_updated = pyqtSignal(int, int, int)  # trial_in_run, trials_per_run, current_run
    countdown_tick = pyqtSignal(int)  # seconds remaining
    waiting_for_space = pyqtSignal()  # request user to press SPACE
    break_requested = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, config: CollectorConfig, lsl_client=None, data_logger=None):
        super().__init__()
        self.config = config
        self.lsl_client = lsl_client
        self.data_logger = data_logger
        self.classifier = MockClassifier(error_rate=config.error_rate)

        self.state = ExperimentState.IDLE
        self.current_run = 0
        self.current_trial_in_run = 0
        self.trial_sequence = []
        self.current_task = None
        self.running = False
        self.paused = False
        self._total_trials_done = 0

        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._countdown_timer = QTimer()
        self._countdown_timer.setSingleShot(True)
        self._countdown_timer.timeout.connect(self._countdown_step)
        self._countdown_remaining = 0
        self._countdown_callback = None

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_data)

    @property
    def total_trials_done(self) -> int:
        return self._total_trials_done

    def start(self):
        self.running = True
        self.paused = False
        self.current_run = 0
        self.current_trial_in_run = 0
        self._total_trials_done = 0

        if self.lsl_client:
            self.lsl_client.start_recording()
        self._poll_timer.start(100)
        self._enter_waiting(self._start_run)

    def stop(self):
        self.running = False
        self._timer.stop()
        self._countdown_timer.stop()
        self._poll_timer.stop()
        if self.lsl_client:
            self.lsl_client.stop_recording()
        self.state = ExperimentState.IDLE
        self.state_changed.emit(self.state)

    def pause(self):
        if not self.running:
            return
        self.paused = True
        self._timer.stop()
        self.state_changed.emit(ExperimentState.IDLE)

    def resume(self):
        if not self.running or not self.paused:
            return
        self.paused = False
        if self.state == ExperimentState.BREAK:
            self._enter_waiting(self._start_run)
        else:
            self._enter_waiting(self._next_trial)

    def resume_from_break(self):
        if self.state == ExperimentState.BREAK:
            self.paused = False
            self._enter_waiting(self._start_run)

    def on_space_pressed(self):
        if self.state == ExperimentState.WAITING:
            self._start_countdown(self._waiting_callback)

    def _enter_waiting(self, callback):
        self._waiting_callback = callback
        self.state = ExperimentState.WAITING
        self.state_changed.emit(self.state)
        self.waiting_for_space.emit()

    def _start_countdown(self, callback):
        self._countdown_remaining = COUNTDOWN_FROM
        self._countdown_callback = callback
        self.state = ExperimentState.COUNTDOWN
        self.state_changed.emit(self.state)
        self.countdown_tick.emit(self._countdown_remaining)
        self._countdown_timer.start(1000)

    def _countdown_step(self):
        self._countdown_remaining -= 1
        if self._countdown_remaining <= 0:
            cb = self._countdown_callback
            self._countdown_callback = None
            cb()
        else:
            self.countdown_tick.emit(self._countdown_remaining)
            self._countdown_timer.start(1000)

    def _start_run(self):
        self.trial_sequence = self.config.generate_trial_sequence()
        self.current_trial_in_run = 0
        self._next_trial()

    def _next_trial(self):
        if not self.running or self.paused:
            return

        if self.current_trial_in_run >= len(self.trial_sequence):
            self.current_run += 1
            if self.current_run >= self.config.n_runs:
                self._finish()
                return
            if (self.config.break_every_n_runs > 0
                    and self.current_run % self.config.break_every_n_runs == 0):
                self.state = ExperimentState.BREAK
                self.state_changed.emit(self.state)
                self.break_requested.emit()
                return
            self._start_run()
            return

        self.current_task = self.trial_sequence[self.current_trial_in_run]
        self.progress_updated.emit(
            self.current_trial_in_run + 1,
            self.config.trials_per_run,
            self.current_run + 1,
        )
        self._enter_idle()

    def _enter_idle(self):
        self.state = ExperimentState.IDLE
        self.state_changed.emit(self.state)
        self._timer.start(int(self.config.idle_duration * 1000))

    def _enter_cue(self):
        self.state = ExperimentState.CUE
        self.state_changed.emit(self.state)
        self.task_changed.emit(self.current_task)
        self._emit_marker(self.config.get_marker(self.current_task))
        self._timer.start(int(self.config.cue_duration * 1000))

    def _enter_imagery(self):
        self.state = ExperimentState.IMAGERY
        self.state_changed.emit(self.state)
        self._timer.start(int(self.config.imagery_duration * 1000))

    def _enter_feedback(self):
        self.state = ExperimentState.FEEDBACK
        self.state_changed.emit(self.state)

        prediction = self.classifier.predict(self.current_task)
        is_correct = prediction == self.current_task

        self.feedback_ready.emit(prediction, is_correct)
        self._emit_marker(self.config.get_feedback_marker(prediction))
        quality = self.config.marker_correct if is_correct else self.config.marker_wrong
        self._emit_marker(quality)

        self._timer.start(int(self.config.feedback_duration * 1000))

    def _on_timeout(self):
        if self.state == ExperimentState.COUNTDOWN:
            self._countdown_step()
        elif self.state == ExperimentState.IDLE:
            self._enter_cue()
        elif self.state == ExperimentState.CUE:
            self._enter_imagery()
        elif self.state == ExperimentState.IMAGERY:
            self._enter_feedback()
        elif self.state == ExperimentState.FEEDBACK:
            self.current_trial_in_run += 1
            self._total_trials_done += 1
            self._next_trial()

    def _emit_marker(self, marker: int):
        if self.data_logger and self.lsl_client and self.lsl_client.lsl_offset is not None:
            timestamp = local_clock() - self.lsl_client.lsl_offset
            self.data_logger.add_event(timestamp, marker)

    def _poll_data(self):
        if self.lsl_client and self.data_logger:
            data, timestamps = self.lsl_client.get_data()
            if len(data) > 0:
                self.data_logger.add_data(data * 1e-6, timestamps)

    def _finish(self):
        self.stop()
        self.state = ExperimentState.FINISHED
        self.state_changed.emit(self.state)
        self.finished.emit()
