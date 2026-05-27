from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QTimer, Qt

from ..config import CollectorConfig, TASK_LABELS, TaskType
from ..data_handler import LSLClient, DataLogger
from ..experiment import ExperimentSession, ExperimentState
from ..stimulus import StimulusWindow


class CollectorWindow(QMainWindow):
    def __init__(self, config: CollectorConfig = None):
        super().__init__()
        self.config = config or CollectorConfig()
        self.setWindowTitle("BrainBot Data Collector")
        self.resize(450, 350)

        self.lsl_client = LSLClient()
        self.data_logger = DataLogger()
        self.experiment = None
        self.stimulus_window = None
        self._current_run = 0

        self._init_ui()

        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._refresh_streams)
        self._refresh_timer.start(2000)
        self._refresh_streams()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- Config group ---
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Subject ID:"))
        self.subject_input = QLineEdit("SUBJ01")
        row.addWidget(self.subject_input)
        config_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("LSL Stream:"))
        self.stream_combo = QComboBox()
        row.addWidget(self.stream_combo)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_streams)
        row.addWidget(self.refresh_btn)
        config_layout.addLayout(row)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # --- Status group ---
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("Idle")
        status_layout.addWidget(self.status_label)

        self.progress_label = QLabel("Trial: 0 / 0  |  Run: 0 / 0")
        status_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.config.total_trials)
        status_layout.addWidget(self.progress_bar)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # --- Controls ---
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

    def _refresh_streams(self):
        streams = self.lsl_client.find_streams()
        current = self.stream_combo.currentText()
        self.stream_combo.clear()
        for s in streams:
            self.stream_combo.addItem(f"{s.name()} ({s.type()})", s)
        idx = self.stream_combo.findText(current)
        if idx >= 0:
            self.stream_combo.setCurrentIndex(idx)

    def _start(self):
        idx = self.stream_combo.currentIndex()
        if idx < 0:
            self.status_label.setText("No stream selected")
            return

        stream_info = self.stream_combo.itemData(idx)
        try:
            self.lsl_client.connect(stream_info)
            self.data_logger.set_stream_info(self.lsl_client.info)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            return

        self._refresh_timer.stop()

        self.stimulus_window = StimulusWindow()
        self.stimulus_window.space_pressed.connect(self._on_space_pressed)
        self.stimulus_window.showFullScreen()

        self.experiment = ExperimentSession(
            self.config, self.lsl_client, self.data_logger
        )
        self.experiment.state_changed.connect(self._on_state_changed)
        self.experiment.task_changed.connect(self._on_task_changed)
        self.experiment.feedback_ready.connect(self._on_feedback)
        self.experiment.progress_updated.connect(self._on_progress)
        self.experiment.countdown_tick.connect(self._on_countdown)
        self.experiment.waiting_for_space.connect(self._on_waiting)
        self.experiment.break_requested.connect(self._on_break)
        self.experiment.finished.connect(self._on_finished)

        self.experiment.start()
        self._set_controls_running(True)

    def _stop(self):
        if self.experiment:
            self.experiment.stop()

        subject = self.subject_input.text()
        filepath = self.data_logger.save(subject, "partial")
        self.data_logger.clear()

        if self.stimulus_window:
            self.stimulus_window.close()
            self.stimulus_window = None

        self._refresh_timer.start(2000)

        self._set_controls_running(False)
        self.status_label.setText(f"Stopped. Saved partial data.")

    def _on_state_changed(self, state: ExperimentState):
        if self.stimulus_window:
            if state == ExperimentState.IDLE:
                self.stimulus_window.show_idle()
            elif state == ExperimentState.IMAGERY:
                task = self.experiment.current_task
                self.stimulus_window.show_imagery(task)
        self.status_label.setText(state.name)

    def _on_countdown(self, seconds: int):
        if self.stimulus_window:
            self.stimulus_window.show_countdown(seconds)
        self.status_label.setText(f"Starting in {seconds}...")

    def _on_waiting(self):
        if self.stimulus_window:
            self.stimulus_window.show_waiting()
        self.status_label.setText("Press SPACE to continue")

    def _on_space_pressed(self):
        if self.experiment:
            self.experiment.on_space_pressed()

    def _on_task_changed(self, task: TaskType):
        if self.stimulus_window:
            self.stimulus_window.show_cue(task)

    def _on_feedback(self, predicted: TaskType, is_correct: bool):
        if self.stimulus_window:
            self.stimulus_window.show_feedback(predicted, is_correct)
        label = TASK_LABELS[predicted]
        tag = "✓" if is_correct else "✗"
        self.status_label.setText(f"Feedback: {label} {tag}")

    def _on_progress(self, trial_in_run: int, trials_per_run: int, current_run: int):
        total_done = (current_run - 1) * trials_per_run + trial_in_run
        self.progress_label.setText(
            f"Trial: {trial_in_run} / {trials_per_run}  |  Run: {current_run} / {self.config.n_runs}"
        )
        self.progress_bar.setValue(total_done)

    def _on_break(self):
        if self.stimulus_window:
            self.stimulus_window.show_idle()

        QMessageBox.information(
            self,
            "Break",
            "Halfway done! Take a break (up to 10 min).\n"
            "Do not remove the EEG cap.\n\n"
            "Press OK when ready to continue.",
        )
        if self.experiment:
            self.experiment.resume_from_break()

    def _on_finished(self):
        subject = self.subject_input.text()
        filepath = self.data_logger.save(subject, "full")
        self.data_logger.clear()

        if self.stimulus_window:
            self.stimulus_window.close()
            self.stimulus_window = None

        self._set_controls_running(False)
        self._refresh_timer.start(2000)

        total = self.config.total_trials
        self.status_label.setText(f"Finished! {total} trials saved.")
        QMessageBox.information(self, "Done", f"Data collection complete.\n{total} trials saved to:\n{filepath}")

    def _set_controls_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.subject_input.setEnabled(not running)
        self.stream_combo.setEnabled(not running)

    def closeEvent(self, event):
        if self.experiment and self.experiment.running:
            self._stop()
        if self.stimulus_window:
            self.stimulus_window.close()
        event.accept()
