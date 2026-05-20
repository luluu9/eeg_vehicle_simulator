from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QComboBox, QPushButton, QGroupBox,
)

from .input_handler import MultiStreamMonitor
from .sequencer import ExperimentSequencer, TASKS, STRATEGIES, ERRP_STRATEGIES
from ..common.constants import StudyClass

_CLASS_LABELS = {
    StudyClass.REST.value:    "○  REST",
    StudyClass.LEFT.value:    "←  LEFT",
    StudyClass.RIGHT.value:   "→  RIGHT",
    StudyClass.FORWARD.value: "↑  FORWARD",
}


class LaunchWindow(QDialog):
    def __init__(self, monitor: MultiStreamMonitor, sequencer: ExperimentSequencer,
                 subject_id: str = "", mi_stream: str = "", errp_stream: str = ""):
        super().__init__()
        self.monitor = monitor
        self.sequencer = sequencer
        self.setWindowTitle("BrainBot — Evaluation Setup")
        self.setMinimumWidth(480)

        self._result: tuple | None = None

        root = QVBoxLayout(self)

        # ── Subject ID ──────────────────────────────────────────────
        grp_subj = QGroupBox("Subject")
        lay_subj = QHBoxLayout(grp_subj)
        lay_subj.addWidget(QLabel("ID:"))
        self._subj = QLineEdit(subject_id)
        self._subj.setPlaceholderText("e.g. S01")
        lay_subj.addWidget(self._subj)
        root.addWidget(grp_subj)

        # ── Streams ──────────────────────────────────────────────────
        grp_streams = QGroupBox("LSL Streams")
        lay_streams = QHBoxLayout(grp_streams)

        col_mi = QVBoxLayout()
        col_mi.addWidget(QLabel("MI (Probabilities)"))
        self._mi_list = QListWidget()
        self._mi_list.setMaximumHeight(100)
        col_mi.addWidget(self._mi_list)
        lay_streams.addLayout(col_mi)

        col_errp = QVBoxLayout()
        col_errp.addWidget(QLabel("ErrP (optional)"))
        self._errp_list = QListWidget()
        self._errp_list.setMaximumHeight(100)
        col_errp.addWidget(self._errp_list)
        lay_streams.addLayout(col_errp)

        root.addWidget(grp_streams)

        # ── Live prediction ──────────────────────────────────────────
        grp_pred = QGroupBox("Live Prediction")
        lay_pred = QHBoxLayout(grp_pred)
        self._pred_label = QLabel("—")
        self._pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pred_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        lay_pred.addWidget(self._pred_label)
        root.addWidget(grp_pred)

        # ── Next experiment ──────────────────────────────────────────
        grp_next = QGroupBox("Next Experiment")
        lay_next = QVBoxLayout(grp_next)

        self._next_label = QLabel()
        self._next_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_label.setStyleSheet("font-size: 16px;")
        lay_next.addWidget(self._next_label)

        lay_override = QHBoxLayout()
        lay_override.addWidget(QLabel("Override:"))
        self._task_combo = QComboBox()
        self._task_combo.addItems(TASKS)
        lay_override.addWidget(self._task_combo)
        self._strategy_combo = QComboBox()
        self._strategy_combo.addItems(STRATEGIES)
        lay_override.addWidget(self._strategy_combo)
        btn_override = QPushButton("Set as next")
        btn_override.clicked.connect(self._apply_override)
        lay_override.addWidget(btn_override)
        lay_next.addLayout(lay_override)

        root.addWidget(grp_next)

        # ── Start ────────────────────────────────────────────────────
        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setStyleSheet("font-size: 18px; padding: 8px;")
        self._start_btn.clicked.connect(self._on_start)
        root.addWidget(self._start_btn)

        # ── Timers ───────────────────────────────────────────────────
        self._stream_timer = QTimer(self)
        self._stream_timer.timeout.connect(self._refresh_streams)
        self._stream_timer.start(2000)

        self._pred_timer = QTimer(self)
        self._pred_timer.timeout.connect(self._refresh_prediction)
        self._pred_timer.start(150)

        self._refresh_streams()
        self._refresh_next_label()

        # Restore previous selections
        self._restore_selection(self._mi_list, mi_stream)
        self._restore_selection(self._errp_list, errp_stream)

        self._subj.textChanged.connect(self._update_start_btn)
        self._mi_list.itemSelectionChanged.connect(self._update_start_btn)
        self._errp_list.itemSelectionChanged.connect(self._update_start_btn)
        self._update_start_btn()

    # ── Helpers ──────────────────────────────────────────────────────

    def _restore_selection(self, lst: QListWidget, name: str):
        for i in range(lst.count()):
            if lst.item(i).text() == name:
                lst.setCurrentRow(i)
                return

    def _refresh_streams(self):
        mi_names = sorted(self.monitor.get_probabilities().keys())
        errp_names = sorted(self.monitor.get_errp().keys())
        self._update_list(self._mi_list, mi_names)
        self._update_list(self._errp_list, errp_names)
        self._update_start_btn()

    def _update_list(self, lst: QListWidget, names: list[str]):
        current = lst.currentItem().text() if lst.currentItem() else ""
        lst.blockSignals(True)
        lst.clear()
        lst.addItems(names)
        self._restore_selection(lst, current)
        lst.blockSignals(False)

    def _refresh_prediction(self):
        mi = self._selected_mi()
        if not mi:
            self._pred_label.setText("—")
            return
        probs = self.monitor.get_probabilities().get(mi)
        if probs is None or len(probs) == 0:
            self._pred_label.setText("—")
            return
        idx = int(probs[:4].argmax())
        self._pred_label.setText(_CLASS_LABELS.get(idx, "?"))

    def _refresh_next_label(self):
        exp = self.sequencer.current
        if exp:
            self._next_label.setText(f"Task {exp.task}  ·  {exp.strategy.capitalize()}  (#{exp.task_id})")
        else:
            self._next_label.setText("All experiments completed")

    def _apply_override(self):
        self.sequencer.override_current(
            self._task_combo.currentText(),
            self._strategy_combo.currentText(),
        )
        self._refresh_next_label()

    def _selected_mi(self) -> str:
        item = self._mi_list.currentItem()
        return item.text() if item else ""

    def _selected_errp(self) -> str:
        item = self._errp_list.currentItem()
        return item.text() if item else ""

    def _errp_required(self) -> bool:
        exp = self.sequencer.current
        return exp is not None and exp.strategy in ERRP_STRATEGIES

    def _update_start_btn(self):
        ok = bool(self._subj.text().strip()) and bool(self._selected_mi())
        if self._errp_required():
            ok = ok and bool(self._selected_errp())
        self._start_btn.setEnabled(ok)

    def _on_start(self):
        self._result = (
            self._subj.text().strip(),
            self._selected_mi(),
            self._selected_errp() or None,
        )
        self.accept()

    # ── Public API ───────────────────────────────────────────────────

    def exec_and_get(self) -> tuple | None:
        self._refresh_next_label()
        if self.exec() == QDialog.DialogCode.Accepted:
            return self._result
        return None
