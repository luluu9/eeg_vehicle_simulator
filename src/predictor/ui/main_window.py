from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QCheckBox, QGroupBox, QScrollArea, QDoubleSpinBox, QComboBox, QTabWidget, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
import pyqtgraph as pg
import numpy as np
from ..core.engine import PredictorEngine
from ..core.classifiers import GroundTruthClassifier
from ...common.constants import StudyClass


PREDICTOR_CLASS_NAMES = [study_class.name for study_class in StudyClass]
PREDICTOR_CLASS_COLORS = ['g', 'r', 'b', 'c']

class ClassifierWidget(QGroupBox):
    def __init__(self, name: str, min_w: float, max_w: float, engine: PredictorEngine):
        super().__init__(name)
        self.engine = engine
        self.name = name
        self.min_w = min_w
        self.max_w = max_w
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        ctrl_layout = QHBoxLayout()
        
        self.active_cb = QCheckBox("Broadcast")
        self.active_cb.setChecked(True)
        self.active_cb.toggled.connect(self.toggle_active)
        ctrl_layout.addWidget(self.active_cb)
        
        ctrl_layout.addWidget(QLabel("Window:"))
        # Slider is integer, 100x scale
        self.win_slider = QSlider(Qt.Orientation.Horizontal)
        self.win_slider.setRange(int(self.min_w*100), int(self.max_w*100))
        self.win_slider.setValue(int(self.min_w*100))
        self.win_slider.valueChanged.connect(self.update_window)
        ctrl_layout.addWidget(self.win_slider)
        
        self.win_label = QLabel(f"{self.min_w:.2f}s")
        ctrl_layout.addWidget(self.win_label)

        # Latency Label
        self.latency_label = QLabel("Lag: --")
        self.latency_label.setStyleSheet("color: gray")
        ctrl_layout.addWidget(self.latency_label)
        
        layout.addLayout(ctrl_layout)
        
        # Split Visualization: Bars | History
        viz_layout = QHBoxLayout()
        
        # 1. Bar Chart (Current)
        self.bar_plot = pg.PlotWidget()
        self.bar_plot.setMaximumWidth(200)
        self.bar_items = pg.BarGraphItem(x=range(len(PREDICTOR_CLASS_NAMES)), height=[0] * len(PREDICTOR_CLASS_NAMES), width=0.6, brush='b')
        self.bar_plot.addItem(self.bar_items)
        # Fix axis
        self.bar_plot.setYRange(0, 1)
        self.bar_plot.getAxis('bottom').setTicks([list(zip(range(len(PREDICTOR_CLASS_NAMES)), PREDICTOR_CLASS_NAMES))])
        viz_layout.addWidget(self.bar_plot)
        
        # 2. History Line Chart
        self.history_plot = pg.PlotWidget()
        self.history_plot.setYRange(0, 1)
        self.history_plot.showGrid(x=True, y=True)
        self.history_plot.addLegend()
        self.lines = {}
        for idx, name in enumerate(PREDICTOR_CLASS_NAMES):
            self.lines[name] = self.history_plot.plot(pen=PREDICTOR_CLASS_COLORS[idx], name=name)
            
        viz_layout.addWidget(self.history_plot)
        
        layout.addLayout(viz_layout)
        self.setLayout(layout)
        
        # Data storage for history
        self.history_data = {name: [] for name in PREDICTOR_CLASS_NAMES}
        self.visible_history = 100
        self.buffer_size = 500
        
    def toggle_active(self, checked):
        # We need to expose this in engine
        if self.name in self.engine.classifiers:
            self.engine.classifiers[self.name].active = checked

    def update_window(self, val):
        sec = val / 100.0
        self.win_label.setText(f"{sec:.2f}s")
        self.engine.set_classifier_window(self.name, sec)
        
    @pyqtSlot(list, float) # expects normalized probabilities list, latency
    def update_viz(self, probs, latency):
        self.bar_items.setOpts(height=probs)
        
        # Update Latency
        self.latency_label.setText(f"Lag: {latency*1000:.0f}ms")
        if latency > 0.5:
             self.latency_label.setStyleSheet("color: red; font-weight: bold")
        elif latency > 0.1:
             self.latency_label.setStyleSheet("color: orange")
        else:
             self.latency_label.setStyleSheet("color: green")
        
        # Update History Buffer
        for i, name in enumerate(PREDICTOR_CLASS_NAMES):
            self.history_data[name].append(probs[i])
            if len(self.history_data[name]) > self.buffer_size:
                self.history_data[name].pop(0)
        
        self._refresh_lines()

    def set_history_length(self, length: int):
        self.visible_history = length
        self._refresh_lines()
        
    def _refresh_lines(self):
        for name in PREDICTOR_CLASS_NAMES:
            data = self.history_data[name]
            # Show only last N points
            if len(data) > self.visible_history:
                vis_data = data[-self.visible_history:]
            else:
                vis_data = data
                
            # X axis: Right (0) to Left (Negative)
            x = np.arange(-len(vis_data) + 1, 1)
            self.lines[name].setData(x, vis_data)

class EEGSignalTab(QWidget):
    def __init__(self, engine: PredictorEngine):
        super().__init__()
        self.engine = engine
        self.show_preprocessed = False
        self.n_channels = 16
        self.display_seconds = 5.0
        self.active_channels = list(range(self.n_channels))
        
        self.init_ui()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_plot)
        
    def set_active(self, active: bool):
        if active:
            self.update_timer.start(100)
        else:
            self.update_timer.stop()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        ctrl_layout = QHBoxLayout()
        self.preprocessed_cb = QCheckBox("Show Preprocessed")
        self.preprocessed_cb.setChecked(False)
        self.preprocessed_cb.toggled.connect(self.on_preprocessed_toggled)
        ctrl_layout.addWidget(self.preprocessed_cb)
        
        self.autoscale_cb = QCheckBox("Autoscale")
        self.autoscale_cb.setChecked(True)
        ctrl_layout.addWidget(self.autoscale_cb)
        
        ctrl_layout.addWidget(QLabel("Window:"))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(1.0, 30.0)
        self.time_spin.setSingleStep(1.0)
        self.time_spin.setSuffix(" s")
        self.time_spin.setValue(self.display_seconds)
        self.time_spin.valueChanged.connect(self.on_time_changed)
        ctrl_layout.addWidget(self.time_spin)
        
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
        content_layout = QHBoxLayout()
        
        # Channel selector
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.channel_list.setMaximumWidth(100)
        for i in range(self.n_channels):
            item = QListWidgetItem(f"Ch{i+1}")
            self.channel_list.addItem(item)
            item.setSelected(True)
        self.channel_list.itemSelectionChanged.connect(self.on_channel_selection_changed)
        content_layout.addWidget(self.channel_list)
        
        # Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.setLabel('left', 'Channel')
        self.plot_widget.showGrid(x=True, y=False)
        self.plot_widget.setLimits(xMin=None, xMax=0)
        content_layout.addWidget(self.plot_widget)
        
        layout.addLayout(content_layout)
        
        self.curves = []
        for i in range(self.n_channels):
            pen = pg.intColor(i, self.n_channels)
            curve = self.plot_widget.plot(pen=pen)
            self.curves.append(curve)
        
    def on_preprocessed_toggled(self, checked):
        self.show_preprocessed = checked
    
    def on_time_changed(self, value):
        self.display_seconds = value
        
    def on_channel_selection_changed(self):
        self.active_channels = [i for i in range(self.channel_list.count())
                                if self.channel_list.item(i).isSelected()]
        for i in range(self.n_channels):
            self.curves[i].setVisible(i in self.active_channels)
        
    def refresh_plot(self):
        handler = self.engine.data_handler
        if handler.srate == 0 or handler._total_samples == 0:
            return
            
        data, timestamps = handler.get_latest_window(self.display_seconds)
        if data is None:
            return
            
        if self.show_preprocessed:
            try:
                data = self.engine.preprocessor.process(data, handler.srate)
                fs = self.engine.preprocessor.target_srate
            except Exception:
                return
        else:
            if data.shape[0] > 16:
                data = data[1:17, :]
            fs = handler.srate
            
        n_samples = data.shape[1]
        t = np.linspace(-n_samples / fs, 0, n_samples)
        
        if not self.show_preprocessed and self.autoscale_cb.isChecked():
            active_data = data[self.active_channels] if len(self.active_channels) > 0 else data
            global_std = np.std(active_data)
            spacing = 2.0
            visible_idx = 0
            for i in range(min(self.n_channels, data.shape[0])):
                if i in self.active_channels:
                    ch = data[i]
                    ch_normalized = (ch - np.mean(ch)) / (global_std if global_std > 0 else 1)
                    self.curves[i].setData(t, ch_normalized + visible_idx * spacing)
                    visible_idx += 1
                else:
                    self.curves[i].clear()
        else:
            spacing = np.std(data) * 4 if np.std(data) > 0 else 1
            visible_idx = 0
            for i in range(min(self.n_channels, data.shape[0])):
                if i in self.active_channels:
                    self.curves[i].setData(t, data[i] + visible_idx * spacing)
                    visible_idx += 1
                else:
                    self.curves[i].clear()
        
        n_visible = len(self.active_channels)
        self.plot_widget.setLimits(
            xMin=-n_samples / fs, xMax=0,
            yMin=-spacing, yMax=max(n_visible, 1) * spacing
        )
        self.plot_widget.setYRange(-spacing, max(n_visible, 1) * spacing)
        self.plot_widget.getAxis('left').setTicks(
            [[(i, f"Ch{ch+1}") for i, ch in enumerate(self.active_channels)]]
        )


class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Predictor Brain")
        self.resize(1000, 800)
        
        self.engine = PredictorEngine()
        self.engine.prediction_made.connect(self.on_prediction)
        self.engine.error_occurred.connect(self.on_error)
        
        self.widgets = {} # name -> ClassifierWidget
        self.ground_truth_classifier = None
        
        self.init_ui()
        self.load_defaults()
        self.refresh_streams()
        
    def init_ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        main_layout = QVBoxLayout(w)
        
        # Top Bar
        top_bar = QHBoxLayout()
        
        # Stream Selection
        top_bar.addWidget(QLabel("Input Stream:"))
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_streams)
        top_bar.addWidget(self.refresh_btn)

        self.stream_combo = QComboBox()
        self.stream_combo.setMinimumWidth(200)
        top_bar.addWidget(self.stream_combo)
        
        self.start_btn = QPushButton("Connect")
        self.start_btn.clicked.connect(self.toggle_start)
        top_bar.addWidget(self.start_btn)
        
        top_bar.addWidget(QLabel("Global Interval:"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 5.0)
        self.interval_spin.setSingleStep(0.1)
        self.interval_spin.setValue(2.0)
        self.interval_spin.valueChanged.connect(self.engine.set_interval)
        top_bar.addWidget(self.interval_spin)

        top_bar.addWidget(QLabel("History (10-200):"))
        self.history_slider = QSlider(Qt.Orientation.Horizontal)
        self.history_slider.setRange(10, 200)
        self.history_slider.setValue(100)
        self.history_slider.setFixedWidth(190)
        self.history_slider.setTickInterval(10)
        self.history_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.history_slider.valueChanged.connect(self.update_history_length)
        top_bar.addWidget(self.history_slider)
        
        main_layout.addLayout(top_bar)
        
        # Tab Widget
        self.tabs = QTabWidget()
        
        # Tab 1: Classifiers
        classifiers_tab = QWidget()
        classifiers_layout = QVBoxLayout(classifiers_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        scroll.setWidget(self.container)
        classifiers_layout.addWidget(scroll)
        self.tabs.addTab(classifiers_tab, "Classifiers")
        
        # Tab 2: EEG Signal
        self.eeg_signal_tab = EEGSignalTab(self.engine)
        self.tabs.addTab(self.eeg_signal_tab, "EEG Signal")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        main_layout.addWidget(self.tabs)
        
    def load_defaults(self):
        self.ground_truth_classifier = GroundTruthClassifier()
        self.add_classifier_ui(self.ground_truth_classifier)
    
    def on_tab_changed(self, index):
        self.eeg_signal_tab.set_active(index == 1)
        
    def add_classifier_ui(self, clf):
        self.engine.add_classifier(clf)
        widget = ClassifierWidget(clf.name, clf.min_window, clf.max_window, self.engine)
        self.container_layout.addWidget(widget)
        self.widgets[clf.name] = widget
    
    def update_history_length(self, val):
        for w in self.widgets.values():
            w.set_history_length(val)

    def refresh_streams(self):
        streams = self.engine.find_streams()
        current_text = self.stream_combo.currentText()
        self.stream_combo.clear()
        
        for stream_info in streams:
            txt = f"{stream_info.name()} ({stream_info.type()})"
            self.stream_combo.addItem(txt, stream_info)
            
        # Restore selection
        idx = self.stream_combo.findText(current_text)
        if idx >= 0:
            self.stream_combo.setCurrentIndex(idx)
        
    def toggle_start(self):
        if self.start_btn.text().startswith("Connect"):
            # Get selected
            idx = self.stream_combo.currentIndex()
            target = None
            if idx >= 0:
                target = self.stream_combo.itemData(idx)
            
            if self.engine.start_stream_input(target):
                self.start_btn.setText("Stop")
                self.start_btn.setStyleSheet("background-color: #ffcccc")
                self.stream_combo.setEnabled(False)
                self.refresh_btn.setEnabled(False)
                if self.ground_truth_classifier:
                    self.ground_truth_classifier.start(target.name())
            else:
                self.start_btn.setText("Retry Connect")
        else:
            self.engine.stop()
            self.start_btn.setText("Connect")
            self.start_btn.setStyleSheet("")
            self.stream_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            
    @pyqtSlot(str, object, float) # object=np.ndarray
    def on_prediction(self, name, probs, latency):
        if name in self.widgets:
            self.widgets[name].update_viz(probs, latency)
            
    @pyqtSlot(str)
    def on_error(self, msg):
        self.statusBar().showMessage(f"Error: {msg}")
