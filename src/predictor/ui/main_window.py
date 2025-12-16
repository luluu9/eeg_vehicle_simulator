from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QCheckBox, QGroupBox, QScrollArea, QDoubleSpinBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
import pyqtgraph as pg
import numpy as np
from ..core.engine import PredictorEngine
from ..core.classifiers import MockClassifier, CSPSVMClassifier
from ...common.constants import LSLChannel

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
        
        layout.addLayout(ctrl_layout)
        
        # Split Visualization: Bars | History
        viz_layout = QHBoxLayout()
        
        # 1. Bar Chart (Current)
        self.bar_plot = pg.PlotWidget()
        self.bar_plot.setMaximumWidth(200)
        self.bar_items = pg.BarGraphItem(x=range(5), height=[0]*5, width=0.6, brush='b')
        self.bar_plot.addItem(self.bar_items)
        # Fix axis
        self.bar_plot.setYRange(0, 1)
        self.bar_plot.getAxis('bottom').setTicks([list(zip(range(5), LSLChannel.names()))])
        viz_layout.addWidget(self.bar_plot)
        
        # 2. History Line Chart
        self.history_plot = pg.PlotWidget()
        self.history_plot.setYRange(0, 1)
        self.history_plot.showGrid(x=True, y=True)
        self.history_plot.addLegend()
        self.lines = {}
        colors = ['g', 'r', 'b', 'c', 'm'] # Relax, Left, Right, Both, Feet
        for idx, name in enumerate(LSLChannel.names()):
            self.lines[name] = self.history_plot.plot(pen=colors[idx], name=name)
            
        viz_layout.addWidget(self.history_plot)
        
        layout.addLayout(viz_layout)
        self.setLayout(layout)
        
        # Data storage for history
        self.history_data = {name: [] for name in LSLChannel.names()}
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
        
    @pyqtSlot(list) # expects normalized probabilities list
    def update_viz(self, probs):
        self.bar_items.setOpts(height=probs)
        
        # Update History Buffer
        for i, name in enumerate(LSLChannel.names()):
            self.history_data[name].append(probs[i])
            if len(self.history_data[name]) > self.buffer_size:
                self.history_data[name].pop(0)
        
        self._refresh_lines()

    def set_history_length(self, length: int):
        self.visible_history = length
        self._refresh_lines()
        
    def _refresh_lines(self):
        for name in LSLChannel.names():
            data = self.history_data[name]
            # Show only last N points
            if len(data) > self.visible_history:
                vis_data = data[-self.visible_history:]
            else:
                vis_data = data
                
            # X axis: Right (0) to Left (Negative)
            x = np.arange(-len(vis_data) + 1, 1)
            self.lines[name].setData(x, vis_data)

class PredictorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Predictor Brain")
        self.resize(1000, 800)
        
        self.engine = PredictorEngine()
        self.engine.prediction_made.connect(self.on_prediction)
        self.engine.error_occurred.connect(self.on_error)
        
        self.widgets = {} # name -> ClassifierWidget
        
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
        
        # Scroll Area for Classifiers
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        scroll.setWidget(self.container)
        
        main_layout.addWidget(scroll)
        
    def load_defaults(self):
        self.add_classifier_ui(MockClassifier())
        
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
            else:
                self.start_btn.setText("Retry Connect")
        else:
            self.engine.stop()
            self.start_btn.setText("Connect")
            self.start_btn.setStyleSheet("")
            self.stream_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            
    @pyqtSlot(str, object) # object=np.ndarray
    def on_prediction(self, name, probs):
        if name in self.widgets:
            self.widgets[name].update_viz(probs)
            
    @pyqtSlot(str)
    def on_error(self, msg):
        self.statusBar().showMessage(f"Error: {msg}")
