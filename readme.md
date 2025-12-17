# EEG Vehicle Simulator

This project implements a modular, LSL-based system for controlling a vehicle simulator using EEG signals. It is architected as two independent applications—a **Predictor** and a **Simulator** communicating via Lab Streaming Layer (LSL).

## Architecture

The system is decoupled to allow flexible testing, model comparison, and distributed processing.

**Data Flow:**
1.  **EEG Source**: `eeg_collector` (Replayer or Live Amp) streams raw EEG data via LSL.
2.  **Predictor App**:
    *   Connects to the raw EEG stream.
    *   Preprocesses data (Bandpass 8-32Hz, Notch 50Hz, Resampling).
    *   Runs multiple classifiers in parallel (e.g., CSP+SVM, Ground Truth).
    *   Broadcasts probability vectors (`[Relax, Left, Right, Both, Feet]`) to unique LSL streams.
3.  **Simulator App**:
    *   Listens to *all* available classifier LSL streams.
    *   Applies a Control Strategy (e.g., Threshold, Accumulator) to map probabilities to car actions (`steer`, `gas`, `brake`).
    *   Renders the simulation (CarRacing) and a Debug HUD.

### Prerequisites
*   Python 3.10+
*   Dependencies: `numpy`, `scipy`, `pylsl`, `pyqt6`, `pyqtgraph`, `gymnasium[box2d]`, `pygame`, `joblib`, `mne` (requirements.txt)