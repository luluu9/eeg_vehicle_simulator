# EEG Vehicle Simulator

## Overview
This project runs a Gymnasium simulator controlled by EEG segments selected using keyboard input.

## How it works
1. You press an arrow key.
2. The key is mapped to an action class (Left / Right / Both / Feet / Relax).
3. A random EEG segment from the matching class is selected.
4. Features are computed from the segment.
5. A CSP + SVM (RBF kernel) model predicts the class.
6. The predicted class is converted to the simulator action vector: `[steer, gas, brake]`.
7. The action vector is sent to the Gymnasium environment.

## Controls
- Left Arrow: Left
- Right Arrow: Right
- Up Arrow: Both
- Down Arrow: Feet

## EEG classes
- 1: Relax
- 2: Left hand
- 3: Right hand
- 4: Both hands
- 5: Both Feet

## Action mapping
- Relax (1): `[0.0, 0.0, 0.0]`
- Left (2): `[-0.5, 0.0, 0.0]`
- Right (3): `[0.0, 0.5, 0.0]`
- Both (4): `[0.0, 0.5, 0.0]`
- Feet (5): `[0.0, 0.0, 1.0]`

## Files
- `simulator.py` : main file, runs the simulator
- `model.joblib` : trained CSP + SVM (RBF) model
- `tools.py` : `split_epochs_into_segments`, `get_freq`
- 

## Requirements
- Python 3.11
- numpy
- mne
- joblib
- gymnasium
- scikit-learn

