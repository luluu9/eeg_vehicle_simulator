import numpy as np
import joblib
from mne import concatenate_raws
from mne.io import read_raw_fif
from tools import split_epochs_into_segments, get_freq
import mne

classifier = joblib.load('model.joblib')

EPOCHS_BY_EVENT_ID = {}
ALL_EPOCHS = None


def initialize_epochs():
    raw_fnames = [
                  r"Konrad/KONRAD-3_sciskanie+ruszanie_run1_20251202_204815_raw.fif"]

    raw = concatenate_raws([read_raw_fif(f, preload=True) for f in raw_fnames])
    eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                    "A16"]
    raw.pick(eeg_channels)
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw.notch_filter(freqs=[50.0])

    konrad_mapping = {
        'A1': 'Cz', 'A2': 'FCz', 'A3': 'CP1', 'A4': 'FC1', 'A5': 'C1',
        'A6': 'CP3', 'A7': 'C3', 'A8': 'FC3', 'A9': 'C4', 'A10': 'FC4',
        'A11': 'Pz', 'A12': 'CP2', 'A13': 'CP4', 'A14': 'C2', 'A15': 'CPz',
        'A16': 'FC2'
    }
    raw.rename_channels(konrad_mapping)

    events, _ = mne.events_from_annotations(raw)
    all_events_id = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both', 5: 'Feet'}
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    mapping = {v: k for k, v in all_events_id.items()}
    task_margin = 0.5
    task_end = 4.5
    reject_criteria = dict(eeg=80e-6)  # 80 µV

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=mapping,
        baseline=None,
        tmin=task_margin,
        tmax=task_end,
        preload=True,
        reject=reject_criteria
    )

    segment_length = 2.0
    step = 0.5

    splitted_epochs = split_epochs_into_segments(epochs, segment_length, step)

    global EPOCHS_BY_EVENT_ID
    for event_id in range(1, 6):
        indices = np.where(splitted_epochs.events[:, -1] == event_id)
        if len(indices) > 0:
            EPOCHS_BY_EVENT_ID[event_id] = indices

    return splitted_epochs



def get_eeg_action(event_id):
    global ALL_EPOCHS, EPOCHS_BY_EVENT_ID

    if ALL_EPOCHS is None:
        ALL_EPOCHS = initialize_epochs()

    if event_id not in EPOCHS_BY_EVENT_ID or len(EPOCHS_BY_EVENT_ID[event_id]) == 0:
        raise ValueError(f"No epochs found for event ID {event_id}.")

    random_index = np.random.choice(EPOCHS_BY_EVENT_ID[event_id][0])

    epoch = ALL_EPOCHS[random_index]
    X_train, _ = get_freq(epoch)
    classification = classifier.predict(X_train)
    print("true event: " + str(event_id))
    print("predicted "+ str(classification))

    steer = 0
    gas = 0
    brake = 0
    if classification == 1:  # relaks
        pass
    elif classification == 2:  # Left
        steer = -1
    elif classification == 3:  # Right
        steer = 1
    elif classification == 4:  # Both hands
        gas = 0.5
    elif classification == 5:  # Both feet
        brake = 1

    action = np.array([steer, gas, brake], dtype=np.float32)
    return action

