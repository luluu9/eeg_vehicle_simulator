import numpy as np
from mne import concatenate_raws
from mne.io import read_raw_fif
from sklearn.model_selection import StratifiedShuffleSplit
import mne

def split_epochs_into_segments(epochs, seg_length_s, step_s=None):
    sfreq = int(epochs.info['sfreq'])
    seg_samples = int(round(seg_length_s * sfreq))
    step_samples = seg_samples if step_s is None else int(round(step_s * sfreq))

    segments = []
    segment_labels = []
    classes = list(epochs.event_id.keys())

    for cls in classes:
        sub = epochs[cls]  # Epochs for this class
        data = sub.get_data()  # shape (n_epochs, n_ch, n_times)
        for ep_idx in range(data.shape[0]):
            n_times = data.shape[2]
            for start in range(0, n_times - seg_samples + 1, step_samples):
                seg = data[ep_idx, :, start:start + seg_samples]
                segments.append(seg)
                segment_labels.append(cls)

    if len(segments) == 0:
        raise ValueError("No segments produced: check seg_length_s and epoch length.")

    data_new = np.stack(segments)  # (n_new, n_ch, seg_samples)
    info = epochs.info.copy()

    # https://mne.tools/stable/documentation/glossary.html#term-events
    event_id_map = epochs.event_id
    print(event_id_map)
    events = np.c_[np.arange(len(data_new)), np.zeros(len(data_new), int),
    np.array([event_id_map[l] for l in segment_labels], int)]
    new_epochs = mne.EpochsArray(data_new, info, events=events, event_id=event_id_map, tmin=0.0)

    # preserve montage if present
    montage = epochs.get_montage()
    if montage is not None:
        new_epochs.set_montage(montage)

    return new_epochs


def get_freq(epochs, fmin=1., fmax=50.):
    psd = epochs.compute_psd(fmin=fmin, fmax=fmax)
    X = psd.get_data()
    freqs = psd.freqs
    return X, freqs



def get_training_data():

    raw_fnames = [r"Konrad/KONRAD-1_sciskanie_run1_20251202_194514_raw.fif",
                  r"Konrad/KONRAD-2_sciskanie_run1_20251202_203846_raw.fif",
                  r"Konrad/KONRAD-4_ruszanie_run1_20251202_205706_raw.fif"]

    raw = concatenate_raws([read_raw_fif(f, preload=True) for f in raw_fnames])
    eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                    "A16"]
    raw.pick(eeg_channels)
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw.notch_filter(freqs=[50.0])

    konrad_mapping = {
        'A1': 'Cz',
        'A2': 'FCz',
        'A3': 'CP1',
        'A4': 'FC1',
        'A5': 'C1',
        'A6': 'CP3',
        'A7': 'C3',
        'A8': 'FC3',
        'A9': 'C4',
        'A10': 'FC4',
        'A11': 'Pz',
        'A12': 'CP2',
        'A13': 'CP4',
        'A14': 'C2',
        'A15': 'CPz',
        'A16': 'FC2'
    }
    raw.rename_channels(konrad_mapping)

    events, _ = mne.events_from_annotations(raw)
    all_events_id = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both', 5: 'Feet'}
    no_feet = {1: 'Relax', 2: 'Left', 3: 'Right', 4: 'Both'}
    left_right_events_id = {1: 'Relax', 2: 'Left', 3: 'Right'}
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    mapping = {v: k for k, v in all_events_id.items()}
    left_right_mapping = {v: k for k, v in left_right_events_id.items()}
    no_feet_mapping = {v: k for k, v in no_feet.items()}
    task_margin = 0.5
    task_end = 4.5
    reject_criteria = dict(
        eeg=80e-6,  # 80 µV
    )

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
    """
    epochs.plot()
    plt.show()
    """

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, rest_idx = next(sss.split(np.zeros(len(epochs.events[:, -1])), epochs.events[:, -1]))

    train_epochs = epochs[train_idx]
    rest_epochs = epochs[rest_idx]

    print(f"Train epochs: {len(train_epochs)}")
    print(f"Test+Valid epochs: {len(rest_idx)}")

    sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    test_idx, valid_idx = next(sss_valid.split(np.zeros(len(rest_epochs.events[:, -1])), rest_epochs.events[:, -1]))

    test_epochs = rest_epochs[test_idx]
    valid_epochs = rest_epochs[valid_idx]

    print(f"Test epochs: {len(test_epochs)}")
    print(f"Valid epochs: {len(valid_epochs)}")

    segment_length = 2.0
    step = 0.5

    splitted_test_epochs = split_epochs_into_segments(test_epochs, segment_length, step)
    splitted_train_epochs = split_epochs_into_segments(train_epochs, segment_length, step)
    splitted_valid_epochs = split_epochs_into_segments(valid_epochs, segment_length, step)

    y_train = splitted_train_epochs.events[:, -1]
    y_test = splitted_test_epochs.events[:, -1]
    y_valid = splitted_valid_epochs.events[:, -1]

    print("X_train")
    print(len(splitted_train_epochs))
    print("y_train")
    print(len(y_train))
    print("X_test")
    print(len(splitted_test_epochs))
    print("y_test")
    print(len(y_test))
    print("X_valid")
    print(len(splitted_valid_epochs))
    print("y_valid")
    print(len(y_valid))
    return splitted_train_epochs, y_train, splitted_test_epochs, y_test, splitted_valid_epochs, y_valid