import mne
import numpy as np

mapping = {
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

def merge_epochs(*epochs):
    if not epochs:
        raise ValueError("no epochs provided")
    parts = []
    for e in epochs:
        if isinstance(e, (list, tuple)):
            parts.extend(e)
        else:
            parts.append(e)
    if len(parts) == 1:
        return parts[0].copy()
    return mne.concatenate_epochs(parts)


def split_annotated_into_segments(file_paths, segment_length_s=2.0, step_s=1.0):
    all_epochs = []
    
    for data_path in file_paths:
        filepath = str(data_path)
        raw = mne.io.read_raw_fif(filepath, preload=True)
        eeg_channels = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]
        raw.pick(picks=eeg_channels)
        raw.info['dev_head_t'] = None
        #raw.set_eeg_reference(ref_channels='average', projection=False)
        raw.resample(sfreq=256)
        raw.filter(l_freq=8.0, h_freq=32.0, fir_design='firwin')
        raw.notch_filter(freqs=[50.0])
           
        raw.rename_channels(mapping)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        events_real = {"relax": 1, "left_hand": 2, "right_hand": 3, "both_hands": 4, "both_feets": 5}
        events_predicted = {"relax_predicted": 11, "left_hand_predicted": 12, "right_hand_predicted": 13, "both_hands_predicted": 14, "both_feets_predicted": 15}
        classification_result = {"correct": 20, "incorrect": 21}
        all_possible_events_id = {**events_real, **events_predicted, **classification_result}
        
        # to make sure that event ids are consistent across recordings
        description_code_to_consistent_id = {str(v): v for v in all_possible_events_id.values()}
        all_events, all_events_id = mne.events_from_annotations(raw, event_id=description_code_to_consistent_id)

        # filter only events that are really in data (all_events_id)
        # all_events_id has string keys, so we need to convert, and event values is continously increasing
        all_events_id_renamed = {k: all_events_id[str(v)] for k, v in all_possible_events_id.items() if str(v) in all_events_id.keys()}

        # events cant happen concurrently, so remove one of them (currently drop exact classification and store only result (correct/incorrect))
        # probably we can merge it in MNE fashion ("[status]/[classification_result]"), but for now just drop
        for events_pred in events_predicted.keys():
            all_events_id_renamed.pop(events_pred, None)
        all_events = np.array([e for e in all_events if e[2] in all_events_id_renamed.values()])

        reject_criteria = dict(
            eeg=80e-6,  # 80 µV
        ) 

        task_margin = 1.0 # event is when cue is shown
        task_duration = 5.0
        task_end = task_margin + task_duration
        epochs = mne.Epochs(
            raw=raw,
            events=all_events,
            event_id=all_events_id_renamed,
            baseline=None,
            tmin=task_margin,
            tmax=task_end,
            preload=True,
            reject=reject_criteria
        )

        all_epochs.append(split_epochs_into_segments(epochs, segment_length_s, step_s))
    
    return mne.concatenate_epochs(all_epochs, on_mismatch='warn')
    
