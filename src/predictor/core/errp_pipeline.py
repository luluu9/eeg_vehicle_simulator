import argparse
import json
from pathlib import Path

import numpy as np
import mne
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

CHANNEL_MAPPING = {
    'A1': 'Cz', 'A2': 'FCz', 'A3': 'CP1', 'A4': 'FC1',
    'A5': 'C1', 'A6': 'CP3', 'A7': 'C3', 'A8': 'FC3',
    'A9': 'C4', 'A10': 'FC4', 'A11': 'Pz', 'A12': 'CP2',
    'A13': 'CP4', 'A14': 'C2', 'A15': 'CPz', 'A16': 'FC2',
}

FEEDBACK_MARKERS = {
    "correct": 20,
    "error": 21,
}

EEG_CHANNELS = [f"A{i}" for i in range(1, 17)]


def load_errp_epochs(
    file_paths: list[Path],
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> mne.Epochs:
    all_epochs = []
    for path in file_paths:
        raw = mne.io.read_raw_fif(str(path), preload=True)
        raw.pick(picks=EEG_CHANNELS)
        raw.info['dev_head_t'] = None
        raw.set_eeg_reference(ref_channels='average', projection=False)
        raw.resample(sfreq=256)
        raw.filter(l_freq=1.0, h_freq=10.0, fir_design='firwin')
        raw.rename_channels(CHANNEL_MAPPING)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        marker_to_id = {str(v): v for v in FEEDBACK_MARKERS.values()}
        events, event_id = mne.events_from_annotations(raw, event_id=marker_to_id)
        event_id_named = {
            name: event_id[str(code)]
            for name, code in FEEDBACK_MARKERS.items()
            if str(code) in event_id
        }

        epochs = mne.Epochs(
            raw, events, event_id_named,
            tmin=tmin, tmax=tmax,
            baseline=(tmin, 0),
            preload=True,
            reject=dict(eeg=80e-6),
        )
        all_epochs.append(epochs)

    return mne.concatenate_epochs(all_epochs, on_mismatch='warn')


def build_pipeline():
    return make_pipeline(
        XdawnCovariances(nfilter=6, estimator='oas', xdawn_estimator='oas'),
        TangentSpace(metric='riemann'),
        SVC(
            class_weight='balanced',
            kernel='linear',
            C=0.01,
            probability=True,
            random_state=42,
        ),
    )


def train_and_evaluate(
    data_files: list[Path],
    output_path: Path,
    tmin: float = -0.2,
    tmax: float = 0.8,
    n_folds: int = 5,
    test_ratio: float = 0.3,
) -> dict:
    epochs = load_errp_epochs(data_files, tmin=tmin, tmax=tmax)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy (train): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Test Accuracy (chronological): {test_acc:.4f}")
    print(classification_report(y_test, y_pred))

    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, output_path)
    print(f"Final model (trained on all {len(y)} epochs) saved to {output_path}")

    unique, counts = np.unique(y, return_counts=True)
    return {
        "n_epochs_total": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "test_accuracy": float(test_acc),
        "test_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary ErrP model")
    parser.add_argument("data_files", nargs="+", help=".fif files from collector")
    parser.add_argument("-o", "--output", default="models/errp_xdawn.joblib")
    parser.add_argument("--tmin", type=float, default=-0.2)
    parser.add_argument("--tmax", type=float, default=0.8)
    args = parser.parse_args()

    files = [Path(f) for f in args.data_files]
    results = train_and_evaluate(files, Path(args.output), tmin=args.tmin, tmax=args.tmax)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
