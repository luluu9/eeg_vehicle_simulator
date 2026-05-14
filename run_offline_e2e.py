"""
End-to-end offline evaluation test.

Two modes:
  --mode stream: Continuous sliding-window predictions (original)
  --mode triggered: Marker-triggered predictions (1:1 with real-time)

Triggered mode (default) mirrors exact real-time flow:
  1. Cue marker (1-4) arrives via LSL
  2. Wait 3.5s of EEG after cue onset
  3. MI classifier processes that exact epoch → pushes prediction
  4. Feedback marker (20=correct, 21=error) is sent
  5. ErrP detector classifies post-feedback window
  6. Fusion strategy makes action decision

This uses the exact same code paths as online evaluation.
"""

import argparse
import json
import time
import threading
from pathlib import Path

import numpy as np
import mne
from pylsl import StreamInfo, StreamOutlet, StreamInlet, local_clock, resolve_streams, proc_clocksync

from src.predictor.core.classifiers import StudyMIClassifier, StudyErrPClassifier
from src.predictor.core.errp_detector import ErrPDetector
from src.predictor.core.preprocessor import EEGPreprocessor
from src.predictor.core.lsl_io import PredictionBroadcaster
from src.common.constants import LSLConfig, ErrPConfig
from src.simulator.input_handler import MultiStreamMonitor
from src.simulator.strategies import (
    STUDY_STRATEGIES, BaselineStrategy, StopStrategy, AutocorrectStrategy, REST_ACTION
)


EEG_CHANNELS = [f"A{i}" for i in range(1, 17)]
STUDY_MI_EVENTS = {1: "rest", 2: "left", 3: "right", 4: "forward"}
FEEDBACK_EVENTS = {20: "correct", 21: "error"}


class LSLReplayStreamer:
    """Streams recorded .fif data through LSL at real-time speed (or accelerated)."""

    def __init__(self, raw: mne.io.Raw, speed: float = 1.0, stream_name: str = "Replay-EEG"):
        self.raw = raw
        self.speed = speed
        self.stream_name = stream_name
        self.sfreq = raw.info["sfreq"]
        self.n_channels = len(raw.ch_names)
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _stream_loop(self):
        eeg_info = StreamInfo(
            self.stream_name, "EEG", self.n_channels,
            self.sfreq, "float32", f"{self.stream_name}_src"
        )
        eeg_outlet = StreamOutlet(eeg_info)

        marker_info = StreamInfo(
            f"{self.stream_name}-markers", "Markers", 1,
            0.0, "int32", f"{self.stream_name}_markers"
        )
        marker_outlet = StreamOutlet(marker_info)

        data = self.raw.get_data()  # (ch, samples)
        annotations = list(zip(
            self.raw.annotations.onset,
            self.raw.annotations.description,
        ))
        annot_idx = 0

        chunk_size = 32
        chunk_duration = chunk_size / self.sfreq
        total_samples = data.shape[1]
        sample_idx = 0
        start_time = local_clock()
        stream_time = 0.0

        print(f"[Replay] Streaming {total_samples / self.sfreq:.1f}s of data ({self.n_channels}ch @ {self.sfreq}Hz)")

        while self._running and sample_idx < total_samples:
            end_idx = min(sample_idx + chunk_size, total_samples)
            chunk = data[:, sample_idx:end_idx].T  # (samples, ch)
            chunk = np.ascontiguousarray(chunk, dtype=np.float32)
            eeg_outlet.push_chunk(chunk.tolist())

            stream_time = sample_idx / self.sfreq
            while annot_idx < len(annotations):
                onset, desc = annotations[annot_idx]
                if onset <= stream_time:
                    try:
                        marker_val = int(desc)
                        marker_outlet.push_sample([marker_val])
                    except ValueError:
                        pass
                    annot_idx += 1
                else:
                    break

            sample_idx = end_idx

            expected_real_time = stream_time / self.speed
            elapsed = local_clock() - start_time
            sleep_time = expected_real_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[Replay] Stream complete ({stream_time:.1f}s)")
        self._running = False


class MIPredictorThread:
    """Runs MI classification on incoming EEG, broadcasts probabilities."""

    def __init__(self, classifier: StudyMIClassifier, window_sec: float = 3.5):
        self.classifier = classifier
        self.preprocessor = EEGPreprocessor(target_srate=256.0)
        self.window_sec = window_sec
        self.broadcaster = PredictionBroadcaster(classifier.name, channel_count=4)
        self._running = False
        self._thread: threading.Thread | None = None
        self._data_buffer: list = []
        self._buffer_lock = threading.Lock()
        self._srate = 0.0

    def start(self, eeg_inlet):
        self._srate = eeg_inlet.info().nominal_srate()
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(eeg_inlet,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self, inlet):
        from pylsl import StreamInlet
        interval = 1.0
        last_pred = time.monotonic()
        buffer = []
        max_samples = int(self._srate * (self.window_sec * 3))

        while self._running:
            chunk, _ = inlet.pull_chunk(timeout=0.0)
            if chunk:
                buffer.extend(chunk)
                while len(buffer) > max_samples:
                    buffer.pop(0)

            now = time.monotonic()
            if now - last_pred >= interval:
                last_pred = now
                needed = int(self._srate * self.window_sec * 2)
                if len(buffer) >= needed:
                    data = np.array(buffer[-needed:]).T  # (ch, samples)
                    try:
                        processed = self.preprocessor.process(data, self._srate)
                        window_samples = int(self.window_sec * self.preprocessor.target_srate)
                        if processed.shape[1] >= window_samples:
                            input_slice = processed[:, -window_samples:]
                            probs = self.classifier.predict_proba(input_slice, self.preprocessor.target_srate)
                            self.broadcaster.push_prediction(probs)
                    except Exception as e:
                        print(f"[MI Pred] Error: {e}")

            time.sleep(0.01)


# Classifier output indices (from StudyMIClassifier: model classes 1-4 → probs[0-3])
CLASS_NAMES = {0: "REST", 1: "LEFT", 2: "RIGHT", 3: "FORWARD"}
# Annotation markers → classifier index
MARKER_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3}

ACTION_TO_CLASS = {
    (0.0, 0.0, 0.0): 0,    # REST
    (-0.5, 0.3, 0.0): 1,   # LEFT
    (0.5, 0.3, 0.0): 2,    # RIGHT
    (0.0, 0.3, 0.0): 3,    # FORWARD
}

MI_MARKERS = {1, 2, 3, 4}


class MarkerTriggeredPredictor:
    """
    MI predictor triggered by cue markers — mirrors exact real-time flow.

    On cue marker (1-4):
      1. Buffer EEG for window_sec seconds after marker onset
      2. Classify that exact epoch
      3. Broadcast MI prediction
      4. Send feedback marker (20=correct, 21=error)
    """

    def __init__(self, classifier: StudyMIClassifier, window_sec: float = 3.5):
        self.classifier = classifier
        self.preprocessor = EEGPreprocessor(target_srate=256.0)
        self.window_sec = window_sec
        self.broadcaster = PredictionBroadcaster(classifier.name, channel_count=4)
        self._running = False
        self._thread: threading.Thread | None = None
        self._srate = 0.0
        self.trial_results: list[dict] = []

    def start(self, eeg_inlet: StreamInlet, marker_inlet: StreamInlet,
              feedback_outlet: StreamOutlet):
        self._srate = eeg_inlet.info().nominal_srate()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(eeg_inlet, marker_inlet, feedback_outlet),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self, eeg_inlet: StreamInlet, marker_inlet: StreamInlet,
              feedback_outlet: StreamOutlet):
        eeg_buffer = []
        eeg_ts_buffer = []
        max_buffer = int(self._srate * 30)

        pending_trial: dict | None = None
        window_samples = int(self._srate * self.window_sec)

        while self._running:
            # Pull EEG
            chunk, ts = eeg_inlet.pull_chunk(timeout=0.0)
            if ts:
                eeg_buffer.extend(chunk)
                eeg_ts_buffer.extend(ts)
                while len(eeg_buffer) > max_buffer:
                    eeg_buffer.pop(0)
                    eeg_ts_buffer.pop(0)

            # Check for cue markers
            if not pending_trial:
                sample, marker_ts = marker_inlet.pull_sample(timeout=0.0)
                if sample and marker_ts:
                    marker_val = int(round(sample[0]))
                    if marker_val in MI_MARKERS:
                        pending_trial = {
                            "marker": marker_val,
                            "onset_ts": marker_ts,
                        }

            # If we have a pending trial, check if enough EEG has been buffered
            if pending_trial and eeg_ts_buffer:
                onset = pending_trial["onset_ts"]
                t_end = onset + self.window_sec
                latest_ts = eeg_ts_buffer[-1]

                if latest_ts >= t_end:
                    # Extract epoch [onset, onset + window_sec]
                    ts_arr = np.array(eeg_ts_buffer)
                    mask = (ts_arr >= onset) & (ts_arr <= t_end)

                    if mask.sum() >= window_samples * 0.8:
                        data = np.array(eeg_buffer)[mask].T  # (ch, samples)
                        true_class = pending_trial["marker"]

                        try:
                            processed = self.preprocessor.process(data, self._srate)
                            target_samples = int(self.window_sec * self.preprocessor.target_srate)
                            if processed.shape[1] >= target_samples:
                                input_slice = processed[:, :target_samples]
                            else:
                                input_slice = processed

                            probs = self.classifier.predict_proba(
                                input_slice, self.preprocessor.target_srate
                            )
                            predicted_class = int(np.argmax(probs))
                            self.broadcaster.push_prediction(probs)

                            true_idx = MARKER_TO_IDX[true_class]
                            is_correct = (predicted_class == true_idx)
                            feedback_marker = 20 if is_correct else 21
                            feedback_outlet.push_sample([feedback_marker])

                            self.trial_results.append({
                                "true_class": true_idx,
                                "predicted_class": predicted_class,
                                "probs": probs.tolist(),
                                "correct": is_correct,
                            })
                            n = len(self.trial_results)
                            acc = sum(t["correct"] for t in self.trial_results) / n
                            print(
                                f"[Trial {n:3d}] true={CLASS_NAMES[true_idx]:>7s} "
                                f"pred={CLASS_NAMES[predicted_class]:>7s} "
                                f"{'✓' if is_correct else '✗'} "
                                f"conf={probs[predicted_class]:.2f} "
                                f"acc={acc:.1%}"
                            )
                        except Exception as e:
                            print(f"[MI Triggered] Error classifying trial: {type(e).__name__}: {e}")

                    pending_trial = None

            time.sleep(0.005)


class ActionLogger:
    """Collects strategy decisions and computes evaluation metrics."""

    def __init__(self, annotations: list[tuple[float, str]], speed: float):
        self.decisions: list[dict] = []
        self._start_time = 0.0
        self._speed = speed
        self._ground_truth = self._build_ground_truth(annotations)

    def _build_ground_truth(self, annotations: list[tuple[float, str]]) -> list[tuple[float, float, int]]:
        """Build (start, end, class_idx) intervals from annotations."""
        intervals = []
        for onset, desc in annotations:
            try:
                marker = int(desc)
                if marker in MARKER_TO_IDX:
                    intervals.append((onset, onset + 3.5, MARKER_TO_IDX[marker]))
            except ValueError:
                continue
        return intervals

    def _get_true_class(self, real_time: float) -> int | None:
        """Map real elapsed time → simulated stream time → ground truth class."""
        stream_time = real_time * self._speed
        for start, end, cls in self._ground_truth:
            if start <= stream_time <= end:
                return cls
        return None

    def start(self):
        self._start_time = time.monotonic()

    def log(self, action: np.ndarray, mi_probs: dict, errp_probs: dict):
        t = time.monotonic() - self._start_time
        action_tuple = tuple(round(float(x), 1) for x in action)
        predicted_class = ACTION_TO_CLASS.get(action_tuple, 0)

        mi_stream = next(iter(mi_probs), None)
        raw_probs = mi_probs[mi_stream].tolist() if mi_stream else [0.0] * 4
        mi_predicted = int(np.argmax(raw_probs))

        errp_stream = next(iter(errp_probs), None)
        errp_vals = errp_probs[errp_stream].tolist() if errp_stream else [0.5, 0.5]

        true_class = self._get_true_class(t)

        self.decisions.append({
            "t": t,
            "action": action.tolist(),
            "predicted_class": predicted_class,
            "mi_predicted": mi_predicted,
            "mi_probs": raw_probs,
            "errp_probs": errp_vals,
            "true_class": true_class,
        })

    def summary(self) -> dict:
        if not self.decisions:
            return {"n_decisions": 0}

        n = len(self.decisions)
        duration = self.decisions[-1]["t"]

        action_classes = [d["predicted_class"] for d in self.decisions]
        mi_classes = [d["mi_predicted"] for d in self.decisions]
        true_classes = [d["true_class"] for d in self.decisions]

        class_counts = {name: action_classes.count(i) for i, name in CLASS_NAMES.items()}
        mi_class_counts = {name: mi_classes.count(i) for i, name in CLASS_NAMES.items()}

        labeled = [(d["mi_predicted"], d["true_class"]) for d in self.decisions if d["true_class"] is not None]
        mi_accuracy = None
        if labeled:
            correct = sum(1 for pred, true in labeled if pred == true)
            mi_accuracy = correct / len(labeled)

        action_labeled = [(d["predicted_class"], d["true_class"]) for d in self.decisions if d["true_class"] is not None]
        action_accuracy = None
        if action_labeled:
            correct = sum(1 for pred, true in action_labeled if pred == true)
            action_accuracy = correct / len(action_labeled)

        errp_detections = [d for d in self.decisions if d["errp_probs"][1] >= 0.5]

        avg_confidence = float(np.mean([max(d["mi_probs"]) for d in self.decisions]))

        return {
            "n_decisions": n,
            "duration_s": round(duration, 2),
            "action_distribution": class_counts,
            "mi_class_distribution": mi_class_counts,
            "mi_accuracy_vs_ground_truth": round(mi_accuracy, 3) if mi_accuracy is not None else None,
            "action_accuracy_vs_ground_truth": round(action_accuracy, 3) if action_accuracy is not None else None,
            "n_labeled_decisions": len(labeled),
            "n_errp_error_detections": len(errp_detections),
            "avg_mi_confidence": round(avg_confidence, 3),
        }


def run_triggered_evaluation(
    data_path: str,
    mi_model_path: str,
    errp_model_path: str,
    strategy_name: str = "baseline",
    speed: float = 5.0,
) -> dict:
    """
    Marker-triggered evaluation — 1:1 with real-time.

    Flow per trial:
      Cue marker → 3.5s EEG epoch → MI classify → feedback marker → ErrP classify → strategy decision
    """
    print(f"\n{'='*60}")
    print(f"TRIGGERED E2E EVALUATION (1:1 with real-time)")
    print(f"  Data:     {data_path}")
    print(f"  MI Model: {mi_model_path}")
    print(f"  ErrP:     {errp_model_path}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Speed:    {speed}x")
    print(f"{'='*60}\n")

    raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)

    mi_clf = StudyMIClassifier(mi_model_path)
    errp_clf = StudyErrPClassifier(errp_model_path)

    streamer = LSLReplayStreamer(raw, speed=speed)

    # Feedback outlet — predictor sends 20/21 after MI classification
    feedback_info = StreamInfo(
        f"{streamer.stream_name}-feedback", "Markers", 1,
        0.0, "int32", f"{streamer.stream_name}_feedback"
    )
    feedback_outlet = StreamOutlet(feedback_info)

    errp_detector = ErrPDetector(errp_clf)
    monitor = MultiStreamMonitor()
    strategy = STUDY_STRATEGIES[strategy_name]()

    streamer.start()
    time.sleep(1.0)

    # ErrP listens to the SAME EEG stream but feedback markers from our feedback outlet
    errp_detector.start(
        eeg_stream_name=streamer.stream_name,
        marker_stream_name=f"{streamer.stream_name}-feedback",
    )
    monitor.start()

    # Connect MI predictor to EEG + markers
    eeg_inlet = None
    marker_inlet = None
    for _ in range(20):
        streams = resolve_streams(wait_time=0.5)
        for s in streams:
            if s.name() == streamer.stream_name and s.type() == "EEG" and not eeg_inlet:
                eeg_inlet = StreamInlet(s, processing_flags=proc_clocksync)
            if s.name() == f"{streamer.stream_name}-markers" and not marker_inlet:
                marker_inlet = StreamInlet(s, processing_flags=proc_clocksync)
        if eeg_inlet and marker_inlet:
            break
        time.sleep(0.2)

    if not eeg_inlet or not marker_inlet:
        print("ERROR: Could not find required streams")
        streamer.stop()
        return {"error": "streams_not_found"}

    mi_predictor = MarkerTriggeredPredictor(mi_clf, window_sec=3.5)
    mi_predictor.start(eeg_inlet, marker_inlet, feedback_outlet)

    print("[E2E] All components running. Waiting for trials...\n")

    # Wait for replay to finish
    while streamer._running:
        time.sleep(0.1)

    # Give extra time for last trial to complete processing
    time.sleep(3.0 / speed + 1.0)

    print("\n[E2E] Stopping components...")
    mi_predictor.stop()
    errp_detector.stop()
    monitor.stop()
    streamer.stop()

    # Build results from trial_results
    trials = mi_predictor.trial_results
    if not trials:
        return {"error": "no_trials_processed", "strategy": strategy_name}

    n_trials = len(trials)
    n_correct = sum(t["correct"] for t in trials)
    mi_accuracy = n_correct / n_trials

    per_class_correct = {}
    per_class_total = {}
    for t in trials:
        cls_name = CLASS_NAMES[t["true_class"]]
        per_class_total[cls_name] = per_class_total.get(cls_name, 0) + 1
        if t["correct"]:
            per_class_correct[cls_name] = per_class_correct.get(cls_name, 0) + 1

    per_class_accuracy = {
        cls: per_class_correct.get(cls, 0) / per_class_total[cls]
        for cls in per_class_total
    }

    avg_confidence = float(np.mean([max(t["probs"]) for t in trials]))

    # Confusion matrix counts
    confusion = {}
    for t in trials:
        true_name = CLASS_NAMES[t["true_class"]]
        pred_name = CLASS_NAMES[t["predicted_class"]]
        key = f"{true_name}→{pred_name}"
        confusion[key] = confusion.get(key, 0) + 1

    results = {
        "mode": "triggered",
        "strategy": strategy_name,
        "n_trials": n_trials,
        "mi_accuracy": round(mi_accuracy, 3),
        "per_class_accuracy": {k: round(v, 3) for k, v in per_class_accuracy.items()},
        "per_class_n_trials": per_class_total,
        "avg_confidence": round(avg_confidence, 3),
        "confusion": confusion,
    }

    if hasattr(strategy, 'correction_count'):
        results["errp_corrections"] = strategy.correction_count

    print(f"\n{'='*60}")
    print(f"RESULTS (marker-triggered, {n_trials} trials)")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*60}")
    return results


def run_offline_evaluation(
    data_path: str,
    mi_model_path: str,
    errp_model_path: str,
    strategy_name: str = "baseline",
    speed: float = 5.0,
    duration: float | None = None,
) -> dict:
    """Run full pipeline offline with recorded data."""
    print(f"\n{'='*60}")
    print(f"OFFLINE E2E EVALUATION")
    print(f"  Data:     {data_path}")
    print(f"  MI Model: {mi_model_path}")
    print(f"  ErrP:     {errp_model_path}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Speed:    {speed}x")
    print(f"{'='*60}\n")

    raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)

    mi_clf = StudyMIClassifier(mi_model_path)
    errp_clf = StudyErrPClassifier(errp_model_path)

    annotations = list(zip(raw.annotations.onset, raw.annotations.description))

    streamer = LSLReplayStreamer(raw, speed=speed)
    errp_detector = ErrPDetector(errp_clf)
    monitor = MultiStreamMonitor()
    strategy = STUDY_STRATEGIES[strategy_name]()
    logger = ActionLogger(annotations, speed)

    streamer.start()
    time.sleep(1.0)

    errp_detector.start(
        eeg_stream_name=streamer.stream_name,
        marker_stream_name=f"{streamer.stream_name}-markers",
    )
    monitor.start()

    from pylsl import resolve_streams, StreamInlet, proc_clocksync
    mi_inlet = None
    for _ in range(20):
        streams = resolve_streams(wait_time=0.5)
        for s in streams:
            if s.name() == streamer.stream_name and s.type() == "EEG":
                mi_inlet = StreamInlet(s, processing_flags=proc_clocksync)
                break
        if mi_inlet:
            break
        time.sleep(0.2)

    if not mi_inlet:
        print("ERROR: Could not find EEG stream for MI predictor")
        streamer.stop()
        return {"error": "no_eeg_stream"}

    mi_predictor = MIPredictorThread(mi_clf, window_sec=3.5)
    mi_predictor.start(mi_inlet)

    print("[E2E] All components running. Processing...")
    logger.start()

    max_duration = duration or (raw.times[-1] / speed + 5)
    start = time.monotonic()
    decision_interval = 1.0
    last_decision = start

    while time.monotonic() - start < max_duration and streamer._running:
        now = time.monotonic()
        if now - last_decision >= decision_interval:
            last_decision = now
            mi_probs = monitor.get_probabilities()
            errp_probs = monitor.get_errp()
            stream = next(iter(mi_probs), None)
            if stream:
                action = strategy.compute(mi_probs, stream, errp_probs)
                logger.log(action, mi_probs, errp_probs)
        time.sleep(0.05)

    print("\n[E2E] Stopping components...")
    mi_predictor.stop()
    errp_detector.stop()
    monitor.stop()
    streamer.stop()

    results = logger.summary()
    results["strategy"] = strategy_name
    if hasattr(strategy, 'correction_count'):
        results["corrections"] = strategy.correction_count
    if hasattr(strategy, 'threshold'):
        results["mi_threshold"] = strategy.threshold
    if hasattr(strategy, 'errp_threshold'):
        results["errp_threshold"] = strategy.errp_threshold

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*60}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline end-to-end evaluation")
    parser.add_argument("data_file", help=".fif recording to replay")
    parser.add_argument("--mi-model", required=True, help="Path to trained MI model")
    parser.add_argument("--errp-model", required=True, help="Path to trained ErrP model")
    parser.add_argument("--strategy", choices=list(STUDY_STRATEGIES.keys()), default="baseline")
    parser.add_argument("--speed", type=float, default=5.0, help="Replay speed multiplier")
    parser.add_argument("--duration", type=float, default=None, help="Max eval duration (stream mode only)")
    parser.add_argument("--mode", choices=["triggered", "stream"], default="triggered",
                        help="triggered=marker-aligned (1:1 real-time), stream=continuous sliding window")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save results JSON to file")
    args = parser.parse_args()

    if args.mode == "triggered":
        results = run_triggered_evaluation(
            args.data_file, args.mi_model, args.errp_model,
            strategy_name=args.strategy, speed=args.speed,
        )
    else:
        results = run_offline_evaluation(
            args.data_file, args.mi_model, args.errp_model,
            strategy_name=args.strategy, speed=args.speed, duration=args.duration,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
