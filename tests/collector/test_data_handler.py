import os
import tempfile

import mne
import numpy as np

from src.collector.data_handler import DataLogger


class TestDataLogger:
    def _make_logger_with_data(self, n_channels=4, sfreq=256, n_samples=512):
        logger = DataLogger(save_dir=tempfile.mkdtemp())
        logger.info = mne.create_info(
            ch_names=[f"EEG_{i:03d}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types="eeg",
        )
        data = np.random.randn(n_samples, n_channels)
        timestamps = np.arange(n_samples) / sfreq
        logger.add_data(data, timestamps)
        return logger, timestamps

    def test_add_event_stores_correctly(self):
        logger = DataLogger()
        logger.add_event(1.0, 2)
        logger.add_event(2.5, 3)
        assert len(logger.events) == 2
        assert logger.events[0] == (1.0, 2)
        assert logger.events[1] == (2.5, 3)

    def test_remove_last_event(self):
        logger = DataLogger()
        logger.add_event(1.0, 2)
        logger.add_event(2.0, 3)
        logger.remove_last_event()
        assert len(logger.events) == 1
        assert logger.events[0] == (1.0, 2)

    def test_remove_last_event_empty(self):
        logger = DataLogger()
        logger.remove_last_event()
        assert len(logger.events) == 0

    def test_add_data_accumulates(self):
        logger = DataLogger()
        d1 = np.ones((10, 4))
        d2 = np.ones((5, 4))
        logger.add_data(d1, np.arange(10))
        logger.add_data(d2, np.arange(5))
        assert len(logger.raw_data) == 2

    def test_clear(self):
        logger = DataLogger()
        logger.add_data(np.ones((10, 4)), np.arange(10))
        logger.add_event(1.0, 2)
        logger.clear()
        assert len(logger.raw_data) == 0
        assert len(logger.events) == 0

    def test_save_creates_fif_file(self):
        logger, timestamps = self._make_logger_with_data()
        filepath = logger.save("test_subject", 1)
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith("_raw.fif")
        os.remove(filepath)

    def test_save_no_data_returns_none(self):
        logger = DataLogger(save_dir=tempfile.mkdtemp())
        assert logger.save("test", 1) is None

    def test_save_preserves_annotations(self):
        logger, timestamps = self._make_logger_with_data(sfreq=256, n_samples=1024)
        logger.add_event(0.5, 2)
        logger.add_event(1.0, 20)
        logger.add_event(2.0, 3)

        filepath = logger.save("test_subject", 1)
        raw = mne.io.read_raw_fif(filepath, verbose=False)
        annotations = raw.annotations

        assert len(annotations) >= 3
        descriptions = list(annotations.description)
        assert "2" in descriptions
        assert "20" in descriptions
        assert "3" in descriptions

        os.remove(filepath)

    def test_save_filename_format(self):
        logger, _ = self._make_logger_with_data()
        filepath = logger.save("SUBJ01", 3)
        basename = os.path.basename(filepath)
        assert basename.startswith("SUBJ01_run3_")
        assert basename.endswith("_raw.fif")
        os.remove(filepath)
