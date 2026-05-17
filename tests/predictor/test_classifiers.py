import numpy as np
import pytest

from src.predictor.core.classifiers import (
    GroundTruthClassifier,
    MockClassifier,
    _map_legacy_probs_to_study,
    _map_model_probs_to_study,
)


class TestLegacyProbabilityMapping:
    def test_drops_both_hands_and_renormalizes(self):
        classes = np.array([1, 2, 3, 4, 5])
        probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25])

        mapped = _map_legacy_probs_to_study(classes, probs)

        expected = np.array([0.1, 0.2, 0.3, 0.25]) / 0.85
        np.testing.assert_allclose(mapped, expected)
        assert mapped.shape == (4,)
        assert mapped.sum() == pytest.approx(1.0)

    def test_ignores_missing_and_unknown_classes(self):
        classes = np.array([2, 5, 99])
        probs = np.array([0.4, 0.6, 0.9])

        mapped = _map_legacy_probs_to_study(classes, probs)

        np.testing.assert_allclose(mapped, [0.0, 0.4, 0.0, 0.6])

    def test_returns_zeros_when_only_removed_class_is_present(self):
        classes = np.array([4])
        probs = np.array([1.0])

        mapped = _map_legacy_probs_to_study(classes, probs)

        np.testing.assert_array_equal(mapped, np.zeros(4))


class TestModelProbabilityMapping:
    def test_preserves_direct_study_class_indices(self):
        classes = np.array([1, 2, 3, 4])
        probs = np.array([0.1, 0.2, 0.3, 0.4])

        mapped = _map_model_probs_to_study(classes, probs)

        np.testing.assert_allclose(mapped, probs)


class TestGroundTruthClassifier:
    @pytest.mark.parametrize(
        ("legacy_index", "expected"),
        [
            (0, [1.0, 0.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0, 0.0]),
            (2, [0.0, 0.0, 1.0, 0.0]),
            (4, [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_maps_supported_legacy_labels_to_study_outputs(self, legacy_index, expected):
        clf = GroundTruthClassifier()
        clf.latest_label_idx = legacy_index

        probs = clf.predict_proba(None, 256.0)

        np.testing.assert_array_equal(probs, expected)

    def test_both_hands_label_becomes_zero_vector(self):
        clf = GroundTruthClassifier()
        clf.latest_label_idx = 3

        probs = clf.predict_proba(None, 256.0)

        np.testing.assert_array_equal(probs, np.zeros(4))

    def test_mock_classifier_uses_four_output_classes(self):
        probs = MockClassifier().predict_proba(np.zeros((2, 2)), 256.0)

        assert probs.shape == (4,)
        assert probs.sum() == pytest.approx(1.0)