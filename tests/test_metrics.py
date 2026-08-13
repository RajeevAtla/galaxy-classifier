import numpy as np
import pytest

from galaxy_classifier.metrics import classification_metrics, confusion_matrix


def test_metrics_and_normalized_confusion() -> None:
    result = classification_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), classes=2)
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.macro_f1 == pytest.approx((2 / 3 + 2 / 3) / 2)
    np.testing.assert_allclose(result.confusion.sum(axis=1), 1)
    assert result.as_dict()["accuracy"] == result.accuracy


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0]), np.array([0, 1]))
    with pytest.raises(ValueError):
        confusion_matrix(np.array([2]), np.array([0]), classes=2)


def test_metrics_handles_empty_support() -> None:
    result = classification_metrics(np.array([0]), np.array([0]), classes=3)
    assert result.weighted_f1 == pytest.approx(1)
    assert result.f1[2] == 0
