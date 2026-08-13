"""Pure classification metrics for Galaxy10 evaluation."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class ClassificationMetrics(NamedTuple):
    """Aggregate and per-class classification metrics."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    confusion: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable aggregate and per-class metrics."""
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "f1": self.f1.tolist(),
            "confusion": self.confusion.tolist(),
        }


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, classes: int = 10
) -> np.ndarray:
    """Return a class-by-class count matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")
    if (
        np.any(labels < 0)
        or np.any(labels >= classes)
        or np.any(predictions < 0)
        or np.any(predictions >= classes)
    ):
        raise ValueError("labels and predictions must be valid class indices")
    return np.bincount(
        labels * classes + predictions, minlength=classes * classes
    ).reshape(classes, classes)


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, classes: int = 10
) -> ClassificationMetrics:
    """Compute accuracy, F1 scores, and a normalized confusion matrix."""
    matrix = confusion_matrix(labels, predictions, classes)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(classes), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(classes), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(classes),
        where=(precision + recall) != 0,
    )
    normalized = np.divide(
        matrix,
        support[:, None],
        out=np.zeros_like(matrix, dtype=np.float64),
        where=support[:, None] != 0,
    )
    total = max(1, int(support.sum()))
    return ClassificationMetrics(
        float(true_positive.sum() / total),
        float(f1.mean()),
        float(np.average(f1, weights=support) if support.sum() else 0.0),
        precision,
        recall,
        f1,
        normalized,
    )
