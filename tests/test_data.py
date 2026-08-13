import numpy as np
import pytest

from galaxy_classifier.data import (
    compute_normalization_stats,
    normalize_image,
    stratified_split,
    validate_dataset,
)


def test_validation_and_train_only_stats() -> None:
    images = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    images[0] = [1, 2, 3]
    images[1] = [3, 4, 5]
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    validate_dataset(images, labels)
    stats = compute_normalization_stats(images, np.array([0, 1]))
    np.testing.assert_allclose(stats.mean, [2, 3, 4])
    np.testing.assert_allclose(stats.std, [1, 1, 1])
    np.testing.assert_allclose(
        normalize_image(images[0], stats).mean(axis=(0, 1)), [-1, -1, -1]
    )


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    labels = np.repeat(np.arange(4), 20)
    first = stratified_split(labels)
    second = stratified_split(labels)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert sorted(np.concatenate(first).tolist()) == list(range(len(labels)))
    assert [len(part) for part in first] == [56, 12, 12]
    assert np.unique(labels[first[0]], return_counts=True)[1].tolist() == [14] * 4
    for part in first[1:]:
        assert np.unique(labels[part], return_counts=True)[1].tolist() == [3] * 4


@pytest.mark.parametrize(
    "images, labels",
    [
        (np.zeros((2, 3, 3, 3)), np.zeros(2, dtype=np.int64)),
        (np.zeros((2, 256, 256, 3)), np.zeros(3, dtype=np.int64)),
        (np.zeros((2, 256, 256, 3)), np.zeros(2, dtype=np.float32)),
        (np.zeros((2, 256, 256, 3)), np.array([0, 10])),
    ],
)
def test_validation_rejects_bad_input(images: np.ndarray, labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_dataset(images, labels)


def test_download_requires_sha256(tmp_path) -> None:
    from galaxy_classifier.data import download_dataset

    with pytest.raises(ValueError, match="SHA256"):
        download_dataset(tmp_path / "data.h5", url="https://example.invalid/data.h5")
