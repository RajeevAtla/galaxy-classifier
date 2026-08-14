"""Pure Galaxy10 data validation, splitting, and normalization utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple
from urllib.request import urlopen

import numpy as np

NUM_CLASSES = 10
IMAGE_SHAPE = (256, 256, 3)
MODEL_IMAGE_SIZE = 224
ZENODO_URL = "https://zenodo.org/api/records/10845026/files/Galaxy10_DECals.h5/content"
ZENODO_SHA256 = "19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571"


class NormalizationStats(NamedTuple):
    """Per-channel training statistics."""

    mean: np.ndarray
    std: np.ndarray


def validate_dataset(images: np.ndarray, labels: np.ndarray) -> None:
    """Validate Galaxy10 image and label arrays.

    Args:
        images: Array shaped ``(N, 256, 256, 3)``.
        labels: Integer array shaped ``(N,)`` with labels in ``[0, 9]``.

    Raises:
        ValueError: If the arrays do not match the Galaxy10 contract.
    """
    if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(
            f"images must have shape (N, {IMAGE_SHAPE}), got {images.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must be a one-dimensional array matching images")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must have an integer dtype")
    if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
        raise ValueError(f"labels must be in [0, {NUM_CLASSES - 1}]")


def stratified_split(
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic stratified train, validation, and test indices.

    Args:
        labels: One-dimensional integer class labels.
        ratios: Train, validation, and test proportions summing to one.
        seed: Seed used for every class permutation.

    Returns:
        Three disjoint index arrays in train, validation, test order.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("ratios must be three non-negative values summing to one")

    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[], [], []]
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        class_indices = rng.permutation(class_indices)
        exact = np.asarray(ratios) * len(class_indices)
        counts = np.floor(exact).astype(int)
        remainder = len(class_indices) - counts.sum()
        for position in np.argsort(-(exact - counts), kind="stable")[:remainder]:
            counts[position] += 1
        start = 0
        for bucket, count in zip(buckets, counts):
            bucket.extend(class_indices[start : start + count].tolist())
            start += count
    return tuple(np.asarray(bucket, dtype=np.int64) for bucket in buckets)  # type: ignore[return-value]


def compute_normalization_stats(
    images: np.ndarray, train_indices: np.ndarray
) -> NormalizationStats:
    """Compute per-channel mean and standard deviation using training images only.

    Args:
        images: Array shaped ``(N, 256, 256, 3)``.
        train_indices: Indices belonging to the training split.

    Returns:
        Float32 channel means and standard deviations in ``g, r, z`` order.
    """
    if len(train_indices) == 0:
        raise ValueError("train_indices must not be empty")
    selected = np.asarray(images)[np.asarray(train_indices)]
    mean = selected.astype(np.float64).mean(axis=(0, 1, 2), dtype=np.float64)
    std = selected.astype(np.float64).std(axis=(0, 1, 2), dtype=np.float64)
    if np.any(std == 0):
        raise ValueError("training data has a zero-variance channel")
    return NormalizationStats(mean.astype(np.float32), std.astype(np.float32))


def normalize_image(image: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Normalize one image with per-channel statistics."""
    return (np.asarray(image, dtype=np.float32) - stats.mean) / stats.std


def preprocess_image(
    image: np.ndarray,
    stats: NormalizationStats,
    *,
    seed: int = 42,
    training: bool = False,
) -> np.ndarray:
    """Resize, augment, and normalize one Galaxy10 image.

    Args:
        image: ``256x256x3`` image in ``g, r, z`` channel order.
        stats: Training-split channel statistics.
        seed: Per-example deterministic augmentation seed.
        training: Whether to apply geometric augmentation.

    Returns:
        A normalized ``224x224x3`` float32 image.
    """
    image = np.asarray(image)
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"image must have shape {IMAGE_SHAPE}, got {image.shape}")
    if training:
        rng = np.random.default_rng(seed)
        if rng.integers(2):
            image = image[:, ::-1]
        if rng.integers(2):
            image = image[::-1]
        image = np.rot90(image, int(rng.integers(4)))
    positions = np.linspace(0, image.shape[0] - 1, MODEL_IMAGE_SIZE).round().astype(int)
    resized = image[positions][:, positions]
    return normalize_image(resized, stats)


def validate_hdf5_keys(
    metadata: Mapping[str, object], image_key: str, label_key: str
) -> None:
    """Validate that an HDF5 file exposes image and label datasets."""
    missing = [key for key in (image_key, label_key) if key not in metadata]
    if missing:
        raise ValueError(f"HDF5 file is missing dataset(s): {', '.join(missing)}")


def download_dataset(
    destination: str | Path,
    *,
    url: str = ZENODO_URL,
    sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download an HDF5 dataset and verify its SHA256 digest.

    Args:
        destination: Local output path.
        url: Published dataset URL.
        sha256: Expected SHA256 digest. Required to avoid unverified downloads.
        chunk_size: Number of bytes copied per read.

    Returns:
        The downloaded path.

    Raises:
        ValueError: If no SHA256 digest is supplied or the digest mismatches.
    """
    if not sha256:
        raise ValueError("--SHA256 is required for a verified dataset download")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urlopen(url) as response, destination.open("wb") as output:
        while chunk := response.read(chunk_size):
            output.write(chunk)
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(f"SHA256 mismatch: expected {sha256}, got {actual}")
    return destination


def prepare_hdf5(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, object]:
    """Validate Galaxy10 HDF5 and write split and normalization artifacts."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to prepare Galaxy10 data") from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(dataset_path, "r") as handle:
        validate_hdf5_keys(handle, "images", "ans")
        images = handle["images"]
        labels = np.asarray(handle["ans"][:])
        if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
            raise ValueError(
                f"images must have shape (N, {IMAGE_SHAPE}), got {images.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError("labels must be a one-dimensional array matching images")
        if not np.issubdtype(labels.dtype, np.integer) or (
            labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES)
        ):
            raise ValueError(f"labels must be integer values in [0, {NUM_CLASSES - 1}]")
        train, validation, test = stratified_split(labels, seed=seed)
        all_images = np.asarray(images[:])
        count = 0
        total = np.zeros(3, dtype=np.float64)
        total_square = np.zeros(3, dtype=np.float64)
        for start in range(0, len(train), 256):
            batch = all_images[train[start : start + 256]].astype(np.float64)
            count += batch.shape[0] * batch.shape[1] * batch.shape[2]
            total += batch.sum(axis=(0, 1, 2))
            total_square += np.square(batch).sum(axis=(0, 1, 2))
    mean = total / count
    std = np.sqrt(np.maximum(total_square / count - np.square(mean), 0))
    if np.any(std == 0):
        raise ValueError("training data has a zero-variance channel")
    np.savez_compressed(
        output_dir / "split.npz", train=train, validation=validation, test=test
    )
    metadata = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "image_key": "images",
        "label_key": "ans",
        "image_shape": list(IMAGE_SHAPE),
        "num_examples": int(len(labels)),
        "seed": seed,
        "ratios": [0.70, 0.15, 0.15],
        "normalization": {
            "channels": ["g", "r", "z"],
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata
