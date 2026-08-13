import hashlib
import io
import json

import h5py
import numpy as np
import pytest

from galaxy_classifier.data import (
    NormalizationStats,
    compute_normalization_stats,
    download_dataset,
    prepare_hdf5,
    preprocess_image,
    stratified_split,
    validate_hdf5_keys,
)


def test_data_error_branches_and_preprocessing():
    images = np.arange(2 * 256 * 256 * 3, dtype=np.uint8).reshape(2, 256, 256, 3)
    with pytest.raises(ValueError, match="one-dimensional"):
        stratified_split(np.zeros((2, 1), dtype=int))
    with pytest.raises(ValueError, match="ratios"):
        stratified_split(np.zeros(2, dtype=int), (1, 0, 0, 0))
    with pytest.raises(ValueError, match="empty"):
        compute_normalization_stats(images, np.array([], dtype=int))
    with pytest.raises(ValueError, match="zero-variance"):
        compute_normalization_stats(np.zeros_like(images), np.array([0]))
    stats = NormalizationStats(np.zeros(3, dtype=np.float32), np.ones(3))
    with pytest.raises(ValueError, match="image must"):
        preprocess_image(np.zeros((1, 1, 3)), stats)
    plain = preprocess_image(images[0], stats)
    augmented = preprocess_image(images[0], stats, seed=1, training=True)
    assert plain.shape == augmented.shape == (224, 224, 3)


def test_hdf5_keys_and_download(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="missing"):
        validate_hdf5_keys({"images": object()}, "images", "ans")
    payload = b"verified"

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr("galaxy_classifier.data.urlopen", lambda url: Response(payload))
    digest = hashlib.sha256(payload).hexdigest()
    destination = download_dataset(tmp_path / "nested" / "data", url="x", sha256=digest)
    assert destination.read_bytes() == payload
    with pytest.raises(ValueError, match="mismatch"):
        download_dataset(tmp_path / "bad", url="x", sha256="0" * 64)
    assert not (tmp_path / "bad").exists()


def test_prepare_hdf5_writes_artifacts(tmp_path):
    dataset = tmp_path / "data.h5"
    images = np.zeros((20, 256, 256, 3), dtype=np.uint8)
    images[:, 0, 0, :] = np.arange(20)[:, None]
    labels = np.repeat(np.arange(10), 2)
    with h5py.File(dataset, "w") as handle:
        handle.create_dataset("images", data=images)
        handle.create_dataset("ans", data=labels)
    metadata = prepare_hdf5(dataset, tmp_path / "artifacts")
    assert metadata["num_examples"] == 20
    assert (tmp_path / "artifacts" / "split.npz").exists()
    assert (
        json.loads((tmp_path / "artifacts" / "metadata.json").read_text())["seed"] == 42
    )


@pytest.mark.parametrize("kind", ["missing", "shape", "labels", "zero"])
def test_prepare_hdf5_rejects_invalid_files(tmp_path, kind):
    path = tmp_path / f"{kind}.h5"
    with h5py.File(path, "w") as handle:
        if kind != "missing":
            shape = (2, 1, 1, 3) if kind == "shape" else (20, 256, 256, 3)
            handle.create_dataset("images", data=np.zeros(shape, dtype=np.uint8))
        if kind == "labels":
            handle.create_dataset("ans", data=np.array([0]))
        elif kind == "zero":
            handle.create_dataset("ans", data=np.repeat(np.arange(10), 2))
            handle["images"][:] = 0
        elif kind != "missing":
            handle.create_dataset("ans", data=np.repeat(np.arange(10), 2))
    with pytest.raises((ValueError, KeyError)):
        prepare_hdf5(path, tmp_path / kind)


def test_preprocess_reaches_flip_branches(monkeypatch):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    stats = NormalizationStats(np.zeros(3), np.ones(3))

    class FixedRng:
        def integers(self, high):
            return 1 if high == 2 else 0

    monkeypatch.setattr(
        "galaxy_classifier.data.np.random.default_rng", lambda seed: FixedRng()
    )
    assert preprocess_image(image, stats, training=True).shape == (224, 224, 3)


def test_preprocess_reaches_false_vertical_flip(monkeypatch):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    stats = NormalizationStats(np.zeros(3), np.ones(3))

    class FixedRng:
        values = iter([1, 0, 0])

        def integers(self, high):
            return next(self.values)

    monkeypatch.setattr(
        "galaxy_classifier.data.np.random.default_rng", lambda seed: FixedRng()
    )
    assert preprocess_image(image, stats, training=True).shape == (224, 224, 3)


def test_prepare_hdf5_validation_errors(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 256, 256, 3)))
        handle.create_dataset("ans", data=np.array([0]))
    with pytest.raises(ValueError, match="matching"):
        prepare_hdf5(path, tmp_path / "out")

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((2, 256, 256, 3)))
        handle.create_dataset("ans", data=np.array([0, 10]))
    with pytest.raises(ValueError, match="integer values"):
        prepare_hdf5(path, tmp_path / "out2")


def test_prepare_hdf5_reports_missing_h5py(monkeypatch, tmp_path):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="h5py"):
        prepare_hdf5(tmp_path / "missing.h5", tmp_path / "out")
