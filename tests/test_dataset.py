import h5py
import numpy as np
import pytest

from galaxy_classifier.data import NormalizationStats
from galaxy_classifier.dataset import HDF5DataSource


def _write_dataset(path, image_shape=(2, 256, 256, 3), labels=(0, 9)):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros(image_shape, dtype=np.uint8))
        handle.create_dataset("ans", data=np.asarray(labels, dtype=np.int64))


def test_hdf5_source_reads_normalized_record_and_reuses_handle(tmp_path):
    path = tmp_path / "data.h5"
    _write_dataset(path)
    source = HDF5DataSource(
        path, np.array([1]), stats=NormalizationStats(np.zeros(3), np.ones(3))
    )
    assert len(source) == 1
    first = source[0]
    handle = source._file
    assert first["image"].shape == (256, 256, 3)
    assert first["label"] == 9
    assert source[0]["label"] == 9
    assert source._file is handle
    source.close()
    assert source._file is None


@pytest.mark.parametrize("indices", [np.array([1.0]), np.array([[0]]), np.array([-1])])
def test_hdf5_source_rejects_bad_indices(tmp_path, indices):
    with pytest.raises(ValueError):
        HDF5DataSource(tmp_path / "missing.h5", indices)


def test_hdf5_source_rejects_bad_records_and_keys(tmp_path):
    path = tmp_path / "data.h5"
    _write_dataset(path, image_shape=(2, 1, 1, 3))
    source = HDF5DataSource(path, np.array([0]))
    with pytest.raises(ValueError, match="image at index"):
        source[0]
    with pytest.raises(IndexError):
        source[1]
    source.close()

    with h5py.File(path, "w") as handle:
        handle.create_dataset("images", data=np.zeros((1, 256, 256, 3)))
    with pytest.raises(ValueError, match="must contain"):
        HDF5DataSource(path, np.array([0]))[0]


def test_hdf5_source_rejects_bad_label(tmp_path):
    path = tmp_path / "data.h5"
    _write_dataset(path, labels=(10,))
    with pytest.raises(ValueError, match="outside"):
        HDF5DataSource(path, np.array([0]))[0]


def test_hdf5_source_reopens_after_pid_change(tmp_path, monkeypatch):
    path = tmp_path / "data.h5"
    _write_dataset(path)
    source = HDF5DataSource(path, np.array([0]))
    source[0]
    monkeypatch.setattr("galaxy_classifier.dataset.os.getpid", lambda: 999)
    source[0]
    source.close()


def test_hdf5_source_import_error(tmp_path, monkeypatch):
    path = tmp_path / "data.h5"
    _write_dataset(path)
    source = HDF5DataSource(path, np.array([0]))
    monkeypatch.setattr(
        "builtins.__import__",
        lambda name, *args, **kwargs: (
            (_ for _ in ()).throw(ImportError())
            if name == "h5py"
            else __import__(name, *args, **kwargs)
        ),
    )
    with pytest.raises(RuntimeError, match="h5py"):
        source[0]
