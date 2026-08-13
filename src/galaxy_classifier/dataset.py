"""Grain data source backed by a lazily opened HDF5 file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from grain import python as grain_python

from .data import IMAGE_SHAPE, NUM_CLASSES, NormalizationStats, normalize_image


class HDF5DataSource(grain_python.RandomAccessDataSource):
    """Random-access Galaxy10 source with one HDF5 handle per worker process.

    Args:
        path: HDF5 file path.
        indices: Dataset indices exposed by this source.
        image_key: HDF5 dataset name containing images.
        label_key: HDF5 dataset name containing labels.
        stats: Optional training-only statistics for image normalization.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        indices: np.ndarray,
        image_key: str = "images",
        label_key: str = "ans",
        stats: NormalizationStats | None = None,
    ) -> None:
        self.path = Path(path)
        raw_indices = np.asarray(indices)
        if not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("indices must have an integer dtype")
        self.indices = raw_indices.astype(np.int64, copy=False)
        self.image_key = image_key
        self.label_key = label_key
        self.stats = stats
        self._file: Any = None
        self._pid: int | None = None
        if self.indices.ndim != 1 or np.any(self.indices < 0):
            raise ValueError(
                "indices must be a one-dimensional array of non-negative integers"
            )

    def __len__(self) -> int:
        """Return the number of examples exposed by this source."""
        return len(self.indices)

    def __getitem__(self, record_key: int) -> dict[str, np.ndarray | np.int64]:
        """Read one example, opening the HDF5 handle lazily in this process."""
        if not 0 <= record_key < len(self):
            raise IndexError(record_key)
        self._open_for_worker()
        index = int(self.indices[record_key])
        image = np.asarray(self._file[self.image_key][index])
        label = np.asarray(self._file[self.label_key][index]).astype(np.int64).item()
        if image.shape != IMAGE_SHAPE:
            raise ValueError(
                f"image at index {index} has shape {image.shape}, "
                f"expected {IMAGE_SHAPE}"
            )
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(
                f"label at index {index} is outside [0, {NUM_CLASSES - 1}]"
            )
        if self.stats is not None:
            image = normalize_image(image, self.stats)
        return {"image": image, "label": np.int64(label)}

    def _open_for_worker(self) -> None:
        pid = os.getpid()
        if self._file is not None and self._pid == pid:
            return
        if self._file is not None:
            self._file.close()
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("h5py is required to read HDF5DataSource") from error
        self._file = h5py.File(self.path, "r")
        self._pid = pid
        if self.image_key not in self._file or self.label_key not in self._file:
            self.close()
            raise ValueError(
                f"HDF5 file must contain {self.image_key!r} and {self.label_key!r}"
            )

    def close(self) -> None:
        """Close this process's HDF5 handle, if it has been opened."""
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            file_handle.close()
            self._file = None
            self._pid = None

    def __del__(self) -> None:
        self.close()
