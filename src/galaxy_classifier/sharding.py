"""Named mesh and explicit sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec


@dataclass(frozen=True)
class MeshConfig:
    """Named 2D data x model mesh configuration."""

    data: int
    model: int

    @property
    def shape(self) -> tuple[int, int]:
        """Return the mesh shape in data, model order."""
        return self.data, self.model


def mesh_config(device_count: int) -> MeshConfig:
    """Return the standard mesh convention for a device count."""
    conventions = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (4, 2)}
    try:
        data, model = conventions[device_count]
    except KeyError as exc:
        raise ValueError("device_count must be one of 1, 2, 4, or 8") from exc
    return MeshConfig(data, model)


def create_mesh(
    devices: list[jax.Device] | tuple[jax.Device, ...] | None = None,
) -> Mesh:
    """Create a row-major named data x model mesh."""
    devices = tuple(jax.devices() if devices is None else devices)
    config = mesh_config(len(devices))
    return Mesh(
        np.asarray(devices, dtype=object).reshape(config.shape),
        axis_names=("data", "model"),
    )


def input_partition_spec() -> PartitionSpec:
    """Partition a `[batch, height, width, channels]` input over data."""
    return PartitionSpec("data", None, None, None)


def logits_partition_spec() -> PartitionSpec:
    """Partition classifier logits over data."""
    return PartitionSpec("data", None)


def replicate_partition_spec() -> PartitionSpec:
    """Return the explicit replicated spec for scalar or small state."""
    return PartitionSpec()
