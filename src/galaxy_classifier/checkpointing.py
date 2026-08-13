"""Small Orbax checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


def checkpoint_metadata(
    *,
    epoch: int,
    step: int,
    best_metric: float,
    config: dict[str, Any] | None = None,
    iterator_state: Any = None,
    mesh: Any = None,
    sharding: Any = None,
) -> dict[str, Any]:
    """Build the stable metadata stored beside a training checkpoint."""
    return {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "config": config or {},
        "iterator_state": iterator_state,
        "mesh": str(mesh) if mesh is not None else None,
        "sharding": sharding,
    }


def save_checkpoint(path: str | Path, state: Any, metadata: dict[str, Any]) -> None:
    """Save a PyTree and JSON metadata with Orbax."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(str(destination), state, force=True)
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def restore_checkpoint(path: str | Path, item: Any) -> tuple[Any, dict[str, Any]]:
    """Restore a PyTree and its metadata."""
    source = Path(path)
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(str(source), item=item)
    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return state, metadata


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata without restoring arrays."""
    metadata_path = Path(path).with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    return json.loads(metadata_path.read_text())
