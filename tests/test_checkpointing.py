import json

import jax.numpy as jnp

from galaxy_classifier.checkpointing import (
    checkpoint_metadata,
    inspect_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def test_checkpoint_metadata_defaults_and_values():
    assert checkpoint_metadata(epoch=1, step=2, best_metric=0.5) == {
        "epoch": 1,
        "step": 2,
        "best_metric": 0.5,
        "config": {},
        "iterator_state": None,
        "mesh": None,
        "sharding": None,
    }


def test_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "checkpoint"
    state = {"weights": jnp.array([1, 2, 3]), "step": 4}
    save_checkpoint(path, state, {"step": 4})
    restored, metadata = restore_checkpoint(path, state)
    assert restored["weights"].tolist() == [1, 2, 3]
    assert restored["step"] == 4
    assert metadata == {"step": 4}
    assert inspect_checkpoint(path) == {"step": 4}


def test_checkpoint_inspection_requires_metadata(tmp_path) -> None:
    path = tmp_path / "missing"
    path.with_suffix(".json").write_text(json.dumps({"step": 1}))
    assert inspect_checkpoint(path) == {"step": 1}


def test_checkpoint_inspection_missing_file(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(tmp_path / "unknown")
