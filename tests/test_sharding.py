import jax
import pytest

from galaxy_classifier.sharding import (
    create_mesh,
    input_partition_spec,
    logits_partition_spec,
    mesh_config,
    replicate_partition_spec,
)


@pytest.mark.parametrize(
    "count, shape", [(1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (8, (4, 2))]
)
def test_mesh_conventions(count, shape):
    assert mesh_config(count).shape == shape


def test_mesh_uses_named_axes():
    mesh = create_mesh([jax.devices()[0]])
    assert mesh.axis_names == ("data", "model")
    assert mesh.shape == {"data": 1, "model": 1}


def test_partition_specs():
    assert tuple(input_partition_spec()) == ("data", None, None, None)
    assert tuple(logits_partition_spec()) == ("data", None)
    assert tuple(replicate_partition_spec()) == ()


def test_invalid_mesh_count():
    with pytest.raises(ValueError, match="one of"):
        mesh_config(3)
