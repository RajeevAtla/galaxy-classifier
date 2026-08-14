from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from galaxy_classifier.model import ViTConfig, ViTTiny


def test_locked_configuration():
    config = ViTConfig()
    assert (config.patch_size, config.embed_dim, config.depth) == (16, 192, 6)
    assert (config.num_heads, config.mlp_dim, config.num_classes) == (3, 768, 10)
    assert config.num_patches == 196


def test_model_shape_and_float32_logits():
    model = ViTTiny(rngs=nnx.Rngs(0), compute_dtype=jnp.float32)
    logits = model(jnp.zeros((2, 224, 224, 3)), deterministic=True)
    assert logits.shape == (2, 10)
    assert logits.dtype == jnp.float32
    assert all(
        cast(Any, param).value.dtype == jnp.float32
        for _, param in nnx.to_flat_state(nnx.state(model, nnx.Param))
    )


def test_model_is_jittable():
    model = ViTTiny(rngs=nnx.Rngs(0), compute_dtype=jnp.float32)
    logits = jax.jit(model)(jnp.ones((1, 224, 224, 3)))
    assert logits.shape == (1, 10)


def test_invalid_configuration():
    try:
        ViTTiny(ViTConfig(image_size=225), rngs=nnx.Rngs(0))
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("expected invalid image size to fail")


def test_dtype_and_attention_branches(monkeypatch):
    import galaxy_classifier.model as model_module

    monkeypatch.setattr(model_module.jax, "default_backend", lambda: "gpu")
    assert model_module._compute_dtype(None) == jnp.bfloat16
    assert model_module._compute_dtype(jnp.float16) == jnp.float16
    with pytest.raises(ValueError, match="divisible"):
        model_module.AttentionBlock(
            ViTConfig(embed_dim=5, num_heads=2), dtype=jnp.float32, rngs=nnx.Rngs(0)
        )
