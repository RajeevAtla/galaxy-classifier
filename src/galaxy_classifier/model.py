"""ViT-Tiny model used by the galaxy classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for the locked ViT-Tiny architecture."""

    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_dim: int = 768
    num_classes: int = 10
    dropout_rate: float = 0.1

    @property
    def num_patches(self) -> int:
        """Return the number of image patches."""
        return (self.image_size // self.patch_size) ** 2


def _compute_dtype(dtype: Any | None) -> Any:
    """Choose bfloat16 on accelerators and float32 on CPU."""
    if dtype is not None:
        return dtype
    return jnp.float32 if jax.default_backend() == "cpu" else jnp.bfloat16


class MlpBlock(nnx.Module):
    """Transformer feed-forward block."""

    def __init__(self, config: ViTConfig, *, dtype: Any, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(
            config.embed_dim,
            config.mlp_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            config.mlp_dim,
            config.embed_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        x = nnx.gelu(self.fc1(x))
        return self.dropout(self.fc2(x), deterministic=deterministic)


class AttentionBlock(nnx.Module):
    """Pre-normalized multi-head self-attention transformer block."""

    def __init__(self, config: ViTConfig, *, dtype: Any, rngs: nnx.Rngs):
        if config.embed_dim % config.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.norm1 = nnx.LayerNorm(
            config.embed_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs
        )
        self.qkv = nnx.Linear(
            config.embed_dim,
            3 * config.embed_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.proj = nnx.Linear(
            config.embed_dim,
            config.embed_dim,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(
            config.embed_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs
        )
        self.mlp = MlpBlock(config, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads

    def __call__(self, x: jax.Array, *, deterministic: bool) -> jax.Array:
        qkv = self.qkv(self.norm1(x))
        batch, tokens, _ = qkv.shape
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        weights = jax.nn.softmax(
            jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(self.head_dim), axis=-1
        )
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(batch, tokens, -1)
        x = x + self.dropout(self.proj(attended), deterministic=deterministic)
        return x + self.mlp(self.norm2(x), deterministic=deterministic)


class ViTTiny(nnx.Module):
    """ViT-Tiny classifier for 224x224 RGB images."""

    def __init__(
        self,
        config: ViTConfig = ViTConfig(),
        *,
        rngs: nnx.Rngs,
        compute_dtype: Any | None = None,
    ):
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        self.compute_dtype = _compute_dtype(compute_dtype)
        self.patch_embed = nnx.Conv(
            config.channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cls_token = nnx.Param(
            jnp.zeros((1, 1, config.embed_dim), dtype=jnp.float32)
        )
        self.pos_embed = nnx.Param(
            jnp.zeros((1, config.num_patches + 1, config.embed_dim), dtype=jnp.float32)
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.blocks = nnx.data(
            [
                AttentionBlock(config, dtype=self.compute_dtype, rngs=rngs)
                for _ in range(config.depth)
            ]
        )
        self.norm = nnx.LayerNorm(
            config.embed_dim,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.head = nnx.Linear(
            config.embed_dim,
            config.num_classes,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, images: jax.Array, *, deterministic: bool = True) -> jax.Array:
        """Return float32 class logits for a batch of images."""
        x = jnp.asarray(images, dtype=self.compute_dtype)
        x = self.patch_embed(x).reshape(
            x.shape[0], self.config.num_patches, self.config.embed_dim
        )
        cls = jnp.broadcast_to(
            self.cls_token.value.astype(self.compute_dtype),
            (x.shape[0], 1, self.config.embed_dim),
        )
        x = self.dropout(
            jnp.concatenate((cls, x), axis=1)
            + self.pos_embed.value.astype(self.compute_dtype),
            deterministic=deterministic,
        )
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return self.head(self.norm(x[:, 0])).astype(jnp.float32)


ViT = ViTTiny
