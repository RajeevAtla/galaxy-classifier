"""Small JAX/NNX training and evaluation loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .checkpointing import checkpoint_metadata, restore_checkpoint, save_checkpoint
from .metrics import ClassificationMetrics, classification_metrics


@dataclass(frozen=True)
class TrainConfig:
    """CPU-friendly defaults for a ViT training run."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 15
    warmup_epochs: int = 5
    steps_per_epoch: int | None = None


@dataclass(frozen=True)
class TrainResult:
    """Summary of a completed training run."""

    history: list[dict[str, float]]
    best_epoch: int
    best_metric: float
    stopped_epoch: int


def _batch(batch: Any) -> tuple[jax.Array, jax.Array]:
    if isinstance(batch, dict):
        return jnp.asarray(batch["image"]), jnp.asarray(batch["label"])
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return jnp.asarray(batch[0]), jnp.asarray(batch[1])
    raise TypeError("batches must be (images, labels) or image/label mappings")


def _forward(model: nnx.Module, images: jax.Array, *, deterministic: bool) -> jax.Array:
    """Call ViT-like models while allowing simple NNX modules in tests."""
    try:
        return cast(Callable[..., jax.Array], model)(
            images, deterministic=deterministic
        )
    except TypeError:
        return cast(Callable[..., jax.Array], model)(images)


def learning_rate(
    step: jax.Array | int,
    *,
    base: float,
    warmup_steps: int,
    total_steps: int,
) -> jax.Array:
    """Return linear-warmup then cosine-decay learning rate."""
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup = jnp.asarray(max(1, warmup_steps), dtype=jnp.float32)
    total = jnp.asarray(max(warmup_steps + 1, total_steps), dtype=jnp.float32)
    warm = base * step / warmup
    progress = jnp.clip((step - warmup) / (total - warmup), 0.0, 1.0)
    cosine = base * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup, warm, cosine)


def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute mean multiclass cross-entropy in float32."""
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32))
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def make_optimizer(
    learning_rate_value: float | Callable[[jax.Array], jax.Array],
    weight_decay: float = 1e-4,
) -> optax.GradientTransformation:
    """Create the baseline AdamW optimizer."""
    return optax.adamw(learning_rate_value, weight_decay=weight_decay)


def loss_and_grad(
    model: nnx.Module,
    images: jax.Array,
    labels: jax.Array,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, Any]:
    """Evaluate model loss and gradients."""

    def loss_fn(model: nnx.Module) -> jax.Array:
        return cross_entropy(
            _forward(model, images, deterministic=deterministic), labels
        )

    return nnx.value_and_grad(loss_fn)(model)


def make_train_step(optimizer: nnx.Optimizer) -> Callable[..., jax.Array]:
    """Return a jitted NNX update function for one batch."""

    @nnx.jit
    def step(
        model: nnx.Module,
        images: jax.Array,
        labels: jax.Array,
    ) -> jax.Array:
        loss, grads = loss_and_grad(model, images, labels)
        optimizer.update(model, grads)
        return loss

    return step


def evaluate_model(model: nnx.Module, batches: Any) -> ClassificationMetrics:
    """Evaluate a model over batches and return classification metrics."""
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    import numpy as np

    for raw_batch in batches:
        images, batch_labels = _batch(raw_batch)
        logits = _forward(model, images, deterministic=True)
        labels.append(np.asarray(batch_labels))
        predictions.append(np.asarray(jnp.argmax(logits, axis=-1)))
    if not labels:
        raise ValueError("evaluation batches must not be empty")
    return classification_metrics(np.concatenate(labels), np.concatenate(predictions))


def _write_metric(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def train_model(
    model: nnx.Module,
    train_batches: Any,
    validation_batches: Any,
    *,
    config: TrainConfig = TrainConfig(),
    run_dir: str | Path | None = None,
    resume: str | Path | None = None,
) -> TrainResult:
    """Train ``model`` and select the checkpoint with best validation macro-F1.

    ``train_batches`` and ``validation_batches`` may be re-iterable sequences or
    callables returning a fresh iterator for each epoch.
    """
    import numpy as np

    def batches(value: Any) -> Any:
        return value() if callable(value) else value

    train_count = config.steps_per_epoch
    if train_count is None:
        train_count = len(train_batches) if hasattr(train_batches, "__len__") else None
    if not train_count:
        raise ValueError("steps_per_epoch is required for unsized training iterators")
    total_steps = train_count * config.max_epochs
    schedule = lambda step: learning_rate(  # noqa: E731
        step,
        base=config.learning_rate,
        warmup_steps=config.warmup_epochs * train_count,
        total_steps=total_steps,
    )
    optimizer = nnx.Optimizer(
        model,
        make_optimizer(schedule, config.weight_decay),
        wrt=nnx.Param,
    )
    start_epoch = 0
    step = 0
    best_metric = -np.inf
    best_epoch = -1
    history: list[dict[str, float]] = []
    if resume is not None:
        restored, metadata = restore_checkpoint(
            resume,
            {"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
        )
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        start_epoch = int(metadata.get("epoch", -1)) + 1
        step = int(metadata.get("step", 0))
        best_metric = float(metadata.get("best_metric", -np.inf))

    step_fn = make_train_step(optimizer)
    jsonl = Path(run_dir) / "metrics.jsonl" if run_dir is not None else None
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
    stale = 0
    for epoch in range(start_epoch, config.max_epochs):
        losses = []
        for raw_batch in batches(train_batches):
            images, labels = _batch(raw_batch)
            losses.append(float(step_fn(model, images, labels)))
            step += 1
        metrics = evaluate_model(model, batches(validation_batches))
        record = {
            "epoch": epoch,
            "step": step,
            "loss": float(np.mean(losses)),
            **metrics.as_dict(),
        }
        history.append(
            {
                "loss": record["loss"],
                "accuracy": record["accuracy"],
                "macro_f1": record["macro_f1"],
                "weighted_f1": record["weighted_f1"],
            }
        )
        if jsonl is not None:
            _write_metric(jsonl, record)
        if metrics.macro_f1 > best_metric:
            best_metric, best_epoch, stale = metrics.macro_f1, epoch, 0
            if run_dir is not None:
                save_checkpoint(
                    Path(run_dir) / "best",
                    {"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
                    checkpoint_metadata(
                        epoch=epoch,
                        step=step,
                        best_metric=best_metric,
                        config=asdict(config),
                    ),
                )
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    return TrainResult(history, best_epoch, float(best_metric), epoch)
