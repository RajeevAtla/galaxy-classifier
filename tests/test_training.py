import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import galaxy_classifier.training as training
from galaxy_classifier.checkpointing import inspect_checkpoint
from galaxy_classifier.model import ViTTiny
from galaxy_classifier.training import (
    TrainConfig,
    cross_entropy,
    learning_rate,
    loss_and_grad,
    make_optimizer,
    train_model,
)


def test_schedule_and_loss() -> None:
    assert float(learning_rate(0, base=1.0, warmup_steps=2, total_steps=10)) == 0
    assert float(learning_rate(2, base=1.0, warmup_steps=2, total_steps=10)) == 1
    assert float(learning_rate(10, base=1.0, warmup_steps=2, total_steps=10)) == 0
    assert cross_entropy(jnp.zeros((2, 2)), jnp.array([0, 1])) == jnp.log(2)
    assert make_optimizer(1e-3).init(np.zeros((1,))) is not None


def test_loss_gradient() -> None:
    model = ViTTiny(rngs=nnx.Rngs(0), compute_dtype=jnp.float32)
    images = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    loss, grads = loss_and_grad(model, images, jnp.array([0]), deterministic=True)
    assert loss.shape == ()
    assert grads is not None


def test_train_model_writes_metrics_and_best_checkpoint(tmp_path) -> None:
    class TestOptimizer:
        def __init__(self, model, transformation, *, wrt):
            self.state = {}

        def update(self, model, grads):
            return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(training.nnx, "Optimizer", TestOptimizer)
    monkeypatch.setattr(
        training.nnx,
        "state",
        lambda value: {"state": {}},
    )
    monkeypatch.setattr(
        training,
        "make_train_step",
        lambda optimizer: (
            lambda model, optimizer, images, labels: training.loss_and_grad(
                model, images, labels
            )[0]
        ),
    )
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    images = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    labels = jnp.array([0, 1])
    batches = [(images, labels)]

    result = train_model(
        model,
        batches,
        batches,
        config=TrainConfig(
            learning_rate=0.01,
            max_epochs=3,
            early_stopping_patience=1,
            warmup_epochs=0,
        ),
        run_dir=tmp_path,
    )

    assert result.best_epoch == 0
    assert result.stopped_epoch == 1
    assert inspect_checkpoint(tmp_path / "best")["best_metric"] == result.best_metric
    records = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert len(records) == 2
    monkeypatch.undo()


def test_batch_and_evaluate_errors():
    assert training._batch({"image": [[1.0, 2.0]], "label": [1]})[0].shape == (1, 2)
    with pytest.raises(TypeError):
        training._batch(object())
    with pytest.raises(ValueError, match="must not be empty"):
        training.evaluate_model(nnx.Linear(2, 2, rngs=nnx.Rngs(0)), [])


def test_evaluate_model_dict_batch():
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    metrics = training.evaluate_model(
        model, [{"image": jnp.ones((1, 2)), "label": jnp.array([0])}]
    )
    assert metrics.accuracy in (0.0, 1.0)


def test_make_train_step_with_tiny_linear():
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, make_optimizer(0.01), wrt=nnx.Param)
    step = training.make_train_step(optimizer)
    loss = step(model, optimizer, jnp.ones((1, 2)), jnp.array([0]))
    assert loss.shape == ()


def test_training_loop_with_callable_batches(tmp_path):
    class Optimizer:
        def __init__(self, *args, **kwargs):
            self.state = {"step": 0}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(training.nnx, "Optimizer", Optimizer)
    monkeypatch.setattr(training.nnx, "state", lambda value: {"state": {}})
    monkeypatch.setattr(training.nnx, "update", lambda *args: None)
    monkeypatch.setattr(
        training,
        "make_train_step",
        lambda optimizer: lambda model, optimizer, images, labels: jnp.array(1.0),
    )
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *args: training.ClassificationMetrics(
            1.0, 1.0, 1.0, np.ones(10), np.ones(10), np.ones(10), np.eye(10)
        ),
    )

    def batches():
        return [(jnp.ones((1, 2)), jnp.array([0]))]

    result = training.train_model(
        nnx.Linear(2, 2, rngs=nnx.Rngs(0)),
        batches,
        batches,
        config=TrainConfig(max_epochs=1, steps_per_epoch=1),
        run_dir=tmp_path,
    )
    assert result.best_epoch == 0
    monkeypatch.undo()


def test_make_train_step_executes_with_tiny_linear(monkeypatch):
    class Optimizer:
        def update(self, model, grads):
            self.updated = True

    optimizer = Optimizer()
    monkeypatch.setattr(training.nnx, "jit", lambda function: function)
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    monkeypatch.setattr(
        training,
        "loss_and_grad",
        lambda model, images, labels: (jnp.array(1.0), object()),
    )
    loss = training.make_train_step(optimizer)(
        model, optimizer, jnp.ones((1, 2)), jnp.array([0])
    )
    assert loss == 1.0


def test_training_callable_batches_and_unsized_error(tmp_path):
    with pytest.raises(ValueError, match="steps_per_epoch"):
        training.train_model(nnx.Linear(2, 2, rngs=nnx.Rngs(0)), iter(()), iter(()))


def test_training_resume_and_no_run_dir(monkeypatch, tmp_path):
    class Optimizer:
        def __init__(self, *args, **kwargs):
            self.state = {}

    monkeypatch.setattr(training.nnx, "Optimizer", Optimizer)
    monkeypatch.setattr(training.nnx, "state", lambda value: {})
    monkeypatch.setattr(training.nnx, "update", lambda *args: None)
    monkeypatch.setattr(
        training,
        "restore_checkpoint",
        lambda *args: (
            {"model": {}, "optimizer": {}},
            {"epoch": 0, "step": 1, "best_metric": 1.0},
        ),
    )
    monkeypatch.setattr(
        training, "make_train_step", lambda _: lambda *args: jnp.array(1.0)
    )
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *args: training.ClassificationMetrics(
            1.0, 0.5, 0.5, np.ones(10), np.ones(10), np.ones(10), np.eye(10)
        ),
    )
    result = training.train_model(
        nnx.Linear(2, 2, rngs=nnx.Rngs(0)),
        lambda: [(jnp.ones((1, 2)), jnp.array([0]))],
        lambda: [(jnp.ones((1, 2)), jnp.array([0]))],
        config=TrainConfig(max_epochs=2, steps_per_epoch=1, early_stopping_patience=1),
        resume=tmp_path / "checkpoint",
    )
    assert result.best_epoch == -1


def test_training_without_run_dir_and_without_early_stop(monkeypatch):
    class Optimizer:
        def __init__(self, *args, **kwargs):
            self.state = {}

    monkeypatch.setattr(training.nnx, "Optimizer", Optimizer)
    monkeypatch.setattr(training.nnx, "state", lambda value: {})
    monkeypatch.setattr(
        training, "make_train_step", lambda _: lambda *args: jnp.array(1.0)
    )
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *args: training.ClassificationMetrics(
            1.0, 1.0, 1.0, np.ones(10), np.ones(10), np.ones(10), np.eye(10)
        ),
    )
    result = training.train_model(
        nnx.Linear(2, 2, rngs=nnx.Rngs(0)),
        [(jnp.ones((1, 2)), jnp.array([0]))],
        [(jnp.ones((1, 2)), jnp.array([0]))],
        config=TrainConfig(max_epochs=2, early_stopping_patience=2),
    )
    assert result.best_epoch == 0
