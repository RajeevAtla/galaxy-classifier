import json
from pathlib import Path

import numpy as np
import pytest
from flax import nnx

import galaxy_classifier.checkpointing as checkpointing
from galaxy_classifier import cli
from galaxy_classifier.metrics import ClassificationMetrics


def test_config_parser_and_jsonable(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("learning_rate = 0.1\n")
    assert cli.load_config(config_path, {"run_dir": None, "x": 2}) == {
        "learning_rate": 0.1,
        "x": 2,
    }
    assert cli._jsonable({1: (Path("x"),)}) == {"1": ["x"]}


def test_parser_subcommands_and_not_found(tmp_path):
    args = cli.build_parser().parse_args(
        ["prepare-data", "--dataset-path", str(tmp_path / "missing")]
    )
    with pytest.raises(FileNotFoundError):
        args.handler(args)
    with pytest.raises(ValueError, match="SHA256"):
        cli.main(["prepare-data", "--dataset-path", str(tmp_path / "x"), "--download"])
    with pytest.raises(ValueError, match="dataset_path"):
        cli.main(["train", "--config", str(tmp_path / "empty.toml")])


def test_inspect_command_and_main(tmp_path, capsys):
    metadata = tmp_path / "checkpoint.json"
    metadata.write_text(json.dumps({"step": 3}))
    assert cli.main(["inspect-checkpoint", str(metadata.with_suffix(""))]) == 0
    assert json.loads(capsys.readouterr().out) == {"step": 3}
    config = tmp_path / "config.toml"
    config.write_text("run_dir = 'runs'\n")
    with pytest.raises(ValueError, match="dataset_path"):
        cli.main(["evaluate", "--config", str(config)])


def test_prepare_data_success(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "dataset.h5"
    dataset.write_bytes(b"x")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(cli, "prepare_hdf5", lambda *args, **kwargs: {"ok": True})
    assert (
        cli.main(
            [
                "prepare-data",
                "--dataset-path",
                str(dataset),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["published_sha256"] is not None


def test_training_and_evaluation_commands(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "split.npz",
        train=np.array([0]),
        validation=np.array([0]),
        test=np.array([0]),
    )
    (data_dir / "metadata.json").write_text(
        json.dumps({"normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}})
    )
    dataset = tmp_path / "data.h5"
    dataset.write_bytes(b"x")
    run_dir = tmp_path / "run"
    config = tmp_path / "config.toml"
    config.write_text(
        f"dataset_path = '{dataset}'\nrun_dir = '{run_dir}'\ndata_dir = '{data_dir}'\n"
    )
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    monkeypatch.setattr(cli, "ViTTiny", lambda **kwargs: model)
    from galaxy_classifier.training import TrainResult

    monkeypatch.setattr(
        cli,
        "train_model",
        lambda *args, **kwargs: TrainResult([], 0, 1.0, 0),
    )
    monkeypatch.setattr(cli, "HDF5DataSource", lambda *args, **kwargs: [])
    assert cli.main(["train", "--config", str(config)]) == 0
    assert "best_epoch" in capsys.readouterr().out
    monkeypatch.setattr(
        checkpointing,
        "restore_checkpoint",
        lambda *args: ({"model": nnx.state(model)}, {}),
    )
    monkeypatch.setattr(
        cli,
        "evaluate_model",
        lambda *args: ClassificationMetrics(
            1.0, 1.0, 1.0, np.ones(1), np.ones(1), np.ones(1), np.ones((1, 1))
        ),
    )
    assert (
        cli.main(["evaluate", "--config", str(config), "--resume", str(run_dir)]) == 0
    )
    assert "accuracy" in capsys.readouterr().out


def test_training_requires_run_dir(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("dataset_path = 'data.h5'\n")
    with pytest.raises(ValueError, match="run_dir"):
        cli.main(["train", "--config", str(config)])


def test_cli_batch_generator_reads_records(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "split.npz",
        train=np.array([0]),
        validation=np.array([0]),
        test=np.array([0]),
    )
    (data_dir / "metadata.json").write_text(
        json.dumps({"normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]}})
    )
    dataset = tmp_path / "dataset.h5"
    dataset.write_bytes(b"x")
    config = tmp_path / "config.toml"
    config.write_text(
        f"dataset_path = '{dataset}'\nrun_dir = '{tmp_path / 'run'}'\n"
        f"data_dir = '{data_dir}'\nglobal_batch_size = 1\n"
    )

    class Source:
        def __init__(self, *args, **kwargs):
            self.records = [{"image": np.zeros((256, 256, 3)), "label": np.int64(0)}]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

        def close(self):
            return None

    from galaxy_classifier.training import TrainResult

    monkeypatch.setattr(cli, "HDF5DataSource", Source)
    monkeypatch.setattr(
        cli, "ViTTiny", lambda **kwargs: nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    )
    monkeypatch.setattr(
        cli, "train_model", lambda *args, **kwargs: TrainResult([], 0, 1.0, 0)
    )
    assert cli.main(["train", "--config", str(config)]) == 0
