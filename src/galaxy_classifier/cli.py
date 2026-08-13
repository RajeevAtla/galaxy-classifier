"""Command-line entry point for Galaxy10 experiments."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from .checkpointing import inspect_checkpoint
from .data import (
    ZENODO_SHA256,
    ZENODO_URL,
    NormalizationStats,
    download_dataset,
    prepare_hdf5,
    preprocess_image,
)
from .dataset import HDF5DataSource
from .model import ViTTiny
from .training import TrainConfig, evaluate_model, train_model


def load_config(path: Path | None, overrides: dict[str, object]) -> dict[str, object]:
    """Load TOML configuration and apply explicit non-None overrides."""
    config: dict[str, object] = {}
    if path is not None and path.exists():
        with path.open("rb") as handle:
            config.update(tomllib.load(handle))
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def build_parser() -> argparse.ArgumentParser:
    """Build the project argument parser."""
    parser = argparse.ArgumentParser(prog="galaxy-classifier")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--dataset-path", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, default=Path("data"))
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--url", default=ZENODO_URL)
    prepare.add_argument("--sha256")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(handler=_prepare_data)
    for name in ("train", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--config", type=Path, default=Path("configs/vit_tiny.toml")
        )
        command.add_argument("--dataset-path", type=Path)
        command.add_argument("--run-dir", type=Path)
        command.add_argument("--resume", type=Path)
        command.set_defaults(handler=_run_training_command)
    inspect = subparsers.add_parser("inspect-checkpoint")
    inspect.add_argument("path", type=Path)
    inspect.set_defaults(handler=_inspect)
    return parser


def _prepare_data(args: argparse.Namespace) -> int:
    dataset_path = args.dataset_path
    if args.download:
        dataset_path = download_dataset(dataset_path, url=args.url, sha256=args.sha256)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = prepare_hdf5(dataset_path, args.output_dir, seed=args.seed)
    metadata["published_url"] = args.url
    metadata["published_sha256"] = ZENODO_SHA256 if args.url == ZENODO_URL else None
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _run_training_command(args: argparse.Namespace) -> int:
    """Build batches and hand the run to the existing training APIs."""
    config = load_config(
        args.config,
        {"dataset_path": args.dataset_path, "run_dir": args.run_dir},
    )
    if not config.get("dataset_path"):
        raise ValueError("dataset_path is required in config or on the command line")
    if not config.get("run_dir"):
        raise ValueError("run_dir is required in config or on the command line")
    train_config = TrainConfig(
        **{key: config[key] for key in asdict(TrainConfig()) if key in config}
    )
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            _jsonable({**config, "train": asdict(train_config)}),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    dataset_path = Path(config["dataset_path"])
    artifact_dir = Path(config.get("data_dir", dataset_path.parent))
    split = np.load(artifact_dir / "split.npz")
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    stats = NormalizationStats(
        np.asarray(metadata["normalization"]["mean"], dtype=np.float32),
        np.asarray(metadata["normalization"]["std"], dtype=np.float32),
    )
    batch_size = int(config.get("global_batch_size", 256))

    def batches(indices: np.ndarray, training: bool = False):
        source = HDF5DataSource(dataset_path, indices, stats=stats)
        try:
            for start in range(0, len(source), batch_size):
                records = []
                for index in range(start, min(start + batch_size, len(source))):
                    record = source[index]
                    record["image"] = preprocess_image(
                        record["image"], stats, seed=index, training=training
                    )
                    records.append(record)
                yield {
                    "image": jnp.asarray(
                        np.stack([record["image"] for record in records])
                    ),
                    "label": jnp.asarray(
                        [record["label"] for record in records], dtype=jnp.int32
                    ),
                }
        finally:
            source.close()

    if args.command == "train":
        model = ViTTiny(rngs=nnx.Rngs(int(config.get("seed", 42))))
        result = train_model(
            model,
            lambda: batches(split["train"], True),
            lambda: batches(split["validation"]),
            config=TrainConfig(
                **{
                    **asdict(train_config),
                    "steps_per_epoch": len(split["train"]) // batch_size + 1,
                }
            ),
            run_dir=run_dir,
            resume=args.resume,
        )
        result_data = (
            asdict(result)
            if hasattr(result, "__dataclass_fields__")
            else {
                name: getattr(result, name)
                for name in ("history", "best_epoch", "best_metric", "stopped_epoch")
            }
        )
        print(json.dumps(result_data, indent=2, sort_keys=True))
        return 0
    checkpoint = args.resume or (run_dir / "best")
    model = ViTTiny(rngs=nnx.Rngs(int(config.get("seed", 42))))
    from .checkpointing import restore_checkpoint

    restored, _ = restore_checkpoint(checkpoint, {"model": nnx.state(model)})
    nnx.update(model, restored["model"])
    metrics = evaluate_model(model, batches(split["test"]))
    print(json.dumps(metrics.as_dict(), indent=2, sort_keys=True))
    return 0


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_checkpoint(args.path), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and execute a command."""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
