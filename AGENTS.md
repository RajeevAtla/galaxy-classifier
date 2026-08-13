# Galaxy Classifier Agent Guide

## Mission

Build a small, reproducible JAX AI Stack project that classifies Galaxy10 DECaLS images with a Vision Transformer. The repository starts nearly empty. Prefer the smallest correct implementation, and do not add speculative framework machinery.

## Non-negotiable decisions

- Dataset: Galaxy10 DECaLS, 17,736 `256x256x3` HDF5 images, ten classes.
- Split: stratified 70/15/15, seed `42` for every random operation.
- Model: ViT-Tiny from scratch: patch `16`, embed `192`, six blocks, three heads, MLP `768`, dropout `0.1`, ten classes.
- Preprocessing: resize to `224x224`; horizontal/vertical flips and quarter-turn rotations only; no channel swapping or color jitter.
- Normalize with training-split-only per-channel statistics for `g`, `r`, and `z`.
- Stack: JAX, Flax NNX, Optax, Orbax, Grain, `h5py`, `ml_dtypes`.
- Training: AdamW, weight decay `1e-4`, linear warmup plus cosine decay, max 100 epochs, early stopping patience 15 on validation macro-F1.
- Batch: global size 256, accumulation 1 by default.
- Precision: bfloat16 inputs/activations/matmul where supported; float32 parameters, optimizer state, labels, reductions, and metrics. CPU tests use float32.
- Distribution: `jax.jit` plus JAX sharding API from day one; named 2D `data x model` mesh. Standard meshes: 1 GPU `1x1`, 2 `2x1`, 4 `2x2`, 8 `4x2`. Shard model parameters as well as batches where compatible.
- CUDA: use the `jax[cuda13]` pip wheel through `uv`; do not require a CUDA toolkit module. The host still needs a compatible NVIDIA driver.
- Data: Grain custom `RandomAccessDataSource` for HDF5. Open each HDF5 handle lazily inside its worker; never share handles across forked workers.
- Checkpoints: Orbax, outside Git, under configurable scratch run directories. Preserve model, optimizer, PRNG, epoch/step, best metric, iterator state, config, mesh, and sharding metadata. Resume only with explicit `--resume`.
- CLI: standard-library `argparse` with `prepare-data`, `train`, `evaluate`, and `inspect-checkpoint` subcommands.
- Config: TOML defaults plus explicit CLI overrides. Save resolved configuration in each run manifest.
- Metrics: JSONL and TensorBoard. Select the best checkpoint by validation macro-F1; report test accuracy, macro-F1, weighted-F1, per-class metrics, and normalized confusion matrix.
- Tests: CPU-only, `pytest-xdist[psutil]`, target 100% line and branch coverage for the package.
- Quality: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyrefly check`, `uv run pytest -n auto`.
- Python: `>=3.11,<3.13`; use `uv`, a proper `pyproject.toml`, and a committed `uv.lock`.
- Slurm: one node, configurable GPU count, default partition `gpu`, `--gres=gpu:1`, `--constraint=ampere`, 4 CPUs, 32G RAM, 12-hour wall time. Run preflight and `nvidia-smi` before training.
- Amarel native builds/tests: load the available GCC module (currently `gcc/5.4`; site images may also provide `gcc/10.2.0/openmpi`) in CPU allocations when a dependency needs compilation.
- Commits: Conventional Commits, fewer than 2,000 changed lines excluding lockfiles. Commit descriptions must explain motivation, implementation, validation, and limitations.

## Canonical labels

Use this exact zero-based mapping everywhere:

0. Disturbed Galaxies
1. Merging Galaxies
2. Round Smooth Galaxies
3. In-between Round Smooth Galaxies
4. Cigar Shaped Smooth Galaxies
5. Barred Spiral Galaxies
6. Unbarred Tight Spiral Galaxies
7. Unbarred Loose Spiral Galaxies
8. Edge-on Galaxies without Bulge
9. Edge-on Galaxies with Bulge

## Repository conventions

- Use a `src/galaxy_classifier/` package and `tests/`.
- Keep CLI code thin. Put computation in pure functions where practical.
- Public functions, classes, and methods use Google-style docstrings.
- Comments explain non-obvious choices, especially HDF5 process boundaries and sharding. Do not comment every obvious line.
- Match existing style. Do not refactor unrelated code.
- Use ASCII unless an existing file clearly requires otherwise.
- Do not commit the HDF5 dataset, checkpoints, generated logs, or secrets.
- Small deterministic preparation artifacts may be committed under `data/`.
- Avoid notebooks, Makefiles, new dependencies, experiment trackers, or compatibility layers unless explicitly needed.
- Do not run substantial Python, tests, downloads, or training on a Slurm login node. Use `srun` or a batch job on compute nodes.

## Agent workflow

- Read this file before editing.
- Announce assumptions when a decision is genuinely unclear; ask the user rather than silently changing an agreed decision.
- Before editing, inspect the relevant files and current worktree. Preserve unrelated user changes.
- Work only in assigned files. Avoid overlapping edits with other agents.
- Run the smallest relevant checks after edits. Report commands and failures precisely.
- Do not commit, amend, push, or create a PR unless the user explicitly asks.
- Do not delete or revert work made by another agent or the user.

## Planned file ownership

Agents may add tests adjacent to their implementation, but should avoid editing another agent's assigned implementation files.

- Bootstrap: `pyproject.toml`, `uv.lock`, package skeleton, tool configuration, `.gitignore`, CI.
- Data: `src/galaxy_classifier/data.py`, `dataset.py`, preparation tests, committed metadata/split artifacts only when source data is available.
- Model/distribution: `model.py`, `sharding.py`, model tests.
- Training/checkpointing: `training.py`, `metrics.py`, `checkpointing.py`, related tests.
- CLI/operations/docs: `cli.py`, `system.py`, `configs/`, `slurm/`, `README.md`.

## Verification gates

At completion, run on an appropriate compute node:

```text
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest -n auto
```

GPU validation must happen inside Slurm and include CUDA/JAX device discovery, a `1x1` mesh smoke test, and eventually a `2x2` four-GPU smoke test. Exact bitwise reproducibility across different GPUs, drivers, and XLA builds is not promised; record those environments.
