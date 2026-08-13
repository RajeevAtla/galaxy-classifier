# Galaxy Classifier

JAX/Flax NNX Vision Transformer baseline for the Galaxy10 DECaLS dataset.

## Development

The project requires Python 3.11 or 3.12. Install the locked environment with:

```text
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest -n auto
```

On Amarel, run the CPU checks in a compute allocation with the site compiler
module loaded when native packages need building:

```text
module purge
module use /projects/community/modulefiles
module load gcc/5.4
srun --partition=main --cpus-per-task=4 --mem=8G --time=00:20:00 \\
  bash -lc 'uv sync --locked --dev --extra cpu && uv run pytest -n auto'
```

Tests use CPU execution. Training uses the CUDA 13 JAX environment inside
NVIDIA's JAX Apptainer image and must run in a Slurm allocation, not on a login
node. This mirrors the working Amarel GPU workflow and avoids compiling native
dependencies on the host.

## Dataset

The default source is the published [Zenodo Galaxy10 DECaLS
record](https://zenodo.org/records/10845026), specifically
`Galaxy10_DECals.h5` at
`https://zenodo.org/api/records/10845026/files/Galaxy10_DECals.h5/content`.
Zenodo publishes an MD5 (`c6b7b4db82b3a5d63d6a7e3e5249b51c`) but not a SHA256,
so downloads require an independently obtained SHA256 via `--sha256` rather
than silently accepting an unverified file:

```text
uv run --no-sync galaxy-classifier prepare-data \
  --dataset-path /scratch/$USER/Galaxy10_DECals.h5 \
  --output-dir /scratch/$USER/galaxy10-data \
  --download --sha256 <sha256>
```

Preparation validates the `images` and `ans` HDF5 datasets, creates the seed-42
stratified split in `split.npz`, and computes training-only `g`, `r`, and `z`
normalization statistics in `metadata.json`. The HDF5 file is deliberately not
tracked by Git.

## Slurm

The supplied jobs follow the Amarel convention: GPU partition, Ampere
constraint, four CPUs, 32 GB RAM, and a 12-hour training limit.

```text
sbatch slurm/preflight.sbatch
sbatch slurm/train.sbatch
```

Checkpoints and run logs belong in configurable scratch directories. Copy
results to backed-up storage when a run is worth keeping; scratch is temporary.

## Model

The baseline is a ViT-Tiny trained from scratch: 16-pixel patches, 192-wide
embeddings, six blocks, three attention heads, 768-wide MLPs, 0.1 dropout, and
ten Galaxy10 classes. It uses a seed-42 stratified 70/15/15 split and a named
`data x model` JAX mesh designed to scale from `1x1` to `2x2` and `4x2`.

## Reproducibility

Exact bitwise results across different NVIDIA GPUs, drivers, CUDA libraries,
and XLA builds are not promised. Runs record the resolved configuration and
hardware/software metadata. Please acknowledge the Rutgers OARC Amarel cluster
when reporting results produced there.
