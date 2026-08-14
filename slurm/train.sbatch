#!/bin/bash
#SBATCH --job-name=galaxy-vit
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu[017,018]
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail
mkdir -p slurm

export APPTAINER_CACHEDIR="/scratch/${USER}/apptainer"
mkdir -p "${APPTAINER_CACHEDIR}"
JAX_IMAGE="${APPTAINER_CACHEDIR}/jax_25.10-py3.sif"
if [ ! -f "${JAX_IMAGE}" ]; then
    apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
fi

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
if command -v git >/dev/null; then
    git rev-parse --short HEAD
fi

apptainer exec --nv --bind "${PWD}:/workspace" "${JAX_IMAGE}" bash -c '
    set -euo pipefail
    cd /workspace
    export HOME="/scratch/${USER}"
    export UV_CACHE_DIR="/tmp/galaxy-classifier-uv-cache"
    export UV_PROJECT_ENVIRONMENT="/tmp/galaxy-classifier-venv"
    /home/rra80/.local/bin/uv sync --locked --extra cuda --extra train
    nvidia-smi
    /home/rra80/.local/bin/uv run python -c "import jax; print(jax.devices())"
    /home/rra80/.local/bin/uv run galaxy-classifier train \
        --config "${TRAIN_CONFIG:-configs/vit_tiny.toml}" \
        --dataset-path "/scratch/${USER}/galaxy-classifier/data/Galaxy10_DECals.h5" \
        --run-dir "/scratch/${USER}/galaxy-classifier/runs/${SLURM_JOB_ID}"
'
