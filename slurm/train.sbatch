#!/bin/bash
#SBATCH --job-name=galaxy-vit
#SBATCH --partition=gpu
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

module purge
module use /projects/community/modulefiles
module load apptainer
module load git/2.35.1-ez82

export APPTAINER_CACHEDIR="/scratch/${USER}/apptainer"
mkdir -p "${APPTAINER_CACHEDIR}"
JAX_IMAGE="${APPTAINER_CACHEDIR}/jax_25.10-py3.sif"
if [ ! -f "${JAX_IMAGE}" ]; then
    apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
fi

echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
git rev-parse --short HEAD

apptainer exec --nv "${JAX_IMAGE}" bash -c '
    set -euo pipefail
    export HOME="/scratch/${USER}"
    export UV_CACHE_DIR="/scratch/${USER}/uv-cache"
    mkdir -p "${UV_CACHE_DIR}"
    uv sync --locked --extra cuda --extra train
    nvidia-smi
    uv run python -c "import jax; print(jax.devices())"
    uv run galaxy-classifier train --config configs/vit_tiny.toml
'
