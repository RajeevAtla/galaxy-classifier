#!/bin/bash
#SBATCH --job-name=galaxy-preflight
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=gpu[017,018]
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail
module purge
module use /projects/community/modulefiles
module load apptainer
export APPTAINER_CACHEDIR="/scratch/${USER}/apptainer"
mkdir -p "${APPTAINER_CACHEDIR}"
JAX_IMAGE="${APPTAINER_CACHEDIR}/jax_25.10-py3.sif"
if [ ! -f "${JAX_IMAGE}" ]; then
    apptainer pull "${JAX_IMAGE}" docker://nvcr.io/nvidia/jax:25.10-py3
fi
apptainer exec --nv "${JAX_IMAGE}" bash -c '
    set -euo pipefail
    export HOME="/scratch/${USER}"
    export UV_CACHE_DIR="/scratch/${USER}/uv-cache"
    mkdir -p "${UV_CACHE_DIR}"
    uv sync --locked --extra cuda --extra train
    nvidia-smi
    uv run python -c "import jax; print(jax.devices()); print(jax.default_backend())"
'
