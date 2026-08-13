#!/bin/bash
#SBATCH --job-name=galaxy-cpu-tests
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err

set -euo pipefail
module purge
module use /projects/community/modulefiles
module load gcc/5.4
uv sync --locked --dev --extra cpu
JAX_PLATFORMS=cpu uv run pytest -n auto --cov --cov-report=term-missing
