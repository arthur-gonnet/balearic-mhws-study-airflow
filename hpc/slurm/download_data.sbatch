#!/bin/bash

#SBATCH --job-name=mhw-download
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mhw-download-%A.out

# Network-bound, not compute-bound - deliberately light on cpus/mem compared to compute_mhws.sbatch.

set -euo pipefail

# TODO: activate this cluster's Python environment, e.g.:
#   module load python/3.11
#   source /path/to/venv/bin/activate

PROJECT_DIR="${SLURM_REMOTE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}"
export BALEARIC_DATA_DIR="${BALEARIC_DATA_DIR:-${PROJECT_DIR}/data}"

cd "$PROJECT_DIR"

python -m pipelines.mhws_pipeline download \
    --dataset "${DATASET:-rep}" \
    ${YEARS:+--years "$YEARS"} \
    ${MONTHS:+--months "$MONTHS"}
