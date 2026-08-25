#!/bin/bash

#SBATCH --job-name=mhw-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mhw-compute-%A_%a.out
# TODO: set an --array range once the compute task is actually chunked across jobs (e.g. one
# array task per region/depth level). Left unset for now - sizing depends on the target cluster
# and dataset size, which aren't known without a real cluster to tune against.
# #SBATCH --array=0-N

set -euo pipefail

# TODO: activate this cluster's Python environment, e.g.:
#   module load python/3.11
#   source /path/to/venv/bin/activate

PROJECT_DIR="${SLURM_REMOTE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}"
export BALEARIC_DATA_DIR="${BALEARIC_DATA_DIR:-${PROJECT_DIR}/data}"

cd "$PROJECT_DIR"

python -m pipelines.mhws_pipeline compute \
    --dataset "${DATASET:-rep}" \
    --region "${REGION:-balears}" \
    --ds-type "${DS_TYPE:-yearly}" \
    --clim-start "${CLIM_START:-1987}" \
    --clim-end "${CLIM_END:-2021}" \
    --task-id "${SLURM_ARRAY_TASK_ID:-0}"
