#!/bin/bash

# Chains download_data.sbatch -> compute_mhws.sbatch on the cluster via Slurm job dependencies.
#
# This is a plain launcher script, not itself an sbatch job (a job's #SBATCH directives can't
# express a dependency on a job that doesn't exist yet at submission time) - run it directly on
# the login node, or have the Airflow DAG's SSH task call it instead of submitting a single job.
#
# Usage: hpc/slurm/run_pipeline.sbatch [DATASET] [REGION] [DS_TYPE] [CLIM_START] [CLIM_END]

set -euo pipefail

DATASET="${1:-rep}"
REGION="${2:-balears}"
DS_TYPE="${3:-yearly}"
CLIM_START="${4:-1987}"
CLIM_END="${5:-2021}"

download_job=$(sbatch --parsable --export=ALL,DATASET="$DATASET" hpc/slurm/download_data.sbatch)
echo "Submitted download job: $download_job"

compute_job=$(sbatch --parsable \
    --dependency=afterok:"$download_job" \
    --export=ALL,DATASET="$DATASET",REGION="$REGION",DS_TYPE="$DS_TYPE",CLIM_START="$CLIM_START",CLIM_END="$CLIM_END" \
    hpc/slurm/compute_mhws.sbatch)
echo "Submitted compute job: $compute_job (depends on $download_job)"
