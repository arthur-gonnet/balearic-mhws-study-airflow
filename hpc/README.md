# hpc folder

Everything needed to run the compute-heavy stage of the pipeline on a Slurm cluster, plus a local
single-node cluster to test that path without access to a real one.

The DAG only submits jobs when `SLURM_SSH_HOST` is set. When it is not set, the computation runs
in-process on the Airflow worker instead, which is convenient for local development.

## What is in here?

 - `slurm/compute_mhws.sbatch` :
     Batch script computing the MHW metrics. This is the one the DAG submits.

 - `slurm/download_data.sbatch` :
     Batch script ingesting raw data, for running the download stage on the cluster.

 - `slurm/run_pipeline.sbatch` :
     Batch script running the whole pipeline in one job.

 - `test-cluster/` :
     A single-node Slurm cluster in a container, with an SSH server, used to test the submission
     path locally. Started with the `slurm-test` Compose profile.

## Configuration

The following variables, set in `.env`, wire the DAG to a cluster.

 - `SLURM_SSH_HOST` :
     Host to submit to. Leaving it empty makes the computation run in-process instead.

 - `SLURM_SSH_USER` :
     User to connect as.

 - `SLURM_REMOTE_PROJECT_DIR` :
     Path of the project on the cluster. It is forwarded explicitly to the job, as a
     non-interactive SSH shell does not reliably source `~/.bashrc`.

 - `SLURM_PARTITION` :
     Partition to submit to. Optional.

## How should this be run?

To test the submission path locally, start the bundled cluster and point the variables at it:

```
docker compose --profile slurm-test up -d
```

```
SLURM_SSH_HOST=slurm-test-cluster
SLURM_SSH_USER=slurmuser
SLURM_REMOTE_PROJECT_DIR=/home/slurmuser/project
SLURM_PARTITION=debug
```

The DAG then submits the job, polls it, and prints the job output into the Airflow task log as it
runs. The job also writes its own output to `logs/mhw-compute-<jobid>_*.out`.

## Adapting to a real cluster

The sbatch scripts carry TODOs where a real cluster needs its own settings:

 - Activating the Python environment, for instance with `module load python/3.11` or by sourcing a
   virtualenv. The scripts assume `python3` already has the dependencies.
 - The `--array` range, left unset until the computation is actually chunked across jobs, for
   instance one array task per region or per depth level.
 - The resource requests (`--cpus-per-task`, `--mem`, `--time`), sized for the test cluster.

The scripts call the same CLI described in [../pipelines/README.md](../pipelines/README.md), so
nothing else has to change to run on a different cluster.
