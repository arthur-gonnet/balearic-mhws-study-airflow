# <h1 style="text-align: center;"> Balearic MHWs study - operational pipeline </h1>

This repository is an operational fork of the code originally used to produce figures for the master thesis entitled "Marine heatwaves in the Balearic Islands region" (see the [original repository](https://github.com/arthur-gonnet/balearic-mhws-study)). It turns that thesis code into a containerized, orchestrated pipeline.

## Overview

This code computes marine heatwave (MHW) metrics in the Balearic Islands region, orchestrated as a scheduled, unattended pipeline rather than a set of notebooks run by hand:

- **Docker Compose** runs a full Airflow 3 stack (postgres, redis, scheduler, dag-processor, worker, triggerer, apiserver).
- **Airflow** orchestrates the pipeline: ingest raw data, compute MHW metrics, validate the result, and save a diagnostic figure.
- **Zarr** is the storage format for both raw ingested datasets and computed MHW results, replacing the original per-month NetCDF files - chunked and append-friendly, so incremental ingestion and parallel dask computation are both cheap.
- **Slurm** sbatch scripts let the compute-heavy stage run on an HPC cluster instead of in-process, via SSH from the Airflow DAG, when one is configured.

## What is in here?

 - `dags/` :
     The Airflow DAGs - one per dataset (`balearic_mhws_rep`, `balearic_mhws_medrea`), both built
     by a shared factory. See [dags/README.md](dags/README.md).

 - `pipelines/` :
     The pipeline's CLI entrypoint. Both the DAGs (local runs) and the Slurm sbatch scripts
     (cluster runs) call into this - no logic is duplicated between the two.
     See [pipelines/README.md](pipelines/README.md).

 - `src/balearic_mhws/` :
     The installable package - data I/O and ingestion, the MHW computation, and plotting.
     See [src/balearic_mhws/README.md](src/balearic_mhws/README.md).

 - `src/notebooks/` :
     Python notebooks from the thesis workflow, kept as an interactive way to explore the data.

 - `hpc/` :
     Slurm sbatch scripts, plus a local single-node Slurm cluster for testing the SSH/sbatch
     submission path without a real cluster. See [hpc/README.md](hpc/README.md).

 - `config/` :
     Airflow configuration (`airflow.cfg`).

 - `tests/` :
     Pytest suite covering the computation, I/O, and ingestion logic without needing network
     access or real data. See [tests/README.md](tests/README.md).

 - `data/` :
     Raw and processed Zarr stores (gitignored - populated by running the pipeline).

## Installation

**Prerequisites**

 - Docker and Docker Compose.
 - A free Copernicus Marine Service account, required to download any data
   (register at https://data.marine.copernicus.eu/register).

**Setup**

 1. Copy `.env.example` to `.env` and fill in the values you need - each one is documented inline
    in that file. At minimum:
    - `AIRFLOW_UID` : on Linux, set to your own uid (`id -u`) so files written into mounted
      volumes aren't owned by root.
    - `USERNAME_COPERNICUS` / `PASSWORD_COPERNICUS` : needed to actually download data.
    - `SLURM_SSH_HOST` and friends : only if pointing the compute stage at a real cluster.

 2. Build the image (installs the scientific stack and this package's dependencies):
    ```
    docker compose build
    ```

 3. Initialise the Airflow database and admin user (first time only):
    ```
    docker compose up airflow-init
    ```

 4. Start the stack:
    ```
    docker compose up -d
    ```

 5. Unpause the DAGs - new DAGs are paused by default, and a paused DAG's manually-triggered runs
    stay queued forever:
    ```
    docker compose exec airflow-scheduler airflow dags unpause balearic_mhws_rep
    docker compose exec airflow-scheduler airflow dags unpause balearic_mhws_medrea
    ```

## How should this be run?

There are three ways in, all running the same underlying code.

**1. The Airflow UI** (the operational path)

Open http://localhost:8080 (default login `airflow`/`airflow`), then trigger `balearic_mhws_rep`
or `balearic_mhws_medrea`. Trigger from the CLI instead with:

```
docker compose exec airflow-scheduler airflow dags trigger balearic_mhws_rep \
  --conf '{"years": "2020:2023"}'
```

Available trigger params are `region`, `ds_type`, `years` and `months` - see
[dags/README.md](dags/README.md). The climatology period is **not** a parameter: it is fixed at
`config.CLIMATOLOGY_PERIOD` (1987-2021) so every run's results stay comparable.

> **Always pass `years`/`months` explicitly.** Omitting them falls through to the ingest
> functions' own defaults, which cover the datasets' full time range - i.e. "ingest everything",
> not "ingest nothing new".

**2. The pipeline CLI** (what the Slurm sbatch scripts call)

Works the same way outside Airflow, given `PYTHONPATH` includes `src` and the repo root:

```
python -m pipelines.mhws_pipeline download --dataset rep --years 2020:2023
python -m pipelines.mhws_pipeline compute --dataset rep
```

See [pipelines/README.md](pipelines/README.md) for the full reference.

**3. The notebooks** (interactive exploration)

`src/notebooks/` holds the thesis-era notebooks, useful for exploring data or producing figures by
hand. They call into the same `balearic_mhws` package the pipeline uses.

## External data

Raw data is downloaded from the Copernicus Marine Service by the pipeline's `download` stage
(`src/balearic_mhws/data/download.py`) directly into Zarr stores under `data/raw/` - it isn't
checked into the repository. A Copernicus Marine account is required; create one at
https://data.marine.copernicus.eu/register.

Ingestion is idempotent: each run only downloads the (year, month) pairs missing from the target
Zarr store, so re-running an overlapping range is a no-op for months already covered, and an
interrupted run resumes where it left off.

## Troubleshooting

**> The pipeline fails saying Copernicus Marine credentials aren't configured.** <br>
Set `USERNAME_COPERNICUS`/`PASSWORD_COPERNICUS` in `.env` and restart the stack
(`docker compose up -d`). They are read from the environment - there are no interactive prompts,
so the pipeline can run unattended.

**> I triggered a DAG run but it stays queued and never starts.** <br>
The DAG is probably still paused (new DAGs are paused by default). Unpause it - see step 5 of the
installation section.

**> I edited a DAG file and Airflow doesn't pick up the change.** <br>
Task-body edits are picked up automatically. After changing the *structure* of the DAG files
(adding, renaming or splitting files), restart the dag-processor and scheduler:
`docker compose restart airflow-dag-processor airflow-scheduler`.

**> Where do I find the logs?** <br>
Airflow task logs are in the UI, or under `logs/` on the host. When the compute stage runs on a
Slurm cluster, the job's own output also lands in `logs/mhw-compute-<jobid>_*.out`.

**> How do I run the compute stage on a real cluster instead of in-process?** <br>
Set `SLURM_SSH_HOST`, `SLURM_SSH_USER`, `SLURM_REMOTE_PROJECT_DIR` (and optionally
`SLURM_PARTITION`) in `.env`. When `SLURM_SSH_HOST` is set, the DAG submits an sbatch job over SSH
instead of computing locally. To try that path without a real cluster, start the bundled
single-node test cluster with `docker compose --profile slurm-test up -d` - see
[hpc/README.md](hpc/README.md).

**> The Copernicus download logs show `ERROR` lines that look like normal messages.** <br>
The `copernicusmarine` library writes its own INFO/WARNING messages and its progress bar to
stderr, and Airflow labels anything arriving on stderr as `ERROR` regardless of content. Judge a
task by its final state, not by those line labels.

## License

This code has been developed by Arthur Gonnet, and is licensed under the GNU General Public License v3.0 (GPLv3).

This code includes a modified version of the *marineHeatWaves* module for python developed by Eric C. J. Oliver (see https://github.com/ecjoliver/marineHeatWaves), under GPLv3 license.

This code uses the *pyMannKendall* module for python developed by Md. Manjurul Hussain and Ishtiak Mahmud (see https://github.com/mmhs013/pyMannKendall), under MIT license.

This work makes use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.

## Contact

> Arthur Gonnet <br>
> br.arthur.gonnet@gmail.com <br>
> https://github.com/arthur-gonnet/
