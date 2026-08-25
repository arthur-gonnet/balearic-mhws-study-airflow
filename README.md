# <h1 style="text-align: center;"> Balearic MHWs study - operational pipeline </h1>

This repository is an operational fork of the code originally used to produce figures for the master thesis entitled "Marine heatwaves in the Balearic Islands region" (see the [original repository](https://github.com/arthur-gonnet/balearic-mhws-study)). It turns that thesis code into a containerized, orchestrated pipeline.

## Overview

This code computes marine heatwave (MHW) metrics in the Balearic Islands region, orchestrated as a scheduled, unattended pipeline rather than a set of notebooks run by hand:

- **Docker Compose** runs a full Airflow 3 stack (postgres, redis, scheduler, dag-processor, worker, triggerer, apiserver).
- **Airflow** orchestrates the pipeline: ingest raw data, compute MHW metrics, validate the result.
- **Zarr** is the storage format for both raw ingested datasets and computed MHW results, replacing the original per-month NetCDF files - chunked and append-friendly, so incremental ingestion and parallel dask computation are both cheap.
- **Slurm** sbatch scripts let the compute-heavy stage run on an HPC cluster instead of in-process, via SSH from the Airflow DAG, when one is configured.

## What is in here?

- `dags/` : the Airflow DAG (`balearic_mhws.py`) - ingest → compute → validate.
- `pipelines/mhws_pipeline.py` : the pipeline's CLI entrypoint. Both the DAG (local runs) and the Slurm sbatch scripts (cluster runs) call into this - no logic is duplicated between the two.
- `src/balearic_mhws/` : the installable package - `data/` (Zarr I/O + Copernicus Marine ingestion), `processing/` (the MHW detection algorithm and its dask-parallelized wrapper), `config.py` (paths, stats metadata, resolved from environment variables).
- `src/balearic_mhws_OLD/` : the original thesis code, kept as reference until the plotting/report generation part of it (`basic_plotter.py`, `utils.py`) is ported to the new architecture.
- `hpc/slurm/` : sbatch scripts for running ingestion/computation on a Slurm cluster.
- `config/` : Airflow configuration (`airflow.cfg`).
- `tests/` : pytest suite covering the computation, I/O, and ingestion logic without needing network access or real data.
- `data/` : raw and processed Zarr stores (gitignored - populated by running the pipeline).

## How should this be run?

1. Copy `.env.example` to `.env` and fill in what you need (see the comments in that file - at minimum `AIRFLOW_UID` on Linux; `USERNAME_COPERNICUS`/`PASSWORD_COPERNICUS` to actually download data; `SLURM_SSH_HOST` and friends only if pointing compute at a real cluster).
2. `docker compose build`
3. `docker compose up airflow-init` (first time only, to set up the database and admin user)
4. `docker compose up -d`
5. Open the Airflow UI at `http://localhost:8080` (default login `airflow`/`airflow`), or trigger the DAG from the CLI:
   ```
   docker compose exec airflow-apiserver airflow dags trigger balearic_mhws \
     --conf '{"dataset": "rep", "region": "balears", "clim_start": 1987, "clim_end": 2021}'
   ```

Running the pipeline CLI directly (what the Slurm sbatch scripts do) works the same way outside Airflow, given `PYTHONPATH` includes `src` and the repo root:
```
python -m pipelines.mhws_pipeline download --dataset rep --years 2020:2023
python -m pipelines.mhws_pipeline compute --dataset rep --clim-start 1987 --clim-end 2021
```

## External data

Raw data is downloaded from the Copernicus Marine Service by the pipeline's `download` stage (`src/balearic_mhws/data/download.py`) directly into Zarr stores under `data/raw/` - it isn't checked into the repository. A Copernicus Marine account is required; create one at https://data.marine.copernicus.eu/register.

## License

This code has been developed by Arthur Gonnet, and is licensed under the GNU General Public License v3.0 (GPLv3).

This code includes a modified version of the *marineHeatWaves* module for python developed by Eric C. J. Oliver (see https://github.com/ecjoliver/marineHeatWaves), under GPLv3 license.

This code uses the *pyMannKendall* module for python developed by Md. Manjurul Hussain and Ishtiak Mahmud (see https://github.com/mmhs013/pyMannKendall), under MIT license.

This work makes use of E.U. Copernicus Marine Service Information; https://doi.org/10.48670/moi-00173; https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1, under a permissive license.

## Contact

> Arthur Gonnet <br>
> br.arthur.gonnet@gmail.com <br>
> https://github.com/arthur-gonnet/
