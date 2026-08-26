# pipelines folder

The pipeline's CLI entrypoint. Both the Airflow DAGs and the Slurm sbatch scripts call into this
CLI, so the download and compute logic is defined in only one place.

## What is in here?

 - `mhws_pipeline.py` :
     The CLI itself, with a `download` and a `compute` subcommand.

## How should this be run?

Given `PYTHONPATH` includes `src` and the repo root:

```
python -m pipelines.mhws_pipeline download --dataset rep --years 1982:2024
python -m pipelines.mhws_pipeline download --dataset medrea --years 1987:2023
python -m pipelines.mhws_pipeline download --dataset bathy
python -m pipelines.mhws_pipeline compute --dataset rep
```

Year and month ranges are given as `'start:stop'`, the stop being exclusive. A single value is
also accepted, so `--years 2020` means just 2020.

### `download`

Ingests raw data into the raw Zarr stores. Only the (year, month) pairs missing from the target
store are downloaded.

 - `--dataset` (required) :
     One of `rep`, `medrea`, `bathy`. The bathymetry is a static one-shot download and takes no
     year or month range.

 - `--years`, `--months` (optional) :
     Ranges to ingest. If omitted, the ingest functions use their own defaults, which cover the
     whole time range of the dataset.

### `compute`

Computes MHW metrics from an ingested dataset and saves them to the processed Zarr stores.

 - `--dataset` (required) :
     One of `rep`, `medrea`.

 - `--region` (default `balears`) :
     Region to compute over.

 - `--ds-type` (default `yearly`) :
     `yearly` for per-year metrics, `all_events` for per-event metrics.

 - `--clim-start`, `--clim-end` (default `config.CLIMATOLOGY_PERIOD`) :
     Climatology period used. The DAGs always pass `config.CLIMATOLOGY_PERIOD` so that the results
     of every run stay comparable.

 - `--task-id` (optional) :
     Slurm array task id. Unused for now, reserved for future chunked runs.
