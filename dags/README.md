# dags folder

The Airflow DAGs orchestrating the pipeline. There is one DAG per dataset, both built from a
shared factory so that their logic is written only once.

One DAG per dataset, rather than a single DAG taking the dataset as a parameter, because the
schedule is fixed per DAG and not per trigger. REP and MEDREA come from different Copernicus
Marine products, updated at different cadences, so each needs its own schedule.

## What is in here?

 - `_balearic_mhws_dag_factory.py` :
     `build_dag(dataset)`, building the whole DAG for one dataset (`'rep'` or `'medrea'`). It is
     not a DAG file itself, as it registers no DAG at module level.

 - `balearic_mhws_rep.py` :
     Registers the `balearic_mhws_rep` DAG, calling `build_dag("rep")`.

 - `balearic_mhws_medrea.py` :
     Registers the `balearic_mhws_medrea` DAG, calling `build_dag("medrea")`.

## The tasks

Both DAGs run the same four tasks.

 1. `ingest_data` :
     Downloads the months missing from the dataset's raw Zarr store. Months already ingested are
     skipped, so an interrupted run resumes where it left off.

 2. `compute_mhws_task` :
     Computes the MHW metrics. Runs in-process by default. If `SLURM_SSH_HOST` is set, it submits
     an sbatch job over SSH instead and streams the job output back into the task log, as
     described in [../hpc/README.md](../hpc/README.md).

 3. `validate_result` :
     Opens the computed Zarr store and fails the run if it has no `year` coordinate, or if `count`
     is entirely NaN. An entirely NaN `count` means no valid grid cell was computed at all.

 4. `plot_diagnostic` :
     Saves a map of `total_days` for the most recent year to `data/products/diagnostics/`, as a
     quick visual check. Skipped for `ds_type='all_events'`, which has no map-shaped output.

## Trigger parameters

 - `region` (default `'balears'`) :
     Region to compute over.

 - `ds_type` (default `'yearly'`) :
     `'yearly'` for per-year metrics, `'all_events'` for per-event metrics.

 - `years`, `months` (default empty) :
     Ranges to ingest, given as `'start:stop'` with an exclusive stop, or as a single value. If
     left empty, the ingest functions use their own defaults, which cover the whole time range of
     the dataset.

The climatology period is not a trigger parameter. It is always `config.CLIMATOLOGY_PERIOD`, so
that the results of every run stay comparable, and so that a run can never end up with a
climatology period equal to the analysed year.

## How should this be run?

From the Airflow UI at http://localhost:8080, or from the CLI:

```
docker compose exec airflow-scheduler airflow dags trigger balearic_mhws_rep \
  --conf '{"years": "2020:2023"}'
```

New DAGs are paused when created, and the runs of a paused DAG stay queued forever. Unpause the
DAG first:

```
docker compose exec airflow-scheduler airflow dags unpause balearic_mhws_rep
```

## Troubleshooting

**> I added or renamed a DAG file and Airflow never picks it up.** <br>
Airflow looks for the substrings `airflow` or `dag` in the raw text of a file before importing it,
so a file that only reaches Airflow through another module is skipped silently. This is why
`balearic_mhws_rep.py` and `balearic_mhws_medrea.py` both carry a real `from airflow.sdk import
DAG` import. After changing the structure of the DAG files, restart the dag-processor and the
scheduler with `docker compose restart airflow-dag-processor airflow-scheduler`.
