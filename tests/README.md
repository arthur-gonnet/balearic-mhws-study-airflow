# tests folder

The pytest suite. The tests use synthetic datasets written to temporary directories, so they need
neither network access nor the real data.

## What is in here?

 - `test_compute_mhws.py` :
     Checks the MHW computation on a synthetic temperature serie with a heatwave injected into it.
     Covers both the yearly and the per-event metrics, the formatting done by
     `get_mhw_ts_from_ds`, and that an entirely NaN serie gives NaN metrics rather than failing.

 - `test_io.py` :
     Checks the Zarr round-trips: the spatio-temporal selectors of `open_rep`/`open_medrea`, and
     the three paths of `write_zarr_incremental` (creating a store, appending to it, and merging
     data that falls before or inside the range already stored).

 - `test_download.py` :
     Checks that the ingestion correctly works out which (year, month) pairs are missing from a
     store, and the normalisation applied to a downloaded dataset (renamed coordinates, the
     Kelvin to Celsius conversion for REP, the timestamps floored to midnight for MEDREA).

 - `test_pipeline.py` :
     Checks the CLI's range parsing and that the compute stage defaults to the configured
     climatology period.

 - `test_dags.py` :
     Checks that the DAGs build with the expected tasks and dependencies, that both DAG files
     import cleanly, and that `validate_result` accepts a computed dataset and rejects one whose
     `count` is entirely NaN.

## How should this be run?

```
python -m pytest tests/
```

`test_dags.py` is skipped outside the container, as `apache-airflow` is only installed in the
Docker image and not in the project dependencies. Running it needs both `pytest` and this folder
inside the container, neither of which is there by default:

```
docker compose exec --user airflow airflow-worker python -m pip install pytest
docker cp tests balearic-mhws-study-airflow-airflow-worker-1:/tmp/tests
docker cp conftest.py balearic-mhws-study-airflow-airflow-worker-1:/tmp/conftest.py
docker compose exec --user airflow airflow-worker bash -c \
  'cd /tmp && PYTHONPATH=/opt/airflow/dags:/opt/airflow/project/src:/opt/airflow/project \
   python -m pytest /tmp/tests/test_dags.py'
```
