"""
Tests for the DAG factory (`dags/_balearic_mhws_dag_factory.py`) - its structure, and the
validation task guarding the computed datasets.

`apache-airflow` is only installed inside the project's Docker image (see `Dockerfile`), not as a
project dependency, so it's normal for these to be skipped when running the suite on a bare host -
run them via `docker compose exec airflow-worker pytest tests/test_dags.py` for real coverage.
"""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("airflow")

from _balearic_mhws_dag_factory import build_dag  # noqa: E402

from balearic_mhws import config  # noqa: E402
from balearic_mhws.data.io import save_mhws  # noqa: E402


@pytest.mark.parametrize("dataset", ["rep", "medrea"])
def test_build_dag_structure(dataset):
    dag = build_dag(dataset)

    assert dag.dag_id == f"balearic_mhws_{dataset}"
    assert set(dag.task_ids) == {"ingest_data", "compute_mhws_task", "validate_result", "plot_diagnostic"}

    # ingest_data -> compute_mhws_task -> validate_result -> plot_diagnostic, with plot_diagnostic
    # also depending directly on compute_mhws_task's result (not just running after validate_result).
    assert dag.get_task("ingest_data").upstream_task_ids == set()
    assert dag.get_task("compute_mhws_task").upstream_task_ids == {"ingest_data"}
    assert dag.get_task("validate_result").upstream_task_ids == {"compute_mhws_task"}
    assert dag.get_task("plot_diagnostic").upstream_task_ids == {"compute_mhws_task", "validate_result"}


def test_dag_files_are_importable_with_expected_dag_id():
    # Regression check for the DAG-discovery gotcha: Airflow pre-filters files by scanning their
    # raw text for "airflow"/"dag" substrings before importing them - a thin file that only
    # transitively imports Airflow through another module gets silently skipped and never even
    # attempted. This at least catches the file failing to import/build a DAG at all; it can't
    # catch the text-scan skip itself, since that happens before Python ever runs this test.
    import balearic_mhws_medrea
    import balearic_mhws_rep

    assert balearic_mhws_rep.balearic_mhws_rep.dag_id == "balearic_mhws_rep"
    assert balearic_mhws_medrea.balearic_mhws_medrea.dag_id == "balearic_mhws_medrea"


def _validate_result_callable():
    return build_dag("rep").get_task("validate_result").python_callable


def _run_args():
    return {
        "dataset": "rep",
        "region": "balears",
        "ds_type": "yearly",
        "clim_start": 2000,
        "clim_end": 2001,
    }


def _write_mhws_store(tmp_path, monkeypatch, count_values):
    pattern = str(tmp_path / "mhws" / "{type}" / "{dataset}_mhws_{region}{detrended}_{clim_start}_{clim_end}.zarr")
    monkeypatch.setattr(config, "MHWS_ZARR_PATTERN", pattern)

    ds = xr.Dataset(
        {"count": (("year", "lat", "lon"), count_values)},
        coords={"year": [2000, 2001], "lat": np.arange(2.0), "lon": np.arange(2.0)},
    )
    save_mhws(ds, ds_type="yearly", dataset_used="rep", region="balears", clim_period=(2000, 2001), progress_bar=False)


def test_validate_result_accepts_a_computed_dataset(tmp_path, monkeypatch):
    _write_mhws_store(tmp_path, monkeypatch, np.ones((2, 2, 2)))

    _validate_result_callable()(_run_args())


def test_validate_result_rejects_an_entirely_nan_count(tmp_path, monkeypatch):
    # An entirely NaN 'count' means every grid cell was land or no-data, so nothing was computed.
    _write_mhws_store(tmp_path, monkeypatch, np.full((2, 2, 2), np.nan))

    with pytest.raises(ValueError, match="entirely NaN"):
        _validate_result_callable()(_run_args())
