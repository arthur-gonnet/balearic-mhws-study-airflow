"""Tests for balearic_mhws.data.io - Zarr round-trips using temp directories, no real data needed."""

import numpy as np
import pandas as pd
import xarray as xr

from balearic_mhws import config
from balearic_mhws.data.io import load_mhws, save_mhws, write_zarr_incremental


def _synthetic_mhws_dataset():
    return xr.Dataset(
        {"total_days": (("year", "lat", "lon"), np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3))},
        coords={"year": [2000, 2001], "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )


def test_save_and_load_mhws_roundtrip(tmp_path, monkeypatch):
    pattern = str(tmp_path / "mhws" / "{type}" / "{dataset}_mhws_{region}{detrended}_{clim_start}_{clim_end}.zarr")
    monkeypatch.setattr(config, "MHWS_ZARR_PATTERN", pattern)

    ds = _synthetic_mhws_dataset()

    save_mhws(ds, ds_type="yearly", dataset_used="rep", region="balears", clim_period=(2000, 2001), progress_bar=False)
    loaded = load_mhws(ds_type="yearly", dataset_used="rep", region="balears", clim_period=(2000, 2001))

    xr.testing.assert_equal(loaded["total_days"].compute(), ds["total_days"])


def test_write_zarr_incremental_creates_then_appends(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time1 = pd.date_range("2000-01-01", periods=5, freq="D")
    ds1 = xr.Dataset({"T": ("time", np.arange(5.0))}, coords={"time": time1})
    write_zarr_incremental(ds1, store_path)

    time2 = pd.date_range("2000-01-06", periods=3, freq="D")
    ds2 = xr.Dataset({"T": ("time", np.arange(3.0) + 100)}, coords={"time": time2})
    write_zarr_incremental(ds2, store_path)

    result = xr.open_zarr(store_path, consolidated=True)
    assert result.sizes["time"] == 8
    np.testing.assert_array_equal(result["T"].values, np.concatenate([ds1["T"].values, ds2["T"].values]))


def test_write_zarr_incremental_is_idempotent(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time = pd.date_range("2000-01-01", periods=5, freq="D")
    ds = xr.Dataset({"T": ("time", np.arange(5.0))}, coords={"time": time})

    write_zarr_incremental(ds, store_path)
    write_zarr_incremental(ds, store_path)  # re-running with the same data should be a no-op

    result = xr.open_zarr(store_path, consolidated=True)
    assert result.sizes["time"] == 5
