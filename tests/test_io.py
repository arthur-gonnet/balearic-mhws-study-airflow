"""Tests for balearic_mhws.data.io - Zarr round-trips using temp directories, no real data needed."""

import numpy as np
import pandas as pd
import xarray as xr

from balearic_mhws import config
from balearic_mhws.data.io import load_mhws, open_medrea, open_rep, save_mhws, write_zarr_incremental


def _synthetic_mhws_dataset():
    return xr.Dataset(
        {"total_days": (("year", "lat", "lon"), np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3))},
        coords={"year": [2000, 2001], "lat": np.arange(3.0), "lon": np.arange(3.0)},
    )


# Wider than the 'balears' region_selector (lon [-0.9, 5.1], lat [37.6, 41.1]) so selecting it
# actually crops something, proving the selector is applied rather than trivially passing through.
_LAT = np.arange(35.0, 44.0)  # 35..43, balears keeps 38..41
_LON = np.arange(-3.0, 8.0)  # -3..7, balears keeps 0..5


def _synthetic_rep_store(tmp_path):
    time = pd.date_range("2000-01-01", periods=5, freq="D")
    data = np.arange(len(time) * len(_LAT) * len(_LON), dtype=float).reshape(len(time), len(_LAT), len(_LON))
    ds = xr.Dataset({"T": (("time", "lat", "lon"), data)}, coords={"time": time, "lat": _LAT, "lon": _LON})

    store_path = tmp_path / "rep.zarr"
    ds.to_zarr(store_path, mode="w", consolidated=True)
    return store_path


def _synthetic_medrea_store(tmp_path):
    time = pd.date_range("2000-01-01", periods=3, freq="D")
    # One extra depth (2500) beyond config.MEDREA_DEFAULT_DEPTH_LEVELS, so the default-depth
    # selector has something to actually exclude.
    depth = np.array(config.MEDREA_DEFAULT_DEPTH_LEVELS + [2500], dtype=float)
    shape = (len(time), len(depth), len(_LAT), len(_LON))
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    ds = xr.Dataset(
        {"T": (("time", "depth", "lat", "lon"), data)},
        coords={"time": time, "depth": depth, "lat": _LAT, "lon": _LON},
    )

    store_path = tmp_path / "medrea.zarr"
    ds.to_zarr(store_path, mode="w", consolidated=True)
    return store_path


def test_open_rep_applies_region_and_time_selectors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REP_ZARR", _synthetic_rep_store(tmp_path))

    ds = open_rep(time_selector=slice("2000-01-01", "2000-01-02"))

    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds.lat.values, np.arange(38.0, 42.0))
    np.testing.assert_array_equal(ds.lon.values, np.arange(0.0, 6.0))


def test_open_rep_region_selector_none_keeps_full_extent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REP_ZARR", _synthetic_rep_store(tmp_path))

    ds = open_rep(region_selector=None)

    assert ds.sizes["lat"] == len(_LAT)
    assert ds.sizes["lon"] == len(_LON)


def test_open_medrea_default_depth_and_region(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDREA_ZARR", _synthetic_medrea_store(tmp_path))

    ds = open_medrea()

    assert sorted(ds.depth.values.tolist()) == sorted(config.MEDREA_DEFAULT_DEPTH_LEVELS)
    np.testing.assert_array_equal(ds.lat.values, np.arange(38.0, 42.0))
    np.testing.assert_array_equal(ds.lon.values, np.arange(0.0, 6.0))


def test_open_medrea_depth_selector_none_keeps_full_water_column(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDREA_ZARR", _synthetic_medrea_store(tmp_path))

    ds = open_medrea(depth_selector=None, region_selector=None)

    assert ds.sizes["depth"] == len(config.MEDREA_DEFAULT_DEPTH_LEVELS) + 1


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


def test_write_zarr_incremental_backfills_gap_across_multiple_native_chunks(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time1 = pd.date_range("2000-01-01", periods=5, freq="D")
    ds1 = xr.Dataset({"T": ("time", np.arange(5.0))}, coords={"time": time1})
    write_zarr_incremental(ds1, store_path)

    time2 = pd.date_range("2000-01-11", periods=5, freq="D")  # skip days 6-10, leaving a gap
    ds2 = xr.Dataset({"T": ("time", np.arange(5.0) + 100)}, coords={"time": time2})
    write_zarr_incremental(ds2, store_path)

    # Two separate writes land as two separate on-disk chunks - backfilling the gap now exercises
    # the merge-rewrite path's per-native-chunk loop, not just a single chunk.
    assert xr.open_zarr(store_path, consolidated=True).chunksizes["time"] == (5, 5)

    time3 = pd.date_range("2000-01-06", periods=5, freq="D")
    ds3 = xr.Dataset({"T": ("time", np.arange(5.0) + 200)}, coords={"time": time3})
    write_zarr_incremental(ds3, store_path)

    result = xr.open_zarr(store_path, consolidated=True).compute()
    assert result.sizes["time"] == 15
    np.testing.assert_array_equal(result["T"].values, np.concatenate([ds1["T"].values, ds3["T"].values, ds2["T"].values]))


def test_write_zarr_incremental_backfills_before_existing_range(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time1 = pd.date_range("2000-01-10", periods=10, freq="D")
    ds1 = xr.Dataset({"T": ("time", np.arange(10.0))}, coords={"time": time1})
    write_zarr_incremental(ds1, store_path)

    # Backfill days before the existing range - takes the merge-and-rewrite fallback path.
    time2 = pd.date_range("2000-01-01", periods=5, freq="D")
    ds2 = xr.Dataset({"T": ("time", np.arange(5.0) + 100)}, coords={"time": time2})
    write_zarr_incremental(ds2, store_path)

    result = xr.open_zarr(store_path, consolidated=True).compute()
    assert result.sizes["time"] == 15
    np.testing.assert_array_equal(result["T"].values, np.concatenate([ds2["T"].values, ds1["T"].values]))
