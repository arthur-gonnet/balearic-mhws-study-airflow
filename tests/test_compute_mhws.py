"""Tests for balearic_mhws.processing.compute_mhws - pure computation, no I/O or network needed."""

import numpy as np
import pandas as pd
import xarray as xr

from balearic_mhws import config
from balearic_mhws.processing.compute_mhws import compute_mhw_yearly


def _synthetic_temperature(n_years=3, n_lat=2, n_lon=2, seed=0):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2000-01-01", periods=365 * n_years, freq="D")

    day_of_year = time.dayofyear.to_numpy()
    seasonal = 18 + 6 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    noise = rng.normal(scale=0.5, size=(len(time), n_lat, n_lon))
    data = seasonal[:, None, None] + noise

    # Inject an obvious marine heatwave: a hot spike well above the seasonal cycle for 10 days.
    data[100:110, :, :] += 8

    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": np.arange(n_lat, dtype=float), "lon": np.arange(n_lon, dtype=float)},
        name="T",
    )


def test_compute_mhw_yearly_shape_and_stats():
    da = _synthetic_temperature()

    ds_mhws = compute_mhw_yearly(da, clim_period=(2000, 2001)).compute()

    assert set(config.mhws_stats).issubset(set(ds_mhws.data_vars))
    assert "year" in ds_mhws.coords
    assert ds_mhws.sizes["year"] == 3

    # At least the year with the injected spike should register some MHW days somewhere.
    assert float(ds_mhws["total_days"].sum()) > 0


def test_compute_mhw_yearly_all_nan_series_returns_nan():
    da = _synthetic_temperature(n_years=1, n_lat=1, n_lon=1)
    da[:] = np.nan

    ds_mhws = compute_mhw_yearly(da, clim_period=(2000, 2000)).compute()

    assert bool(ds_mhws["total_days"].isnull().all())
