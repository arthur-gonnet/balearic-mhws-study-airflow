"""
Tests for balearic_mhws.data.download's pure logic - no network/credentials needed.

The actual `copernicusmarine.subset()` calls are not tested here; only the idempotency logic
that decides which (year, month) pairs still need downloading, and the normalisation applied
to a downloaded dataset before it is written to a store.
"""

import numpy as np
import pandas as pd
import xarray as xr

from balearic_mhws.data.download import _missing_year_months, _normalize_medrea, _normalize_rep


def test_missing_year_months_empty_store(tmp_path):
    store_path = tmp_path / "does-not-exist.zarr"

    missing = _missing_year_months(store_path, years=[2020], months=[1, 2, 3])

    assert missing == {2020: [1, 2, 3]}


def test_missing_year_months_partial_coverage(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time = pd.to_datetime(["2020-01-15", "2020-02-15"])
    ds = xr.Dataset({"T": ("time", np.zeros(2))}, coords={"time": time})
    ds.to_zarr(store_path, mode="w", consolidated=True)

    missing = _missing_year_months(store_path, years=[2020], months=[1, 2, 3])

    assert missing == {2020: [3]}


def test_missing_year_months_fully_covered(tmp_path):
    store_path = tmp_path / "raw.zarr"

    time = pd.to_datetime(["2020-01-15"])
    ds = xr.Dataset({"T": ("time", np.zeros(1))}, coords={"time": time})
    ds.to_zarr(store_path, mode="w", consolidated=True)

    missing = _missing_year_months(store_path, years=[2020], months=[1])

    assert missing == {}


def test_normalize_rep():
    # Named as the raw Copernicus product does, with the extra variables the pipeline drops.
    ds = xr.Dataset(
        {
            "analysed_sst": (("time", "latitude", "longitude"), np.full((1, 2, 2), 300.15)),
            "analysis_error": (("time", "latitude", "longitude"), np.zeros((1, 2, 2))),
            "mask": (("time", "latitude", "longitude"), np.zeros((1, 2, 2))),
        },
        coords={"time": pd.to_datetime(["2020-01-01"]), "latitude": [39.0, 39.1], "longitude": [2.0, 2.1]},
    )

    out = _normalize_rep(ds)

    assert set(out.coords) == {"time", "lat", "lon"}
    assert list(out.data_vars) == ["T"]
    # 300.15 K is 27 °C.
    np.testing.assert_allclose(out["T"].values, 27.0)
    assert out["T"].attrs["unit"] == "°C"


def test_normalize_medrea():
    # MEDREA timestamps sit at midday and are floored to midnight, to ease comparison with REP.
    ds = xr.Dataset(
        {"thetao": (("time", "latitude", "longitude"), np.full((2, 2, 2), 15.0))},
        coords={
            "time": pd.to_datetime(["2020-01-01T12:00:00", "2020-01-02T12:00:00"]),
            "latitude": [39.0, 39.1],
            "longitude": [2.0, 2.1],
        },
    )

    out = _normalize_medrea(ds)

    assert set(out.coords) == {"time", "lat", "lon"}
    assert list(out.data_vars) == ["T"]
    assert out["T"].attrs["unit"] == "°C"
    np.testing.assert_array_equal(out.time.values, pd.to_datetime(["2020-01-01", "2020-01-02"]).values)
