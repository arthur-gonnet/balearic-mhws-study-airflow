"""
Tests for balearic_mhws.data.download's pure logic - no network/credentials needed.

The actual `copernicusmarine.subset()` calls are not tested here; only the idempotency logic
that decides which (year, month) pairs still need downloading.
"""

import numpy as np
import pandas as pd
import xarray as xr

from balearic_mhws.data.download import _missing_year_months


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
