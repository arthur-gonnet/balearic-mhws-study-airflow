"""
Downloads REP/MEDREA/bathymetry data from the Copernicus Marine Service and ingests it into the
project's raw Zarr stores (see `balearic_mhws.config` for their paths).

Ingestion is idempotent: each call only downloads the (year, month) pairs missing from the target
Zarr store, so re-running `ingest_rep`/`ingest_medrea` for an overlapping range is a no-op for
already-covered months.

Credentials are read from the environment (`config.COPERNICUS_USERNAME`/`COPERNICUS_PASSWORD`) -
no interactive prompts, so this can run unattended from Airflow or a Slurm job. Create an account
at https://data.marine.copernicus.eu/ if you don't have one.
"""

import tempfile
from calendar import monthrange
from pathlib import Path
from typing import Dict, Iterable, List

import copernicusmarine
import pandas as pd
import xarray as xr

from .. import config
from .io import write_zarr_incremental

########################################################################################################################
##################################### HELPERS ###########################################################################
########################################################################################################################


def _missing_year_months(store_path: Path, years: Iterable[int], months: Iterable[int]) -> Dict[int, List[int]]:
    """Returns {year: [missing months]} for the (year, month) pairs not yet in the store's time coordinate."""

    requested = {(int(year), int(month)) for year in years for month in months}

    covered = set()
    if store_path.exists():
        existing = xr.open_zarr(store_path, consolidated=True)
        idx = pd.DatetimeIndex(existing.time.values)
        covered = set(zip(idx.year.tolist(), idx.month.tolist()))
        existing.close()

    missing: Dict[int, List[int]] = {}
    for year, month in sorted(requested - covered):
        missing.setdefault(year, []).append(month)

    return missing


def _download_month_netcdf(dataset_id: str, variables: List[str], year: int, month: int, tmp_dir: str) -> Path:
    """Downloads one (year, month) subset of a Copernicus Marine dataset to a temporary NetCDF file."""

    if not config.COPERNICUS_USERNAME or not config.COPERNICUS_PASSWORD:
        raise RuntimeError(
            "Copernicus Marine credentials not configured. Set USERNAME_COPERNICUS/PASSWORD_COPERNICUS "
            "in .env (sign up for free at https://data.marine.copernicus.eu/register)."
        )

    path = Path(tmp_dir) / f"{dataset_id}_{year}_{month:02d}.nc"

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        output_filename=str(path),

        start_datetime=f"{year}-{month:02d}-01T00:00:00",
        end_datetime=f"{year}-{month:02d}-{monthrange(year, month)[1]}T23:00:00",

        username=config.COPERNICUS_USERNAME,
        password=config.COPERNICUS_PASSWORD,
        **config.SPATIAL_EXTENT,
    )

    return path


def _normalize_rep(ds: xr.Dataset) -> xr.Dataset:
    """Uniformises coordinate/variable names and units to match the rest of the pipeline."""

    ds = ds.drop_vars([v for v in ('analysis_error', 'mask', 'sea_ice_fraction') if v in ds.variables])

    if 'longitude' in ds.coords:
        ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat'})

    ds = ds.rename({'analysed_sst': 'T'})
    ds['T'] = ds['T'] - 273.15
    ds['T'].attrs['unit'] = '°C'

    return ds


def _normalize_medrea(ds: xr.Dataset) -> xr.Dataset:
    """Uniformises coordinate/variable names and units to match the rest of the pipeline."""

    if 'longitude' in ds.coords:
        ds = ds.rename({'longitude': 'lon'})
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat'})

    ds = ds.rename({'thetao': 'T'})
    ds['T'].attrs['unit'] = '°C'

    # Move time points to 0am instead of 12am, easing comparison with REP
    ds['time'] = ds.time.dt.floor('1D')

    return ds


########################################################################################################################
##################################### INGEST ############################################################################
########################################################################################################################


def ingest_rep(years: Iterable[int] = range(1982, 2024), months: Iterable[int] = range(1, 13)) -> None:
    """
    Downloads missing REP SST data and appends it to the REP Zarr store (`config.REP_ZARR`).

    Parameters
    ----------
    years : Iterable[int], default=range(1982, 2024)
        Years to ensure are present in the Zarr store.

    months : Iterable[int], default=range(1, 13)
        Months to ensure are present in the Zarr store, for each requested year.
    """

    missing = _missing_year_months(config.REP_ZARR, years, months)

    if not missing:
        print("REP Zarr store already up to date.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        for year, year_months in sorted(missing.items()):
            print(f"Downloading REP {year} ({len(year_months)} month(s): {year_months})...")

            monthly_paths = [
                _download_month_netcdf(config.REP_DATASET_ID, ["analysed_sst"], year, month, tmp_dir)
                for month in year_months
            ]

            ds = xr.open_mfdataset(monthly_paths)
            ds = _normalize_rep(ds)
            ds = ds.chunk({"time": -1, "lat": config.ZARR_SPATIAL_CHUNK, "lon": config.ZARR_SPATIAL_CHUNK})

            write_zarr_incremental(ds, config.REP_ZARR)
            ds.close()

            for path in monthly_paths:
                path.unlink()


def ingest_medrea(years: Iterable[int] = range(1987, 2023), months: Iterable[int] = range(1, 13)) -> None:
    """
    Downloads missing MEDREA temperature data and appends it to the MEDREA Zarr store (`config.MEDREA_ZARR`).

    Parameters
    ----------
    years : Iterable[int], default=range(1987, 2023)
        Years to ensure are present in the Zarr store.

    months : Iterable[int], default=range(1, 13)
        Months to ensure are present in the Zarr store, for each requested year.
    """

    missing = _missing_year_months(config.MEDREA_ZARR, years, months)

    if not missing:
        print("MEDREA Zarr store already up to date.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        for year, year_months in sorted(missing.items()):
            print(f"Downloading MEDREA {year} ({len(year_months)} month(s): {year_months})...")

            monthly_paths = [
                _download_month_netcdf(config.MEDREA_DATASET_ID, ["thetao"], year, month, tmp_dir)
                for month in year_months
            ]

            ds = xr.open_mfdataset(monthly_paths)
            ds = _normalize_medrea(ds)
            ds = ds.chunk({
                "time": -1,
                "lat": config.ZARR_SPATIAL_CHUNK,
                "lon": config.ZARR_SPATIAL_CHUNK,
                "depth": -1,
            })

            write_zarr_incremental(ds, config.MEDREA_ZARR)
            ds.close()

            for path in monthly_paths:
                path.unlink()


def ingest_bathy() -> None:
    """Downloads the MEDREA bathymetry dataset and writes it to `config.BATHY_ZARR` (one-shot, not incremental)."""

    if config.BATHY_ZARR.exists():
        print("Bathymetry Zarr store already exists, skipping.")
        return

    if not config.COPERNICUS_USERNAME or not config.COPERNICUS_PASSWORD:
        raise RuntimeError(
            "Copernicus Marine credentials not configured. Set USERNAME_COPERNICUS/PASSWORD_COPERNICUS "
            "in .env (sign up for free at https://data.marine.copernicus.eu/register)."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "medrea_bathy.nc"

        copernicusmarine.subset(
            dataset_id=config.MEDREA_BATHY_DATASET_ID,
            variables=["deptho"],
            output_filename=str(path),
            username=config.COPERNICUS_USERNAME,
            password=config.COPERNICUS_PASSWORD,
            **config.SPATIAL_EXTENT,
        )

        ds = xr.open_dataset(path)

        if 'longitude' in ds.coords:
            ds = ds.rename({'longitude': 'lon'})
        if 'latitude' in ds.coords:
            ds = ds.rename({'latitude': 'lat'})

        ds = ds.rename({'deptho': 'depth'})

        config.BATHY_ZARR.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(config.BATHY_ZARR, mode='w', consolidated=True, zarr_format=config.ZARR_FORMAT)
        ds.close()

    print(f"Saved bathymetry to {config.BATHY_ZARR}.")
