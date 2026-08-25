"""
Zarr-based dataset I/O for the balearic_mhws package.

Raw REP/MEDREA/bathymetry data lives in consolidated, append-friendly Zarr stores (populated by
`balearic_mhws.data.download`, already normalised to common variable/coordinate names and units).
This module only opens those stores and applies spatio-temporal subsetting, and reads/writes the
computed MHW datasets produced by `balearic_mhws.processing.compute_mhws`.
"""

import contextlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import xarray as xr
from dask.diagnostics.progress import ProgressBar
from dask.utils import format_time

from .. import config

########################################################################################################################
##################################### RAW DATASETS ######################################################################
########################################################################################################################


def _region_selector(region_selector: Optional[str]) -> Tuple[Optional[slice], Optional[slice]]:
    """Resolves a named region into (lon_selector, lat_selector) slices."""

    if region_selector == 'balears':
        return slice(-0.9, 5.1), slice(37.6, 41.1)

    if region_selector is None:
        return None, None

    raise ValueError(f"Unknown region_selector {region_selector!r}, only 'balears' is currently defined.")


def _open_zarr_store(path: Path, label: str, chunks: str | None = 'auto') -> xr.Dataset:
    """
    Opens a Zarr store, dropping its on-disk encoding (chunking, compressors, zarr-format-3-only
    keys like 'serializer', ...) right away - it otherwise rides along through any downstream
    computation and gets rejected the moment that computation is saved with zarr_format=2 (see
    config.ZARR_FORMAT). We never round-trip data unchanged, so this encoding is never useful
    downstream anyway.
    """

    if not path.exists():
        raise FileNotFoundError(f"{label} Zarr store not found at {path}. Run the download stage first.")

    return xr.open_zarr(path, chunks=chunks, consolidated=True).drop_encoding()


def open_rep(
        time_selector: Optional[str | slice] = None,
        lon_selector: Optional[float | slice] = None,
        lat_selector: Optional[float | slice] = None,
        region_selector: Optional[str] = 'balears',
) -> xr.Dataset:
    """
    Opens the REP Zarr store, with optional spatio-temporal subsetting.

    Parameters
    ----------
    time_selector : str | slice[str], optional
        Time selection applied using xarray's `.sel()`.

    lon_selector, lat_selector : float | slice[float], optional
        Longitude/latitude selectors applied using xarray's `.sel()`. Overridden by `region_selector`.

    region_selector : str, default='balears', optional
        Applies a named spatial selector to the dataset. Overrides `lon_selector`/`lat_selector`.

    Returns
    ----------
    ds_rep : xarray.Dataset
        The REP dataset.
    """

    ds_rep = _open_zarr_store(config.REP_ZARR, "REP")

    if region_selector is not None:
        lon_selector, lat_selector = _region_selector(region_selector)

    if lon_selector is not None:
        ds_rep = ds_rep.sel(lon=lon_selector, method=(None if isinstance(lon_selector, slice) else 'nearest'))

    if lat_selector is not None:
        ds_rep = ds_rep.sel(lat=lat_selector, method=(None if isinstance(lat_selector, slice) else 'nearest'))

    if time_selector is not None:
        ds_rep = ds_rep.sel(time=time_selector)

    return ds_rep


def open_medrea(
        time_selector: Optional[str | slice | List[str]] = None,
        lon_selector: Optional[float | slice | List[float]] = None,
        lat_selector: Optional[float | slice | List[float]] = None,
        depth_selector: Optional[float | slice | List[float]] = 'default',
        region_selector: Optional[str] = 'balears',
) -> xr.Dataset:
    """
    Opens the MEDREA Zarr store, with optional spatio-temporal subsetting.

    Parameters
    ----------
    time_selector : str | slice[str], optional
        Time selection applied using xarray's `.sel()`.

    lon_selector, lat_selector : float | slice[float], optional
        Longitude/latitude selectors applied using xarray's `.sel()`. Overridden by `region_selector`.

    depth_selector : float | slice[float] | list[float], default='default', optional
        Depth selector applied using xarray's `.sel()`. Defaults to `config.MEDREA_DEFAULT_DEPTH_LEVELS`,
        a handful of representative depths - the full water column has ~111 levels, computing MHWs at
        every one of them is ~100x more expensive for little added insight. Pass `None` for no depth
        filtering (the full water column) or a `slice`/explicit list for something else.

    region_selector : str, default='balears', optional
        Applies a named spatial selector to the dataset. Overrides `lon_selector`/`lat_selector`.

    Returns
    ----------
    ds_medrea : xarray.Dataset
        The MEDREA dataset.
    """

    if depth_selector == 'default':
        depth_selector = config.MEDREA_DEFAULT_DEPTH_LEVELS

    ds_medrea = _open_zarr_store(config.MEDREA_ZARR, "MEDREA")

    if region_selector is not None:
        lon_selector, lat_selector = _region_selector(region_selector)

    if lon_selector is not None:
        ds_medrea = ds_medrea.sel(lon=lon_selector, method=(None if isinstance(lon_selector, slice) else 'nearest'))

    if lat_selector is not None:
        ds_medrea = ds_medrea.sel(lat=lat_selector, method=(None if isinstance(lat_selector, slice) else 'nearest'))

    if depth_selector is not None and 'depth' in ds_medrea.dims:
        ds_medrea = ds_medrea.sel(depth=depth_selector, method=(None if isinstance(depth_selector, slice) else 'nearest'))

    if time_selector is not None:
        ds_medrea = ds_medrea.sel(time=time_selector)

    return ds_medrea


def open_bathy(region_selector: Optional[str] = 'balears') -> xr.Dataset:
    """
    Opens the MEDREA bathymetry Zarr store, with optional spatial subsetting.

    Parameters
    ----------
    region_selector : str, default='balears', optional
        Applies a named spatial selector to the dataset.

    Returns
    ----------
    ds_bathy : xarray.Dataset
        The bathymetry dataset.
    """

    # chunks=None: tiny static field, and staying dask-chunked here leaks into apply_regional_mask.
    ds_bathy = _open_zarr_store(config.BATHY_ZARR, "Bathymetry", chunks=None)

    if region_selector is not None:
        lon_selector, lat_selector = _region_selector(region_selector)
        ds_bathy = ds_bathy.sel(lon=lon_selector, lat=lat_selector)

    return ds_bathy


########################################################################################################################
##################################### MHWS DATASETS #####################################################################
########################################################################################################################


class _LoggingProgressBar(ProgressBar):
    """
    dask ProgressBar variant that prints one log line per update instead of redrawing a bar in
    place with '\\r' - '\\r'-based redraws don't render in Airflow's captured/structured logs, they
    just silently vanish, leaving a long-running compute task with no visible progress at all.

    Print cadence backs off exponentially (starting at `start_interval`, doubling up to
    `max_interval`) so a multi-hour computation doesn't spam the log with hundreds of near-identical
    lines, while still confirming quickly that the task is alive right after it starts. Actual
    progress - e.g. a chunk finishing, including the final 100% - always prints immediately
    regardless of that schedule, since it's real information, not a heartbeat.
    """

    def __init__(self, start_interval: float = 15.0, max_interval: float = 600.0):
        super().__init__(dt=1.0)
        self._next_print = start_interval
        self._interval = start_interval
        self._max_interval = max_interval
        self._last_frac = None

    def _draw_bar(self, frac, elapsed):
        if frac == self._last_frac and elapsed < self._next_print:
            return

        print(f"Computing: {int(100 * frac)}% complete - {format_time(elapsed)} elapsed")

        self._last_frac = frac
        self._interval = min(self._interval * 2, self._max_interval)
        self._next_print = elapsed + self._interval


def _mhws_zarr_path(
        ds_type: str,
        dataset_used: str,
        detrended: bool,
        region: str,
        clim_period: Tuple[int, int],
) -> Path:
    return Path(config.MHWS_ZARR_PATTERN.format(
        type=ds_type,
        dataset=dataset_used,
        detrended='_detrended' if detrended else '',
        region=region,
        clim_start=clim_period[0],
        clim_end=clim_period[1],
    ))


def save_mhws(
        ds_mhws: xr.Dataset,
        ds_type: str,
        dataset_used: str,
        detrended: bool = False,
        region: str = 'balears',
        clim_period: Tuple[int, int] = config.CLIMATOLOGY_PERIOD,
        progress_bar: bool = True,
) -> xr.Dataset:
    """
    Saves a MHWs dataset as a Zarr store.

    Parameters
    ----------
    ds_mhws : xarray.Dataset
        MHWs dataset to save.

    ds_type : str
        Can be 'yearly' or 'all_events'.

    dataset_used : str
        Describes the dataset from which the MHWs computations were performed.
        Can be `'rep'`, `'medrea_bot'` or `'medrea_50m'` for example.

    region : str, default='balears'
        Region in which the MHWs computations were performed.

    clim_period : tuple[int, int], default=config.CLIMATOLOGY_PERIOD
        Climatology period used for the MHWs computations.

    progress_bar : bool, default=True
        If `True`, shows a progress bar while writing the dataset.

    Returns
    ----------
    ds_mhws : xarray.Dataset
        The computed MHWs dataset.
    """

    zarr_path = _mhws_zarr_path(ds_type, dataset_used, detrended, region, clim_period)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    # compute_mhw_yearly/compute_mhw_all_events stack and unstack dimensions internally
    # (e.g. lat/lon/depth for MEDREA), which can leave dask chunks uneven along the unstacked
    # dimensions - Zarr rejects a chunk that's larger than the one before it. MHW output datasets
    # are always small (stats x grid x years, not raw input volumes), so rechunking to one chunk
    # per dimension before writing is cheap and guarantees valid, uniform Zarr chunk encoding.
    ds_mhws = ds_mhws.chunk({dim: -1 for dim in ds_mhws.dims})

    print(f"Saving MHWs dataset to {zarr_path}")

    with _LoggingProgressBar() if progress_bar else contextlib.nullcontext():
        ds_mhws.to_zarr(zarr_path, mode='w', consolidated=True, zarr_format=config.ZARR_FORMAT)

    print(" -> Saved!")

    return ds_mhws


def load_mhws(
        ds_type: str,
        dataset_used: str,
        detrended: bool = False,
        region: str = 'balears',
        clim_period: Tuple[int, int] = config.CLIMATOLOGY_PERIOD,
) -> xr.Dataset:
    """
    Loads a MHWs dataset from its Zarr store.

    Parameters
    ----------
    ds_type : str
        Can be 'yearly' or 'all_events'.

    dataset_used : str
        Describes the dataset from which the MHWs computations were performed.

    region : str, default='balears'
        Region in which the MHWs computations were performed.

    clim_period : tuple[int, int], default=config.CLIMATOLOGY_PERIOD
        Climatology period used for the MHWs computations.

    Returns
    ----------
    ds_mhws : xarray.Dataset
        The loaded MHWs dataset.
    """

    zarr_path = _mhws_zarr_path(ds_type, dataset_used, detrended, region, clim_period)

    if not zarr_path.exists():
        raise FileNotFoundError(f"MHWs Zarr store not found at {zarr_path}.")

    # chunks=None: small already-computed results; dask-backed arrays break plain scalar formatting.
    ds_mhws = xr.open_zarr(zarr_path, chunks=None, consolidated=True)

    print("Loaded MHWs dataset.")

    return ds_mhws


########################################################################################################################
##################################### INGEST HELPER #####################################################################
########################################################################################################################


def write_zarr_incremental(ds: xr.Dataset, store_path: Path, append_dim: str = 'time') -> None:
    """
    Writes a dataset to a Zarr store, creating it if missing or merging in only the new
    `append_dim` values otherwise - making repeated ingestion runs idempotent.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write. Must be chunked (or chunkable) along `append_dim`.

    store_path : Path
        Destination Zarr store.

    append_dim : str, default='time'
        Dimension along which new data is merged in.
    """

    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if not store_path.exists():
        ds.to_zarr(store_path, mode='w', consolidated=True, zarr_format=config.ZARR_FORMAT)
        print(f"Created new Zarr store at {store_path}.")
        return

    existing = xr.open_zarr(store_path, consolidated=True).drop_encoding()

    ds_new = ds.sel({append_dim: ~ds[append_dim].isin(existing[append_dim])})

    if ds_new[append_dim].size == 0:
        existing.close()
        print(f"No new {append_dim} values to write to {store_path}.")
        return

    if bool((ds_new[append_dim] > existing[append_dim].max()).all()):
        # Fast path: purely new data past the end - append without touching what's already there.
        existing.close()
        ds_new.to_zarr(store_path, mode='a', append_dim=append_dim, consolidated=True, zarr_format=config.ZARR_FORMAT)
        print(f"Appended {ds_new[append_dim].size} {append_dim} value(s) to {store_path}.")
        return

    # New data falls before/within the existing range (e.g. backfilling older months after newer
    # ones were already ingested) - to_zarr's append mode can only extend the end, so the only way
    # to keep the store's `append_dim` sorted is to merge in-memory and rewrite the whole store.
    combined = xr.concat([existing, ds_new], dim=append_dim).sortby(append_dim)
    existing.close()
    combined = combined.chunk({append_dim: config.ZARR_TIME_CHUNK})
    combined.to_zarr(store_path, mode='w', consolidated=True, zarr_format=config.ZARR_FORMAT)
    print(f"Merged {ds_new[append_dim].size} {append_dim} value(s) into {store_path} (rewrote the full store).")
