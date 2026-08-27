"""
Zarr-based dataset I/O for the balearic_mhws package.

Raw REP/MEDREA/bathymetry data lives in consolidated, append-friendly Zarr stores (populated by
`balearic_mhws.data.download`, already normalised to common variable/coordinate names and units).
This module only opens those stores and applies spatio-temporal subsetting, and reads/writes the
computed MHW datasets produced by `balearic_mhws.processing.compute_mhws`.
"""

import contextlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        use_compute_zarr: bool = False,
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

    use_compute_zarr : bool, default=False
        Opens `config.MEDREA_COMPUTE_ZARR` instead of the raw store. That store is rechunked for
        MHW computation (see `build_medrea_compute_store`) and only holds the default depths, so
        `depth_selector` must stay `'default'` or `None` when this is set.

    Returns
    ----------
    ds_medrea : xarray.Dataset
        The MEDREA dataset.
    """

    if depth_selector == 'default':
        depth_selector = config.MEDREA_DEFAULT_DEPTH_LEVELS

    ds_medrea = _open_zarr_store(config.MEDREA_COMPUTE_ZARR if use_compute_zarr else config.MEDREA_ZARR, "MEDREA")

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


def build_medrea_compute_store() -> None:
    """
    Rebuilds `config.MEDREA_COMPUTE_ZARR` from the raw MEDREA store.

    Computing MHWs needs a point's whole time serie at once, and Zarr can only read a chunk by
    decompressing it whole - the raw store's chunks span a whole year *and* every depth/lat/lon
    split, so getting any handful of points touches most of the store regardless of how few years
    or depths are asked for. Built one (time, lat, lon) region at a time - matching the raw
    store's own lat/lon chunk grid - so only a few of its chunks are ever decompressed at once,
    grouping `config.MEDREA_COMPUTE_YEARS_PER_CHUNK` years per chunk and keeping only
    `config.MEDREA_DEFAULT_DEPTH_LEVELS`.
    """

    existing = xr.open_zarr(config.MEDREA_ZARR, consolidated=True).drop_encoding()
    existing = existing.sel(depth=config.MEDREA_DEFAULT_DEPTH_LEVELS, method='nearest')

    years_per_chunk = config.MEDREA_COMPUTE_YEARS_PER_CHUNK
    target_time = years_per_chunk * config.ZARR_TIME_CHUNK
    time_native = existing.chunksizes['time']
    time_bounds = np.concatenate([[0], np.cumsum(time_native)])
    time_batches = [
        (int(time_bounds[i]), int(time_bounds[min(i + years_per_chunk, len(time_native))]))
        for i in range(0, len(time_native), years_per_chunk)
    ]

    # Lat/lon chunking reused as-is from the raw store - every region below has to line up with
    # the target's own chunk grid, and the raw store's split is already a reasonable size.
    lat_native = existing.chunksizes['lat']
    lon_native = existing.chunksizes['lon']
    lat_bounds = np.concatenate([[0], np.cumsum(lat_native)])
    lon_bounds = np.concatenate([[0], np.cumsum(lon_native)])

    tmp_path = config.MEDREA_COMPUTE_ZARR.parent / f"{config.MEDREA_COMPUTE_ZARR.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    # Skeleton only - shape, dtype and chunk grid, no data read (compute=False, never computed).
    # Every region below is written explicitly, so what its still-empty chunks start as doesn't
    # matter.
    skeleton = existing.chunk({'time': target_time, 'depth': -1, 'lat': tuple(lat_native), 'lon': tuple(lon_native)})
    skeleton.to_zarr(tmp_path, mode='w', compute=False, zarr_format=config.ZARR_FORMAT)

    n_regions = len(time_batches) * len(lat_native) * len(lon_native)
    done = 0
    for t0, t1 in time_batches:
        for li in range(len(lat_native)):
            for lj in range(len(lon_native)):
                region = {
                    'time': slice(t0, t1),
                    'lat': slice(int(lat_bounds[li]), int(lat_bounds[li + 1])),
                    'lon': slice(int(lon_bounds[lj]), int(lon_bounds[lj + 1])),
                }
                # Coordinates with no dimension in `region` (e.g. 'depth') are already fully
                # written by the skeleton and can't be part of a region write.
                # Every dim rechunked to one piece, not just the ones in `region` - 'depth' isn't
                # part of the region (it's fully kept, not sliced per-region) but still needs to
                # match the skeleton's single depth chunk, not the raw store's native sub-chunks.
                piece = existing.isel(**region).drop_vars([c for c in existing.coords if c not in region])
                piece = piece.chunk(-1)
                writer = piece.to_zarr(tmp_path, region=region, compute=False)
                writer.compute(scheduler='synchronous')
                done += 1
                print(f"Wrote region {done}/{n_regions} to {tmp_path}.")

    existing.close()
    xr.open_zarr(tmp_path).close()  # zarr's own consolidate needs a real read/write handle
    import zarr
    zarr.consolidate_metadata(str(tmp_path))

    if config.MEDREA_COMPUTE_ZARR.exists():
        shutil.rmtree(config.MEDREA_COMPUTE_ZARR)
    tmp_path.rename(config.MEDREA_COMPUTE_ZARR)

    print(f"Built {config.MEDREA_COMPUTE_ZARR}.")


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
        """Sets the print cadence, starting at `start_interval` and doubling up to `max_interval`."""

        super().__init__(dt=1.0)
        self._next_print = start_interval
        self._interval = start_interval
        self._max_interval = max_interval
        self._last_frac = None

    def _draw_bar(self, frac, elapsed):
        """Prints one progress line, immediately if progress was made, on the schedule otherwise."""

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
    """Builds the Zarr store path of a computed MHWs dataset from `config.MHWS_ZARR_PATTERN`."""

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
        # Rechunked onto the store's own layout first, as Zarr refuses a write whose chunks
        # straddle the chunks already on disk - a freshly downloaded month holds its depth in a
        # single chunk, where the store splits it.
        ds_new = ds_new.chunk({d: existing.chunksizes[d][0] for d in existing.dims if d != append_dim})
        existing.close()

        # Written one chunk at a time, as the appended values rarely fill the store's last chunk
        # along `append_dim` - Zarr then rereads and rewrites that whole chunk, once per position
        # along the other dimensions, which the default scheduler would do all at once.
        writer = ds_new.to_zarr(
            store_path, mode='a', append_dim=append_dim, consolidated=True, zarr_format=config.ZARR_FORMAT,
            compute=False,
        )
        writer.compute(scheduler='synchronous')

        print(f"Appended {ds_new[append_dim].size} {append_dim} value(s) to {store_path}.")
        return

    # New data falls before/within the existing range (e.g. backfilling older months after newer
    # ones were already ingested). Rebuilt one of `existing`'s own on-disk chunks at a time (not
    # the whole store in one pass): a Zarr chunk can only be read by decompressing it whole, so a
    # single MEDREA chunk (~111 depth levels) costs ~1.3GB regardless of how the *output* is
    # chunked - touching several at once is what was exhausting RAM here.
    existing_times = existing[append_dim].values
    chunk_sizes = existing.chunksizes[append_dim]
    bounds = np.concatenate([[0], np.cumsum(chunk_sizes)])

    ds_new = ds_new.sortby(append_dim)
    insert_pos = existing_times.searchsorted(ds_new[append_dim].values)
    # Which existing chunk each new value should be merged into - the one it falls inside, or the
    # nearest one if it falls in a gap between/outside existing chunks.
    window_of_new = np.clip(bounds[1:].searchsorted(insert_pos, side='left'), 0, len(chunk_sizes) - 1)

    # Non-append-dim chunk sizes computed once (not per window): a window mixing in new data and
    # a window of untouched `existing` data start from different chunk structures, so 'auto'
    # computed separately per window can pick different sizes - and Zarr rejects a window's write
    # if its chunk grid doesn't match what an earlier window already established.
    other_dims = [d for d in existing.dims if d != append_dim]
    reference = existing.chunk({append_dim: config.ZARR_TIME_CHUNK, **{d: 'auto' for d in other_dims}})
    fixed_chunks = {d: reference.chunksizes[d][0] for d in other_dims}

    tmp_path = store_path.parent / f"{store_path.name}.tmp"

    for window in range(len(chunk_sizes)):
        window_existing = existing.isel({append_dim: slice(bounds[window], bounds[window + 1])})
        new_mask = window_of_new == window

        if new_mask.any():
            window_ds = xr.concat([window_existing, ds_new.isel({append_dim: new_mask})], dim=append_dim)
            window_ds = window_ds.sortby(append_dim)
        else:
            window_ds = window_existing

        window_ds = window_ds.chunk({**fixed_chunks, append_dim: config.ZARR_TIME_CHUNK})

        # Writing to a sibling path and swapping it in at the end (not onto store_path directly)
        # keeps a crash mid-rewrite from ever leaving store_path half-migrated or corrupted.
        if window == 0:
            writer = window_ds.to_zarr(tmp_path, mode='w', consolidated=True, zarr_format=config.ZARR_FORMAT, compute=False)
        else:
            writer = window_ds.to_zarr(
                tmp_path, mode='a', append_dim=append_dim, consolidated=True, zarr_format=config.ZARR_FORMAT, compute=False,
            )
        writer.compute(scheduler='synchronous')

    existing.close()
    shutil.rmtree(store_path)
    tmp_path.rename(store_path)

    print(f"Merged {ds_new[append_dim].size} {append_dim} value(s) into {store_path} (rewrote the full store).")
