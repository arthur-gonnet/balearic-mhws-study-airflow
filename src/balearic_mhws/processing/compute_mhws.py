"""
This script gathers all the code used to compute marine heatwaves metrics from a temperature datasets.


Functions description
----------
    - compute_mhw_yearly(...):
        Computes marine heatwaves annual metrics from a temperature dataset.

    - compute_mhw_yearly_wrapped(...):
        Wrapper for 'compute_mhw_yearly'

    - compute_mhw_all_events(...):
        Computes marine heatwaves event metrics from a temperature dataset.

    - compute_mhw_all_events_wrapped(...):
        Wrapper for 'compute_mhw_all_events'

    - get_mhw_ts_from_ds(...):
        Format the statistics produced by 'compute_mhw_all_events' in a more friendly way.


Examples
----------
    from balearic_mhws.data.io import open_rep, open_medrea, save_mhws
    from balearic_mhws.processing.compute_mhws import compute_mhw_yearly

    # For REP

    ds_rep = open_rep()

    ds_mhws = compute_mhw_yearly(
        ds_rep,
        using_dataset = 'rep',
        clim_period = clim_period,
    )

    ds_mhws = save_mhws(
        ds_mhws,
        ds_type = 'yearly',
        dataset_used = 'rep',
        region = 'balears',
        clim_period = clim_period,
    )

    # For MEDREA

    ds_medrea = open_medrea(
        region_selector = 'balears',
        depth_selector = [0, 50, 100, 150, 200, 500, 700, 1000, 1500, 2000],
    )

    ds_mhws = compute_mhw_yearly(
        ds_medrea,
        using_dataset = 'medrea',
        clim_period = clim_period,
    )

    save_mhws(
        ds_mhws,
        ds_type = 'yearly',
        dataset_used = 'medrea',
        clim_period = clim_period,
        region = 'balears',
    )
"""

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Basic imports
import math
from datetime import date
from typing import Optional, List, Dict, Tuple

# Advanced imports
import numpy as np
from numpy.typing import NDArray
import xarray as xr
from dask.diagnostics.progress import ProgressBar

# Local imports
from . import marineHeatWaves as mhw
from .. import config as opts

########################################################################################################################
##################################### CODE #############################################################################
########################################################################################################################


def compute_mhw_yearly(
        # Dataset options
        ds: xr.Dataset | xr.DataArray,
        using_dataset: Optional[str] = None,
        var_name: str = 'T',

        # Computation options
        clim_period: tuple[int, int] = (1987,2021),
        detrend: bool = False,
) -> xr.Dataset:
    """
    Computes marine heatwaves annual metrics from a temperature dataset.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dataset or DataArray of temperature. Time dimension must be named `'time'`.
        This function should work with any number of additional dimensions for the dataset.
        Only tested using daily temperature.

    using_dataset: str, optional
        Only used to add acknowledgments to the dataset attributes. Can be 'rep' or 'medrea'.

    var_name: str, default='T', optional
        Optional parameter to specify the variable name used. Only used if providing
        a dataset, so that the correct DataArray is extracted from it.

    clim_period: tuple[int, int], default=(1987,2021)
        The climatology period used. Must be in the dataset time definition range.

    detrend: bool, default=False
        Option to detrend the temperature time serie before computing MHWs.

    Returns
    ----------
    ds_mhws: xr.Dataset
        A dataset containing all the 26 annual metrics of `opts.mhws_stats`.
        Includes every dimension of the original dataset, except `'time'` that became `'year'`
        along the process.

    Notes
    ----------
        `compute_mhw_yearly()` uses Dask's lazy evaluation. This enables not to load the
        entire dataset on the disk at once, but rather compute the dataset chunk-wise.
        Also, the computation is triggered only when saving the dataset using `save_mhws()`
        or using `ds_mhws.compute()`. More information in Dask's documentation.
    """

    # Compatibility snippet so that the function can work with both DataArray and Dataset
    if isinstance(ds, xr.Dataset):
        ds = ds[var_name]

    # Investigate the stacking possibilities of the dataarray
    stackable_dims = [
        dim for dim in ds.dims
        if not dim == 'time' and ds[dim].size > 1
    ]
    stack = len(stackable_dims) > 1

    # Stacking possible dimensions for perfomance sake
    if stack:
        print("Stacking dimensions: ", stackable_dims)
        ds = ds.stack(pos=stackable_dims)

    # Years of the dataset
    years = np.unique(ds['time.year'].values)

    # Chunk every dimension of the dataset but 'time'
    ds = ds.chunk({dim: -1 if dim == 'time' else 'auto' for dim in ds.dims})

    # Computing the mhws
    #   results contain the resulting mhws statistics
    #   shape is (mhws_stats=26, pos, years)
    results = xr.apply_ufunc(
        # Inputs
        compute_mhw_yearly_wrapped,
        ds.time,
        ds,
        kwargs={
            'clim_period': clim_period,
            'detrend': detrend
        },

        # Dimensions of input and output
        input_core_dims     = [['time'], ['time']],
        output_core_dims    = [(['year'])] * len(opts.mhws_stats),

        # Type and size of output
        output_dtypes       = [float] * len(opts.mhws_stats),
        dask_gufunc_kwargs  = dict(
            output_sizes = {
                'year': len(years)
            }
        ),

        # Dask Options
        vectorize   = True,
        dask        = 'parallelized',
    )

    # Merging the result in a xarray Dataset, assigning coordinates back again
    ds_mhws = xr.Dataset({
        var: data.assign_coords(year=years)
        for var, data in zip(opts.mhws_stats, results)
    })

    # Unstacking 'pos' dimension
    if stack:
        ds_mhws = ds_mhws.unstack('pos')

    # Ordering 'lon' and 'lat' if needed, so that they have the right order for plotting maps
    if 'lon' in stackable_dims and 'lat' in stackable_dims:
        ds_mhws = ds_mhws.transpose('lat', 'lon', 'year', ...)

    # Adding variables metadata
    for stat in opts.mhws_stats:
        ds_mhws[stat].attrs['shortname']  = opts.mhws_stats_shortname[stat]
        ds_mhws[stat].attrs['longname']   = opts.mhws_stats_longname[stat]
        ds_mhws[stat].attrs['unit']       = opts.mhws_stats_units[stat]

    # Adding dataset attributes
    ds_mhws.attrs['climatologyPeriod'] = f'{clim_period[0]}-{clim_period[1]}'
    ds_mhws.attrs['description'] = opts.mhw_yearly_dataset_description

    if using_dataset and using_dataset.lower() == 'rep':
        ds_mhws.attrs['acknowledgment'] = opts.rep_acknowledgment

    elif using_dataset and using_dataset.lower() == 'medrea':
        ds_mhws.attrs['acknowledgment'] = opts.medrea_acknowledgment

    # Returning the final Dataset!
    return ds_mhws



def compute_mhw_yearly_wrapped(t: NDArray, sst: NDArray, clim_period: Tuple[int, int], detrend: bool) -> Tuple[NDArray]:
    """
    Wrapper for compute_mhw_yearly. Computes annual MHW statistics for a given temperature time series.

    Parameters
    ----------
    t: np.array
        Time array of the time series.

    sst: np.array
        Temperature array of the time series.

    clim_period: tuple[int, int]
        The climatology period used. Must be in the dataset time definition range.

    detrend: bool
        Option to detrend the temperature time serie before computing MHWs.

    Returns
    ----------
    mhw_stats: tuple[np.array]
        A tuple containing all the 26 annual metrics of `opts.mhws_stats`.
    """

    # Ignore all-nan timeseries for performance sake
    if np.isnan(sst).all():
        nans = np.array([
            np.nan
            for _ in range(t[0].astype('datetime64[Y]').astype(int) + 1970, t[-1].astype('datetime64[Y]').astype(int) + 1970+1)
        ])

        return tuple(nans for _ in opts.mhws_stats)

    # Array manipulation to fit mhw module requirements
    t = t.astype('datetime64[D]').astype(int) + 719163 # to ordinal time
    temp = sst.copy()

    # Computing MHWs using mhw module
    mhws, clim = mhw.detect(t, temp, climatologyPeriod=clim_period, cutMhwEventsByYear=True)
    mhwBlock = mhw.blockAverage(t, mhws, clim, temp=temp)

    # Return annual metrics
    return tuple(mhwBlock[stat] for stat in opts.mhws_stats)



# Every stats availables
mhws_all_events_stats = [
    'time_start', 'time_end', 'time_peak',
    'index_start', 'index_end', 'index_peak',
    'duration', 'duration_moderate', 'duration_strong', 'duration_severe', 'duration_extreme',
    'intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative', 'intensity_max_relThresh',
    'intensity_mean_relThresh', 'intensity_var_relThresh', 'intensity_cumulative_relThresh', 'intensity_max_abs',
    'intensity_mean_abs', 'intensity_var_abs', 'intensity_cumulative_abs', 'category', 'rate_onset', 'rate_decline',
]

# Stats to compute timeseries of
mhws_all_events_useful_stats = [
    'duration', 'duration_moderate', 'duration_strong', 'duration_severe', 'duration_extreme',
    'intensity_max', 'intensity_mean', 'intensity_var', 'intensity_cumulative', 'intensity_max_abs',
    'intensity_mean_abs', 'intensity_var_abs', 'intensity_cumulative_abs', 'category', 'rate_onset', 'rate_decline',
]

clim_keys = ['thresh', 'seas', 'missing', 'sst']




def compute_mhw_all_events(
        # Dataset options
        ds: xr.Dataset | xr.DataArray,
        using_dataset: Optional[str] = None,
        var_name: str = 'T',

        # Computation options
        clim_period: Tuple[int, int] = (1987,2021),
        detrend: bool = False,
) -> xr.Dataset:
    """
    Computes marine heatwaves event metrics from a temperature dataset.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dataset or DataArray of temperature. Time dimension must be named `'time'`.
        This function should work with any number of additional dimensions for the dataset.
        Only tested using daily temperature.

    using_dataset: str, optional
        Only used to add acknowledgments to the dataset attributes. Can be 'rep' or 'medrea'.

    var_name: str, default='T', optional
        Optional parameter to specify the variable name used. Only used if providing
        a dataset, so that the correct DataArray is extracted from it.

    clim_period: tuple[int, int], default=(1987,2021)
        The climatology period used. Must be in the dataset time definition range.

    detrend: bool, default=False
        Option to detrend the temperature time serie before computing MHWs.

    Returns
    ----------
    ds_mhws: xr.Dataset
        A dataset containing all the 26 event metrics of `mhws_all_events_stats`, as well as
        the 4 daily metrics of `clim_keys`.
        Includes every dimension of the original dataset. A new dimension has been added:
        `'event_number'`, meant for attributing the event metrics to a specific event.
    """

    # Compatibility snippet so that the function can work with both DataArray and Dataset
    if isinstance(ds, xr.Dataset):
        ds = ds[var_name]

    # Investigate the stacking possibilities of the dataarray
    stackable_dims = [
        dim for dim in ds.dims
        if not dim == 'time' and ds[dim].size > 1
    ]
    stack = len(stackable_dims) > 1

    # Stacking possible dimensions for perfomance sake
    if stack:
        print("Stacking dimensions: ", stackable_dims)
        ds = ds.stack(pos=stackable_dims)

    # Chunk every dimension of the dataset but 'time'
    ds = ds.chunk({dim: -1 if dim == 'time' else 'auto' for dim in ds.dims})

    # As using parallelization, dimension sizes must be fixed. This is the maximum
    # theoretical value possible for event_number
    # Maximum is when 5d MHWs are separated by 3d gaps
    max_number_of_event = math.ceil(ds.time.size/8)

    # Computing the mhws
    #   result contains the resulting mhws statistics
    #   shape is (things=33, pos, event_number/time)
    results = xr.apply_ufunc(
        # Inputs
        compute_mhw_all_events_wrapped,
        ds.time,
        ds,
        kwargs={
            'clim_period': clim_period,
            'detrend': detrend
        },

        # Dimensions of input and output
        input_core_dims     = [['time'], ['time']],
        output_core_dims    = [['event_number']] * len(mhws_all_events_stats) + [['time']] * len(clim_keys),

        # Type and size of output
        output_dtypes       = [float] * (len(mhws_all_events_stats) + len(clim_keys)),
        dask_gufunc_kwargs  = dict(
            output_sizes = {
                'event_number': max_number_of_event,
                'time': ds.time.size,
            }
        ),

        # Dask Options
        vectorize   = True,
        dask        = 'parallelized',
    )

    # Merging the result in a xarray Dataset, assigning coordinates back again
    ds_mhws = xr.Dataset({
        var: data.assign_coords(event_number=range(max_number_of_event))
        for var, data in zip(mhws_all_events_stats, results[:len(mhws_all_events_stats)])
    } | {
        'clim_' + var: data.assign_coords(time=ds.time)
        for var, data in zip(clim_keys, results[len(mhws_all_events_stats):])
    })

    # Unstacking 'pos' dimension
    if stack:
        ds_mhws = ds_mhws.unstack('pos')


    # To reduce dataset size, event numbers are limited to the maximum value reached in the dataset.
    # To get this value, the computation must be performed

    with ProgressBar():
        print("Computing MHWs dataset")
        ds_mhws = ds_mhws.compute()

    # Finding the first nan value that is for every grid point
    dims_to_check = [dim for dim in ds_mhws.dims if dim not in ('time', 'event_number')]

    if len(dims_to_check) > 0:
        nan_mask = ds_mhws.time_start.isnull().all(dim=dims_to_check)
    else:
        nan_mask = ds_mhws.time_start.isnull()

    # Reduce dataset by removing the nans
    if nan_mask.any():
        first_nan = nan_mask.argmax().item()
        if first_nan != 0:
            ds_mhws = ds_mhws.sel(event_number=slice(0, first_nan))

    else:
        first_nan = None


    # Ordering 'lon' and 'lat' if needed, so that they have the right order for plotting maps
    if 'lon' in stackable_dims and 'lat' in stackable_dims:
        ds_mhws = ds_mhws.transpose('lat', 'lon', 'year', ...)

    # Adding attributes
    ds_mhws.attrs['climatologyPeriod'] = f'{clim_period[0]}-{clim_period[1]}'
    ds_mhws.attrs['description'] = opts.mhw_dataset_description

    if using_dataset and using_dataset.lower() == 'rep':
        ds_mhws.attrs['acknowledgment'] = opts.rep_acknowledgment

    elif using_dataset and using_dataset.lower() == 'medrea':
        ds_mhws.attrs['acknowledgment'] = opts.medrea_acknowledgment

    # Returning the final Dataset!
    return ds_mhws



def compute_mhw_all_events_wrapped(t: NDArray, sst: NDArray, clim_period: tuple[int, int], detrend: bool):
    """
    Wrapper for compute_mhw_all_events. Computes MHW statistics for a given temperature time series.

    Parameters
    ----------
    t: np.array
        Time array of the time series.

    sst: np.array
        Temperature array of the time series.

    clim_period: tuple[int, int]
        The climatology period used. Must be in the dataset time definition range.

    detrend: bool
        Option to detrend the temperature time serie before computing MHWs.

    Returns
    ----------
    mhw_stats: tuple[np.array]
        A tuple containing all the 26 event metrics of `mhws_all_events_stats`, as well as
        the 4 daily metrics of `clim_keys`.
    """

    # Ignore all-nan timeseries for performance sake
    if np.isnan(sst).all():
        nans = np.array([
            np.nan
            for _ in range(t[0].astype('datetime64[Y]').astype(int) + 1970, t[-1].astype('datetime64[Y]').astype(int) + 1970+1)
        ])

        return tuple(nans for _ in opts.mhws_stats)

    # Array manipulation to fit mhw module requirements
    t = t.astype('datetime64[D]').astype(int) + 719163 # to ordinal time
    temp = sst.copy()

    # Computing MHWs using mhw module
    mhws, clim = mhw.detect(t, temp, climatologyPeriod=clim_period)

    # Adjusting values
    categories = {'Moderate': 1, 'Strong': 2, 'Severe': 3, 'Extreme': 4}
    mhws['category'] = [categories[cat] for cat in mhws['category']]
    clim['sst'] = temp.copy()

    # Return event and daily metrics
    return tuple(
        [
            np.pad(np.array(mhws[stat], dtype=np.float64), (0, math.ceil(len(t)/8)-len(mhws[stat])), constant_values=np.nan)
            for stat in mhws_all_events_stats
        ] + [
            clim[key].astype(float) for key in clim_keys
        ]
    )


def get_mhw_ts_from_ds(
        ds_mhws: xr.Dataset,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        depth: Optional[float] = None,

        calculate_mhw_mask: bool = True
) -> Tuple[NDArray, Dict]:
    """
    Format the statistics produced by 'compute_mhw_all_events' in a more friendly way.

    Parameters
    ----------
    ds_mhws: xr.Dataset
        Dataset produced by 'compute_mhw_all_events'.

    lon: float, optional
        If more than one longitude coordinates, must select one.

    lat: float, optional
        If more than one latitude coordinates, must select one.

    depth: float, optional
        If more than one depth coordinates, must select one.

    calculate_mhw_mask: bool, default=True
        Option to spread event metrics on the daily grid.

    Returns
    ----------
    time: np.array
        Array of time

    mhws: dict
        All MHWs metrics.
    """

    # Extract only one time series
    if not (lon is None or lat is None or depth is None):
        ds = ds_mhws.sel(lon=lon, lat=lat, depth=depth, method='nearest')
    elif not depth is None:
        ds = ds_mhws.sel(depth=depth, method='nearest')
    else:
        ds = ds_mhws

    time = ds.time.values

    # There is no MHW event
    if ds.time_start.isnull().all():
        return -1, -1

    first_nan = ds.time_start.isnull().argmax().item()

    if first_nan != 0:
        # Cut the dataset when all the values are nan
        ds = ds.isel(event_number=slice(0, first_nan))

    # Format the statistics in a more friendly way
    mhws = {}

    for stat in mhws_all_events_stats:
        mhws[stat] = ds[stat].values

    for stat in clim_keys:
        mhws['clim_'+stat] = ds['clim_'+stat].values

    mhws['n_events'] = len(mhws['time_start'])

    mhws['date_start'] = [date.fromordinal(time) for time in mhws['time_start'].astype(int)]
    mhws['date_end'] = [date.fromordinal(time) for time in mhws['time_end'].astype(int)]
    mhws['date_peak'] = [date.fromordinal(time) for time in mhws['time_peak'].astype(int)]

    mhws['index_start'] = mhws['index_start'].astype(int)
    mhws['index_end'] = mhws['index_end'].astype(int)

    # Spread event metrics on the daily grid
    if calculate_mhw_mask:
        mhw_mask = np.ones(ds.time.size, dtype=bool)
        mhws['mhw_number'] = np.full(ds.time.size, np.nan)

        for stat in mhws_all_events_useful_stats:
            mhws[f'mhw_{stat}'] = np.full(ds.time.size, np.nan)

        for ev in range(mhws['n_events']):
            mhw_mask[mhws['index_start'][ev]:mhws['index_end'][ev]+1] = False
            mhws['mhw_number'][mhws['index_start'][ev]:mhws['index_end'][ev]+1] = ev+1

            for stat in mhws_all_events_useful_stats:
                mhws[f'mhw_{stat}'][mhws['index_start'][ev]:mhws['index_end'][ev]+1] = mhws[stat][ev]

        mhws['mhw_intensity'] = mhws['clim_sst'] - mhws['clim_seas']

        mhws['mhw_intensity'][mhw_mask] = np.nan

        mhws['mhw_mask'] = mhw_mask

    # Returns the time array and all the metrics
    return time, mhws
