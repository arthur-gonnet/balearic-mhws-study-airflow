########################################################################################################################
##################################### USER NOTES #######################################################################
########################################################################################################################

"""
This script gathers all the code used to compute marine heatwaves metrics from a temperature datasets. 


Functions description
----------
    - apply_regional_mask(ds, region, ...):
        Either apply to a dataset or returns a regional mask.

    - extract_transect(ds, pos0, pos1):
        Extract a transect from a dataset.
        
    - lon_to_str(value, ...):
        Returns a formatted string from a longitude or latitude float value.

    - nice_range(vmin, vmax, ...) -> tuple:
        Uses matplotlib's MaxNLocator to compute nice rounded range and ticks.

    - soft_add_values(original, values, ...) -> dict:
        Softly copies the key and value pairs of one dictionnary to another.

    - soft_override_value(dict, key, value):
        Softly add a key and value pair into a dictionnary.

    - not_null(*args):
        Returns the first not null value from the passed arguments.

    - bold(text: str) -> str:
        Formats a text to Mathtext bold format.
"""

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Basic imports
import copy
from typing import Literal, Optional, List, Dict, Tuple, Any

# Advanced imports
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
from matplotlib.ticker import MaxNLocator
import pymannkendall as mk

########################################################################################################################
##################################### CODE #############################################################################
########################################################################################################################


def apply_regional_mask(
        ds: xr.Dataset | xr.DataArray,
        region: str | int,
        ds_bathy: xr.Dataset | xr.DataArray,
        return_mask: bool = False
) -> xr.Dataset | xr.DataArray:
    """
    Either apply to a dataset or returns a regional mask.

    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dataset to which apply the regional dataset.
    
    region: str | int
        The region that is wanted. Should be one of: `'CC'`, `'continental_coast'`,
        `'BIC'`, `'balearic_coast'`, `'DBS'`, `'balearic_sea_deep'`, `'DWAR'`, `'west_algerian_deep'`,
        or one of their associated index.
    
    ds_bathy: xr.Dataset | xr.DataArray
        Bathymetry Dataset or DataArray. If Dataset, the variable must be named `'depth'`.
        Should use the same grid as the main dataset `ds`.
    
    return_mask: bool, default=False
        If True, the mask is not applied to the dataset, but rather, return a boolean
        mask DataArray.

    Returns
    ----------
    ds: xr.Dataset | xr.DataArray
        Either the original dataset `ds` with every grid point outside the selected
        region being changed to nans, or a boolean mask as DataArray.
    """
    
    # Compatibility snippet so that the function can work with both DataArray and Dataset
    if isinstance(ds_bathy, xr.Dataset):
        ds_bathy = ds_bathy.depth

    # Find region
    regions = [
        'continental_coast',
        'balearic_coast',
        'balearic_sea_deep',
        'west_algerian_deep',
    ]
    regions_shortnames = {
        'CC': 'continental_coast',
        'BIC': 'balearic_coast',
        'DBS': 'balearic_sea_deep',
        'DWAR': 'west_algerian_deep',
    }

    if isinstance(region, int): region = regions[region]
    if region.upper() in regions_shortnames: region = regions_shortnames[region.upper()]

    # Create grid points
    lon, lat = np.meshgrid(ds.lon.values, ds.lat.values)

    # Preparing iterables
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    # Function to add back coordinates
    def to_xarray(mask):
        return xr.DataArray(
            mask,
            coords={'lat': ds.lat, 'lon': ds.lon},
            dims=('lat', 'lon')
        )

    if region in ['continental_coast', 'balearic_coast']:
        # Define polygon
        coords_island_poly = [
            (0.69, 39.19),
            (3.4, 40.75),
            (4.86, 40.26),
            (4.5, 39.2),
            (1, 38)
        ]  # (lon, lat)
        island_polygon  = Polygon(coords_island_poly)
        island_mask     = np.array([island_polygon.contains(Point(lon_, lat_)) for lon_, lat_ in zip(lon_flat, lat_flat)])
        island_mask_2d  = island_mask.reshape(len(ds.lat), len(ds.lon))

        # Apply only the continental_coast
        if region == 'continental_coast':
            continental_coast_mask = to_xarray((ds_bathy < 200) & ~island_mask_2d)

            if return_mask:
                return continental_coast_mask
            else:
                return ds.where(continental_coast_mask)
        
        # Apply only the balearic_coast
        elif region == 'balearic_coast':
            balearic_coast_mask = to_xarray((ds_bathy < 200) & island_mask_2d)

            if return_mask:
                return balearic_coast_mask
            else:
                return ds.where(balearic_coast_mask)

    elif region in ['balearic_sea_deep', 'west_algerian_deep']:
        # Define polygon
        coords_basin_poly = [
            (20, 20), (-5, 20), (-5, 38.8), # Basic rectangle
            (0.12, 38.8), (1.41, 38.83), # Ibiza Channel
            (1.6, 39.08), (3.13, 39.94), # Mallorca Channel
            (4.24, 40), (20, 40), # Menorca Channel
        ]  # (lon, lat)
        basin_polygon   = Polygon(coords_basin_poly)
        basin_mask      = np.array([basin_polygon.contains(Point(lon_, lat_)) for lon_, lat_ in zip(lon_flat, lat_flat)])
        basin_mask_2d   = basin_mask.reshape(len(ds.lat), len(ds.lon))

        # Apply only the balearic_sea_deep
        if region == 'balearic_sea_deep':
            balearic_sea_deep_mask = to_xarray((ds_bathy > 200) & ~basin_mask_2d)

            if return_mask:
                return balearic_sea_deep_mask
            else:
                return ds.where(balearic_sea_deep_mask)
        
        # Apply only the west_algerian_deep
        elif region == 'west_algerian_deep':
            west_algerian_deep_mask = to_xarray((ds_bathy > 200) & basin_mask_2d)

            if return_mask:
                return west_algerian_deep_mask
            else:
                return ds.where(west_algerian_deep_mask)

    # Print the bad news
    print(f"Region not found {region}")



def apply_mk_test(y: NDArray):
    """
    Wrapper function to apply the Mann-Kendall test on a dataset

    Parameters
    ----------
    y: numpy.array
        A time series.

    Returns
    ----------
    h: bool
        Assess if a trend is present in the time series.

    p: float
        The p-value of the significance test.

    slope: float
        The slope of the trend.
    """
    
    if np.isnan(y).all():
        return np.nan, np.nan, np.nan
    
    trend, h, p, z, tau, s, var_s, slope, intercept = mk.hamed_rao_modification_test(y, alpha=0.05)
    
    return h, p, slope



def extract_transect(
        ds: xr.Dataset | xr.DataArray,
        pos0: Tuple[float, float],
        pos1: Tuple[float, float]
) -> xr.Dataset:
    """
    Extract a transect from a dataset.
    
    Parameters
    ----------
    ds: xr.Dataset | xr.DataArray
        Dataset from which the transect should be extracted.
    
    pos0: tuple[float, float]
        Starting position of the transect, given as (lat, lon).
        If the longitude of `pos1` is less than the one from `pos0`, then `pos0`
        becomes the ending point of the transect to keep longitudes ascending.
    
    pos1: tuple[float, float]
        Ending position of the transect, given as (lat, lon).

    Returns
    ----------
    ds_transect: xr.Dataset
        Dataset containing the transect.
    """

    # First and last point
    lat0, lon0 = pos0
    lat1, lon1 = pos1

    if lon1 < lon0:
        print(f"Reversing point order to keep longitudes ascending (given initial lon0:{lon0:.1f}, lon1:{lon1:.1f}).")
        lon0, lon1 = lon1, lon0
        lat0, lat1 = lat1, lat0

    # All coordinates
    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values

    # Find closest points to start and end point on the grid
    i0 = np.abs(lat_vals - lat0).argmin()
    j0 = np.abs(lon_vals - lon0).argmin()
    i1 = np.abs(lat_vals - lat1).argmin()
    j1 = np.abs(lon_vals - lon1).argmin()

    # Dind number of points
    npts = max(abs(i1 - i0), abs(j1 - j0)) + 1

    # Find intermediate indexes
    lat_idx = np.round(np.linspace(i0, i1, npts)).astype(int)
    lon_idx = np.round(np.linspace(j0, j1, npts)).astype(int)

    # Get lats and lons of the transect
    lat_real = ds['lat'].isel(lat=('transect', lat_idx))
    lon_real = ds['lon'].isel(lon=('transect', lon_idx))

    # Calculate the distance along transect as new coordinate
    dist_km = [0.0] + [geodesic((lat0, lon0), (float(lat_real[i]), float(lon_real[i]))).km for i in range(1, npts)]
    dist_da = xr.DataArray(dist_km, dims='transect')

    # Extract data from dataset
    ds_transect = ds.isel(
        lat = xr.DataArray(lat_idx, dims='transect'),
        lon = xr.DataArray(lon_idx, dims='transect')
    )

    # Create final dataset with new coordinates
    ds_transect = ds_transect.assign_coords({
        'transect': np.arange(npts),
        'lat': lat_real,
        'lon': lon_real,
        'distance': dist_da
    })

    # Returning the final Dataset!
    return ds_transect


def lon_to_str(
    value: float,
    axis: str = 'lon',
    precision: int = 2,
) -> str:
    """
    Returns a formatted string from a longitude or latitude float value.
    For a longitude of `2.12587` it returns `"2.13°E"`.
    
    Parameters
    ----------
    value: float
        longitude or latitude float value.

    axis: float, default='lon'
        Either `'lon'` or `'lat'`.

    precision: float, default=2
        Number of digits after the comma.
    
    Returns
    ----------
    lon_str: str
        Formatted string.
    """

    # Save precision as string pattern
    precision = f":.{precision}f" if not precision is None else ''

    suffix = ''

    # Get suffix
    if axis == 'lon':
        suffix = 'W' if value < 0 else '' if value == 0 else 'E'
    elif axis == 'lat':
        suffix = 'S' if value < 0 else '' if value == 0 else 'N'

    # Final pattern
    pattern = "{value" + str(precision) + "}°{suffix}"

    # Finally apply pattern to obtain final string
    return pattern.format(value=abs(value), suffix=suffix)


def nice_range(
    vmin: Optional[float],
    vmax: Optional[float],
    nbins: int = 10
) -> Tuple[Optional[float], Optional[float]]:
    """
    Uses matplotlib's MaxNLocator to compute nice rounded range and ticks.
    
    Parameters
    ----------
    vmin: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    vmax: float | None
        Initial vmin value. Note that if one of vmin or vmax is None, the range is returned unchanged.
    
    nbins: int | 'auto', default: 10
        Matplotlib's MaxNLocator parameter.
        Maximum number of intervals; one less than max number of ticks.
        If the string 'auto', the number of bins will be automatically determined based on the length of the axis.

    Returns
    ----------
    nice_range: tuple
        Modified range.
    """
    
    # There is nothing to change if there are empty values
    if vmin is None or vmax is None:
        return vmin, vmax

    # There is nothing to change if there are empty values
    if vmin == 0 and vmax == 0:
    # if (vmin < 1e-10 and vmin > 1e-10 and vmax < 1e-10 and vmax > 1e-10) or (vmin is None and vmax is None):
        return 0., 1.
    
    # Use MaxNLocator to find good range
    locator = MaxNLocator(nbins=nbins, prune=None)
    ticks = locator.tick_values(vmin, vmax)

    # Return the result
    return ticks[0], ticks[-1] # ticks[1] - ticks[0] is the step


def soft_add_values(original: Dict, values: Dict, inplace: bool = True):
    """
    Copies the key and value pairs of the `values` dictionnary into the `original` dictionnary.
    If a key exists in both `values` and `original` dictionnary, the one of the `original` dictionnary is kept.

    Parameters
    ----------
    original: dict
        Original dictionnary to modify.
    
    values: dict
        Dictionnary containing the values to add to the `original` dictionnary.
    
    inplace: bool, default=True
        If True, this function returns a modified copy of `original` dictionnary using `copy.deepcopy()`.
        If False, the original dictionnary is modified in place.
    
    Returns
    ----------
    original: dict
        Either a copy or original `original` dictionnary.
    """

    if not inplace:
        original = copy.deepcopy(original)

    for key in values:
        if key not in original:
            original[key] = values[key]
    
    return original


def soft_override_value(dict: Dict, key: Any, value: Any):
    """
    Add a key and value pair into a dictionnary only if `value` is not `None`.

    Parameters
    ----------
    dict: dict
        Original dictionnary to modify.
    
    key
        The key of the value to modify.
    
    value
        The value to be added. If `None`, then nothing is done.
    """

    if not value is None:
        dict[key] = value


def not_null(*args):
    """
    Returns the first not null value from the passed arguments.

    Parameters
    ----------
    *args
        Choices of value. The first not null value is returned.
    """

    for arg in args:
        if not arg is None:
            return arg


def bold(text: str) -> str:
    """
    Formats a text to Mathtext bold format.
    """
    return '\n'.join(r"$\bf{" + word.replace(' ', ' \\ ') + "}$" for word in str(text).split('\n'))
