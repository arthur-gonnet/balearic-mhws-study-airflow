########################################################################################################################
##################################### USER NOTES #######################################################################
########################################################################################################################

"""
This script gather all the functions used to plot figures. 


Functions description
----------
    - subplot(nrows, ncols, subplots_settings, ...):
        Builds a figure with subplots with matplotlib.

    - plot_map(lon, lat, data, ...):
        Plots 2D data (lon, lat) onto a map with land features.

    - plot_transect(depth, abscissa, data, ...):
        Plots 2D data (depth, any) onto a transect.

    - plot_vertical_mean(depths, vars, ...):
        Plots 1D data (depth) as a line plot.

    - plot_timeserie(times, vars, ...):
        Plots 1D data (time) as a line plot.

    - plot_bars(depths, vars, ...):
        Plots 1D data (depth) as a bar plot.

    - get_locator_from_ticks(ticks, ...):
        Helper function that return a matplotlib Locator from a giving ticks parameter.

      
Examples
----------
    from balearic_mhws.data.io import load_mhws
    from balearic_mhws.plotting.plot import plot_map, plot_vertical_mean, plot_timeserie, plot_transect, subplot

    ds_mhws_medrea = load_mhws("yearly", "medrea", False, "balears", (1987,2021))

    plot_map(
        lon = ds_mhws_medrea.lon,
        lat = ds_mhws_medrea.lat,
        data = ds_mhws_medrea.total_days.sel(year=2022, depth=0, method='nearest')
    )

    plot_vertical_mean(
        depths = ds_mhws_medrea.depth,
        vars = ds_mhws_medrea.total_days.mean(dim=['lon', 'lat']).mean(dim='year')
    )

    plot_timeserie(
        times = ds_mhws_medrea.year,
        vars = ds_mhws_medrea.total_days.sel(depth=200, method='nearest').mean(dim=['lon', 'lat'])
    )

    subplots_settings = {}

    for i, year in enumerate([2021, 2022]):
        subplots_settings.append(dict(
            pos = i+1,
            func = plot_map,
            lon = ds_mhws_medrea.lon,
            lat = ds_mhws_medrea.lat,
            data = ds_mhws_medrea.total_days.sel(year=year, depth=0, method='nearest')
        ))

    subplot(
        nrows = 2, ncols = 1,
        subplots_settings = subplots_settings
    )

    ds_mhws_transect = load_mhws("yearly", "medrea", False, "transect_ibiza_channel", (1987,2021))

    plot_transect(
        abscissa = ds_mhws_transect.distance,
        depth = ds_mhws_transect.depth,
        data = ds_mhws_transect.total_days.sel(year=2022)
    )
"""

########################################################################################################################
##################################### IMPORTS ##########################################################################
########################################################################################################################

# Basic imports
from typing import Literal, Optional, List, Dict, Tuple

# Advanced imports
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter, FuncFormatter, FixedLocator, AutoMinorLocator
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local imports
from .utils import soft_override_value, soft_add_values


########################################################################################################################
##################################### COLORMAPS #########################################################################
########################################################################################################################

# Ported from the thesis codebase's pyscripts/options.py - these are plotting-only definitions
# that were deliberately dropped from balearic_mhws.config during that earlier port, but are
# still needed here since the plotting functions below (and their callers) reference them via
# the `cmap` parameter.

category_colors = ['#ffe7a7', '#fdb941', '#fe821f', '#e84e1c', '#be2e19', '#8b1c15', '#5b0b07', '#2b0403']
category_colormap_lin = mcolors.LinearSegmentedColormap.from_list('linear_category_cmap', category_colors)
category_colormap_seq = lambda n : mcolors.ListedColormap(category_colormap_lin(np.linspace(0, 1, n)))

# Category from Oliver
category_colors_oliver = { 1: '#ffd86e', 2: '#ff621f', 3: '#df391b', 4: '#861a15'}
category_colormap_oliver = mcolors.ListedColormap([category_colors_oliver[i] for i in category_colors_oliver])

# Colormaps to be used for each MHWs statistics
mhws_stats_cmaps = {
    'count':                'cmo.ice_r',                 # 'Purples',
    'total_days':           'cmo.tempo',                 # 'Blues',
    'moderate_days':        'cmo.tempo',                 # 'Blues',
    'strong_days':          'cmo.tempo',                 # 'Blues',
    'severe_days':          'cmo.tempo',                 # 'Blues',
    'extreme_days':         'cmo.tempo',                 # 'Blues',

    'duration':             'cmo.matter',                   # 'Blues', 'cmo.matter'

    'total_icum':           'cmo.solar_r',               # 'Oranges',
    'intensity_cumulative': 'cmo.solar_r',               # 'Oranges',

    'intensity_max_max':    'cmo.amp',                   # 'Reds',
    'intensity_max':        'cmo.amp',                   # 'Reds',
    'intensity_mean':       'cmo.amp',                   # 'Reds',
    'intensity_mean_byday': 'cmo.amp',
    'intensity_var':        'cmo.amp',                   # 'Reds',

    'total_scum':           category_colormap_lin,
    'severity_cumulative':  category_colormap_lin,

    'severity_max_max':     category_colormap_lin,
    'severity_max':         category_colormap_lin,
    'severity_mean':        category_colormap_lin,
    'severity_mean_byday':  category_colormap_lin,
    'severity_var':         category_colormap_lin,

    'rate_onset':           'cmo.speed',                 # 'YlOrRd',
    'rate_decline':         'cmo.speed',                 # 'YlOrRd',

    'temp_min':             'cmo.thermal',
    'temp_mean':            'cmo.thermal',
    'temp_max':             'cmo.thermal',
    'mean_thresh':          'cmo.amp',
}


########################################################################################################################
##################################### CODE #############################################################################
########################################################################################################################


def subplot(
        # Parameter of subplot
        nrows: int, ncols: int,
        subplots_settings: List[dict],
        pad_subplots: Optional[Tuple[float, float]] = None,

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        figsize: Optional[Tuple[float, float]] = None,
        subplotsize: Tuple[float, float] = (7.3, 4.67),
        figdpi: float = 200,
        fig_fontsize: int = 14,
        fig_title: Optional[str] = None,
        fig_is_a_map: bool = False,

        # Parameter of colorbar
        fig_cbar: bool = False,
        fig_cbar_row: bool = False,
        fig_vmin: Optional[float] = None,
        fig_vmax: Optional[float] = None,
        fig_cmap: Optional[str] = None,
        fig_cbar_unit: Optional[str] = None,
        fig_cbar_ticks: Optional[int | Tuple[float, float] | List[float]] = None,
        fig_cbar_orientation: str = 'vertical',
        fig_cbar_pad: float = 0.005,
        fig_cbar_fraction: float = 0.025,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",
):
    """
    Builds a figure with subplots with matplotlib.

    Parameters
    ----------
    nrows: int 
        The number of rows in the subplot grid.

    ncols: int 
        The number of columns in the subplot grid.

    subplots_settings: list[dict]
        A list containing the settings of each subplot. Each subplot setting dictionnary
        must contain :

        'pos': the index of the position of the subplot as in the function `plt.subplot()`. 
        'func': the plotting function to plot on the desired axe.
        ...: all the arguments to be passed to the plotting function.

        Note : if using a colorbar, the displayed data show be named 'data'.
    
    pad_subplots: tuple[float, float], optional
        This option adjust the padding existing between the subplots as the form `(horizontal, vertical)`. Uses a relative unit.
        Beware that it disables the constrained_layout property (enabled by default).
    
    fig: plt.Figure, optional
        If created beforehand, the figure to which subplots will be added. Otherwise, a new
        figure is created.

    figsize: tuple[float, float], optional
        Width, height of the figure in inches. Overrides `subplotsize`. 

    subplotsize: tuple[float, float], default=(7.3, 4.67)
        The size of each subplots in inches. Thus, `figsize=(ncols*subplotsize[0], nrows*subplotsize[1])`.
        Overriden if `figsize` is defined.
    
    figdpi: float, default=100   
        The resolution of the figure in dots-per-inch.

    fig_fontsize: int, default=14
        Fontsize that will be applied to the whole figure.
    
    fig_title: str, optional
        Title of the whole figure.
    
    fig_is_a_map: bool, default=False
        This option should be enabled when displaying a map. It defines the projection
        mode for the axe, which is required for some cases when plotting maps.
    
    fig_cbar: bool, default=False
        Add a colorbar at the figure scope. When doing so, the vmin and vmax values are
        defined at figure scope with the minimal and maximal values of all the subplots.
        Also, subplot-scoped colorbar are disabled. Overriden if `fig_cbar_row` is enabled.
    
    fig_cbar_row: bool, default=False
        Add a colorbar for each row of subplots. When doing so, the vmin and vmax values are
        defined at row scope with the minimal and maximal values of all the subplots in the given row.
        Also, subplot-scoped colorbar are disabled. Overrides `fig_cbar`.
    
    fig_cbar_ticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the colorbar. Giving an int will use a `MaxNLocator`.
        Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        Giving a list of float will use a `FixedLocator` with the values given.
    
    fig_cbar_pad: float, default=0.005
        Defines the pad between the plot and the colorbar. Uses a relative unit.

    fig_cbar_fraction: float, default=0.025
        Defines the fraction size of the colorbar.

    show_plots: bool, default=False
        Shows the pending plots to show using `plt.show()`.
    """
    
    # Apply subplotsize so that figsize = ncols*subplotwidth, nrows*subplotheight
    if figsize is None:
        figsize = (ncols*subplotsize[0], nrows*subplotsize[1])
    
    # If adding a figure colorbar, compute the range of it using data of whole figure
    if fig_cbar and not fig_cbar_row:
        if fig_vmin is None:
            fig_vmin = np.nanmin([np.nanmin(subplot_setting['data']) for subplot_setting in subplots_settings])
        
        if fig_vmax is None:
            fig_vmax = np.nanmax([np.nanmax(subplot_setting['data']) for subplot_setting in subplots_settings])
            
    # Apply fontsize globally
    with plt.rc_context({'font.size': fig_fontsize}):
        # Create a new figure only if the figure didn't exist beforehand
        if not fig:
            fig = plt.figure(figsize=figsize, dpi=figdpi, constrained_layout=(pad_subplots is None))
            print(f"Making figure of {figsize[0]*figdpi}x{figsize[1]*figdpi}px")

        # Adjust subplot padding (beware that it disables the constrained_layout property)
        if pad_subplots:
            fig.subplots_adjust(wspace=pad_subplots[0], hspace=pad_subplots[1])

        # If the colorbars are defined by rows, compute the range of it using data of the row
        if fig_cbar_row:
            fig_vmins = []
            fig_vmaxs = []

            for row in range(nrows):
                vmin = np.nanmin([np.nanmin(subplot_setting['data']) for subplot_setting in subplots_settings if (subplot_setting['pos']-1)//ncols == row])
                vmax = np.nanmax([np.nanmax(subplot_setting['data']) for subplot_setting in subplots_settings if (subplot_setting['pos']-1)//ncols == row])
                
                # FIX: This does not work for subplots spanning on more than one position

                fig_vmins.append(vmin)
                fig_vmaxs.append(vmax)

        # Plot every subplot on its own axe
        for subplot_setting in subplots_settings:
            subplot_setting = subplot_setting.copy()
            pos = subplot_setting.pop('pos')
            plotting_func = subplot_setting.pop('func')

            row = (pos-1) // ncols
            col = (pos-1) % ncols

            # Creating the subplot ax, using projection or not depending if it is a map or not
            if fig_is_a_map:
                projection_mode = ccrs.PlateCarree()
                ax = fig.add_subplot(nrows, ncols, pos, projection=projection_mode)
            else:
                ax = fig.add_subplot(nrows, ncols, pos)

            # If the colorbar is at the row scope, adjust subplot scoped colorbar
            if fig_cbar_row:
                soft_override_value(subplot_setting, "vmin", fig_vmins[row])
                soft_override_value(subplot_setting, "vmax", fig_vmaxs[row])
                
                # If it is the last column of the row, displays the colorbar of the row
                if col == ncols-1:
                    soft_add_values(subplot_setting, {"add_cbar": True})
                    soft_override_value(subplot_setting, "cbar_shrink", fig_cbar_fraction)
                    soft_override_value(subplot_setting, "cbar_pad", fig_cbar_pad)
                    soft_override_value(subplot_setting, "cbar_unit", fig_cbar_unit)
                
                else:
                    soft_add_values(subplot_setting, {"add_cbar": False})
            
            # If the colorbar is at the figure scope, disable subplot scoped colorbar
            elif fig_cbar:
                soft_add_values(subplot_setting, {"add_cbar": False})

            # Add options
            soft_override_value(subplot_setting, "vmin", fig_vmin)
            soft_override_value(subplot_setting, "vmax", fig_vmax)
            soft_override_value(subplot_setting, "cmap", fig_cmap)
            soft_override_value(subplot_setting, "fontsize", fig_fontsize)


            # Finally, plotting onto the axe
            plotting_func(fig=fig, ax=ax, **(subplot_setting))
        
        # Add figure title
        if fig_title:
            fig.suptitle(fig_title)

        # If the colorbar is at the figure scope, displays it 
        if fig_cbar and not fig_cbar_row:
            norm = mcolors.Normalize(vmin=fig_vmin, vmax=fig_vmax)
            mappable = cm.ScalarMappable(norm=norm, cmap=fig_cmap)

            opts = dict(
                fraction=fig_cbar_fraction, pad=fig_cbar_pad, orientation=fig_cbar_orientation
            )

            fig_cbar_tick_labels = None
            
            # Find the correct ticks for the color bar
            if fig_cbar_ticks:
                if isinstance(fig_cbar_ticks, list) and isinstance(fig_cbar_ticks[0], list):
                    fig_cbar_ticks, fig_cbar_tick_labels = fig_cbar_ticks

                cbar_locator = get_locator_from_ticks(fig_cbar_ticks)
                opts["ticks"] = cbar_locator

            # Add the colorbar
            cbar = fig.colorbar(mappable, label=fig_cbar_unit, ax=fig.axes, **opts)

            # If ticks labels are specified, add them
            if fig_cbar_tick_labels:
                cbar.ax.set_xticklabels(fig_cbar_tick_labels)
        
        # Align the y labels
        fig.align_ylabels()

        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png")
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig



def plot_map(
        # Parameter of data
        lon: xr.DataArray = [0],
        lat: xr.DataArray = [0],
        data: xr.DataArray = [[np.nan]],

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10, 6),
        figdpi: float = 100,
        fig_kwargs: Dict = {},

        # Parameter of text
        fontsize: float = 14,
        title: Optional[str] = None,
        fontsize_title: float = 1.2,
        ylabel: Optional[str] = None,
        ylabel_pad: float = -0.22,
        texts: List[dict] = [],

        # Parameters of graph
        extent: Optional[List[float] | str] = "balears",
        aspect: Optional[float] = 1.29, # Best fit
        cmap: str | mcolors.Colormap = 'viridis',
        zero_to_nan: bool = False,
        xticks: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks: Optional[int | Tuple[float, float] | List[float]] = None,
        bottom_labels: bool = True,
        left_labels: bool = True,
        legend: bool | dict = False,
        pcolormesh_kwargs: dict = {},

        # Parameter of contours
        contours_levels: Optional[int | list[float]] = None,
        contours_data = None,
        contours_kwargs: Dict = {},

        # Parameter of colorbar
        add_cbar: bool = True,
        cbar_unit: str = "",
        cbar_orientation: str = "vertical",
        cbar_shrink: float = 1,
        cbar_pad: float = 0.005,
        cbar_ticks: Optional[int | Tuple[float, float] | List[float]] = None,
        cbar_inversed: bool = False,
        cbar_kwargs: Dict = {},
        vmin: Optional[float] = None, vmax: Optional[float] = None,
        vlim: Optional[Tuple[float, float]] = None,

        # Parameter for saving the figure
        save_plot: bool = False,
        save_path: str = "",

        show_plots: bool = False,

        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots 2D data onto a map with land features using matplotlib's `pcolormesh` function.

    Parameters
    ----------
    lon: xr.DataArray
        DataArray containing the longitude coordinates of the data.

    lat: xr.DataArray,
        DataArray containing the latitude coordinates of the data.
    
    data: xr.DataArray
        DataArray containing the data.
    
    fig: plt.Figure, optional
        If created beforehand, the figure to plot on. Otherwise, a new figure is created.
    
    ax: plt.Axes, optional
        If created beforehand, the axe to plot on. Otherwise, a new axe is created.
    
    figsize: tuple[float, float], default=(10, 6)
        If creating a new figure, (width, height) of the figure in inches.
    
    figdpi: float, default=100   
        The resolution of the figure in dots-per-inch.
    
    fig_kwargs: dict, optional   
        Arguments passed to `plt.figure(...)`.
    
    fontsize: int, default=14
        Fontsize that will be applied to the plot.
    
    title: str, optional
        Title of the axe.

    fontsize_title: int, default=1.2
        Fontsize that will be applied to the title, relative to the current fontsize.
        Final fontsize of title is `fontsize*fontsize_title`.
    
    extent: list[float] | str | None, default="balears"
        Extent of the map to plot. Can be a list of the shape `[min_lon, max_lon, min_lat, max_lat]`,
        one region of `'med'` or `'balears'`, or None to fit the extent of the data.

    aspect: float | None, default=1.29
        Aspect ratio of the map. A value of 1.29 gives Mercator-like aspect in the Balearic region,
        even if using a `PlateCarree` projection. This enables to avoid projecting to Mercator, as it
        can be computationally-requiring, thus saving some computational resources.
    
    cmap: str | matplotlib.colors.Colormap, default='viridis'
        The colormap to be used when displaying the data. Passed directly to the `pcolormesh` function.

    zero_to_nan: bool, default=False
        Changes every zero values of the data into nans before plotting.
    
    contours_levels: int | list[float], optional
        Option to add contour levels on the map. If given as int, a list of levels is generated using
        `MaxNLocator`, otherwise if given as list, using the list of levels directly.
    
    contours_labels_fontsize: int, default=8
        Defines the fontsize of the contour labels.
    
    xticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis. Giving an int will use a `MaxNLocator`.
        Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.

    bottom_labels: bool, default=True
        Defines if the labels of the bottom ticks should be displayed or not.
    
    left_labels: bool, default=True
        Defines if the labels of the left ticks should be displayed or not.
    
    show_plots: bool, default=False
        Defines if showing the figure after plotting.
    
    save_plot: bool, default=False
        Defines if saving the figure after plotting.
    
    save_path: bool, default=False
        Defines the path to which save the figure.

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`

    Example
    --------
        import xarray as xr
        from balearic_mhws.plotting.plot import plot_map

        ds = xr.load_dataset(...)

        plot_map(lon=ds.lon, lon=ds.lat, lon=ds.sst)
    """

    if vlim and (not vmin) and (not vmax):
        vmin, vmax = vlim

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # Define the projection mode for the map
        projection_mode = ccrs.PlateCarree()

        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight', **fig_kwargs)
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111, projection=projection_mode)

        # Apply region selector
        if extent == "med":
            extent = [-6, 36.33, 30, 46]
        elif extent == "balears":
            extent = [-1, 5, 37.7, 41]

        # Defines the plot extent
        if extent:
            ax.set_extent(extent, crs=projection_mode)

        # Defines the plot aspect
        if aspect:
            ax.set_aspect(aspect) # This is a trick for performance sake, giving faster results, it imitates Mercator projection in 10sec vs 60sec
        
        # Change the grid settings and the coordinates labels
        gl = ax.gridlines(linestyle=':', draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = bottom_labels
        gl.left_labels = left_labels
        
        # Define the x ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            gl.xlocator = xlocator
        
        # Define the y ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            gl.ylocator = ylocator

        # Option for plotting
        pcm_opts = {
            'transform': projection_mode,
            'cmap': cmap,
            'vmin': vmin,
            'vmax': vmax,
            
            # Extra
            # "shading": "nearest"
            **pcolormesh_kwargs
        }
        
        # Changes 0 into nans if asked
        if zero_to_nan:
            data = data.where(data != 0, np.nan)

        # if vmin is None:
        #     vmin = None if np.isnan(data).all() else np.nanmin(data)
        
        # if vmax is None:
        #     vmax = None if np.isnan(data).all() else np.nanmax(data)

        # Plotting the data!!
        pcm = ax.pcolormesh(lon, lat, data, zorder=-1, shading="auto", **pcm_opts)
        ax.set_title(title, fontsize=fontsize*fontsize_title)

        # If ylabel is asked, adding it as so
        if ylabel:
            ax.text(ylabel_pad, 0.55, ylabel, va='bottom', ha='center',
                rotation='vertical', rotation_mode='anchor', fontsize=fontsize*fontsize_title,
                transform=ax.transAxes)

        # Shows the isolines on the plots
        if contours_levels:
            contours_kwargs = contours_kwargs.copy()

            if isinstance(contours_levels, int):
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)

                # If the contour level is a int, generate nice contour levels using MaxNLocator
                locator = MaxNLocator(contours_levels)
                contours_levels = locator.tick_values(vmin, vmax)

            contours = ax.contour(*contours_data, levels=contours_levels, transform=projection_mode, **contours_kwargs)
            # ax.clabel(contours, levels=contours_levels, colors='k', fontsize=contours_labels_fontsize, zorder=-1)
            # contours = ax.contour(lon, lat, data, levels=contours_levels, colors='k', linewidths=1, alpha=0.6, transform=projection_mode, zorder=-1)
            # ax.clabel(contours, levels=contours_levels, colors='k', fontsize=contours_labels_fontsize, zorder=-1)

        # Add colorbar
        if add_cbar:
            opts = dict(
                label = cbar_unit,
                orientation = cbar_orientation,
                shrink = cbar_shrink,
                pad = cbar_pad,
                **cbar_kwargs
            )
            
            if cbar_ticks:
                cbar_locator = get_locator_from_ticks(cbar_ticks)
                opts["ticks"] = cbar_locator

            cbar = fig.colorbar(pcm, ax=ax, **opts)

            if cbar_inversed:
                cbar.ax.invert_yaxis()


        # Additions of coastlines and land
        ax.coastlines(resolution="10m")       # Coastlines
        ax.add_feature(cfeature.LAND)        # Land
        # ax.add_feature(cfeature.BORDERS, alpha=0.2)

        if legend:
            if isinstance(legend, dict):
                ax.legend(**legend)

            else:
                ax.legend()

        for text in texts:
            ax.text(transform=ax.transAxes, **text)
        
        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png", transparent=True)
            plt.clf()
            plt.close("all")
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig, ax



def plot_transect(
        # Parameter of data
        depth: xr.DataArray,
        abscissa: xr.DataArray,
        data: xr.DataArray,

        along_lon: bool = False,
        along_lat: bool = False,
        abscissa_is_time: bool = False,

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10,6),
        figdpi: float = 100,

        # Parameter of text
        fontsize: int = 14,
        title: Optional[str] = None,
        show_pos: bool = True,

        # Parameter of colorbar
        add_cbar: bool = True,
        cbar_unit: Optional[str] = None,
        cbar_orientation: str = "vertical",
        cbar_shrink: float = 1,
        cbar_pad: float = 0.005,
        cbar_ticks: Optional[int | Tuple[float, float] | List[float]] = None,
        cbar_inversed: bool = False,
        vmin: Optional[float] = None, vmax: Optional[float] = None,

        # Parameters of graph
        cmap: str | mcolors.Colormap = "managua",
        zero_to_nan: bool = False,
        norm: str = "linear",
        contours_levels: Optional[int | List[float]] = None,
        contours_labels_fontsize: int = 8,
        xticks: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks: Optional[int | Tuple[float, float] | List[float]] = None,
        bottom_labels: bool = True,
        left_labels: bool = True,

        # Parameter for saving the figure
        save_plot: bool = False,
        save_path: str = "",
        dpi: int = 160,

        show_plots: bool = False,
):
    """
    Plots 2D data (depth, any) onto a transect.

    """

    # Sometimes, data is given transposed, then transposing it makes it work
    if data.shape != (len(depth), len(abscissa)):
        data = data.T

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)

        # Change the grid settings and the coordinates labels
        ax.grid(True, ls=':', alpha=0.7)
        
        # Define the x ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)
        
        # Define the y ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        # Option for plotting
        pcm_opts = {
            "cmap": cmap,
            "norm": norm,
            
            # Extra
            # "shading": "nearest"
        }

        # Changes 0 into nans if asked
        if zero_to_nan:
            data = data.where(data != 0, np.nan)

        # if vmin is None:
        #     vmin = None if np.isnan(data).all() else np.nanmin(data)
        
        # if vmax is None:
        #     vmax = None if np.isnan(data).all() else np.nanmax(data)
        #     vmin, vmax = nice_range(vmin, vmax)

        # Plotting the data!!
        pcm = ax.pcolormesh(abscissa, depth, data, shading="auto", vmin=vmin, vmax=vmax, **pcm_opts)
        ax.set_title(title)

        # Invert the y axis, as depth should be given in positive values
        ax.invert_yaxis()
        
        # Format the depth values to include meters
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f} m"))
        
        # If plotting along an horizontal basis
        if not abscissa_is_time:
            # If abscissa is given in longitudinal degrees
            if along_lon:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x)}°{'W' if x < 0 else '' if x == 0 else 'E'}"))

            # If abscissa is given in latitudinal degrees
            elif along_lat:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x)}°{'S' if x < 0 else '' if x == 0 else 'N'}"))

            # Otherwise, abscissa should be given in km
            else:
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}km"))
            
            if along_lat and show_pos:
                lat0 = data.lat.isel(transect=0, depth=0).item()
                lat1 = data.lat.isel(transect=data.transect.size-1, depth=0).item()
                ax.text(
                    0.01, 0.01,
                    f"{abs(lat0):.1f}°{'S' if lat0 < 0 else '' if lat0 == 0 else 'N'}",
                    ha='left', va='bottom', transform=ax.transAxes
                )
                ax.text(
                    0.99, 0.01,
                    f"{abs(lat1):.1f}°{'S' if lat1 < 0 else '' if lat1 == 0 else 'N'}",
                    ha='right', va='bottom', transform=ax.transAxes
                )
            elif show_pos:
                lon0 = data.lon.isel(transect=0, depth=0).item()
                lon1 = data.lon.isel(transect=data.transect.size-1, depth=0).item()
                ax.text(
                    0.01, 0.01,
                    f"{abs(lon0):.1f}°{'W' if lon0 < 0 else '' if lon0 == 0 else 'E'}",
                    ha='left', va='bottom', transform=ax.transAxes
                )
                ax.text(
                    0.99, 0.01,
                    f"{abs(lon1):.1f}°{'W' if lon1 < 0 else '' if lon1 == 0 else 'E'}",
                    ha='right', va='bottom', transform=ax.transAxes
                )

        # Hide labels on axis if not necessary
        if not bottom_labels:
            ax.set_xticklabels([])
        if not left_labels:
            ax.set_yticklabels([])

        # Shows the isolines on the plots
        if contours_levels:
            if isinstance(contours_levels, int):
                if vmin is None:
                    vmin = np.nanmin(data)
                if vmax is None:
                    vmax = np.nanmax(data)

                # If the contour level is a int, generate nice contour levels using MaxNLocator
                locator = MaxNLocator(contours_levels)
                contours_levels = locator.tick_values(vmin, vmax)
            
            contours = ax.contour(abscissa, depth, data, levels=contours_levels, colors='k', linewidths=1, alpha=0.6)
            ax.clabel(contours, levels=contours_levels, colors='k', fontsize=contours_labels_fontsize)


        # Add colorbar
        if add_cbar:
            opts = dict(
                label=cbar_unit, orientation=cbar_orientation, shrink=cbar_shrink, pad=cbar_pad,
            )

            if cbar_ticks:
                cbar_locator = get_locator_from_ticks(cbar_ticks)
                opts["ticks"] = cbar_locator

            cbar = fig.colorbar(pcm, ax=ax, **opts)

            if cbar_inversed:
                cbar.ax.invert_yaxis()

        # If saving the figure is asked
        if save_plot:
            fig.savefig(save_path, format="png", dpi=dpi)
        
        # If showing the figure is asked
        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    # If the figure is not to be shown nor to be saved, returning it
    if not show_plots and not save_plot:
        return fig, ax



def plot_vertical_mean(
        # Parameter of data
        depths: Dict[str, xr.DataArray] | xr.DataArray,
        vars: Dict[str, xr.DataArray] | xr.DataArray,
        colors: Optional[Dict[str, str] | str] = None,
        ls: Optional[Dict[str, str] | str] = None,

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (4, 6),
        figdpi = 200,

        # Parameter of text
        fontsize: int = 14,
        title: Optional[str] = None,
        unit: Optional[str] = None,
        ylabel: Optional[str] = None,

        # Parameters of graph
        grid: bool = True,
        nans_to_zero: bool = False,
        xlim: Tuple[Optional[float], Optional[float]] = (None, None),
        ylim: Tuple[Optional[float], Optional[float]] = (None, None),
        xticks: Optional[int | Tuple[float, float] | List[float]] = None,
        xticks_minor: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks_minor: Optional[int | Tuple[float, float] | List[float]] = None,
        xticks_formatter: plt.Formatter = None,
        yticks_formatter: plt.Formatter = None,
        bottom_labels: bool = True,
        left_labels: bool = True,
        top_labels: bool = False,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    """
    Plots 1D data (depth) as a line plot.

    Parameters
    ----------
    depths: dict[str, xr.DataArray] | xr.DataArray
        Dictionary attributing to a name a data array representing the depth array associated with the variable to plot. For plotting only one variable, a single DataArray can be passed.

    vars: dict[str, xr.DataArray] | xr.DataArray
        Dictionary attributing to a name a data array representing the variable data to plot. For plotting only one variable, a single DataArray can be passed.
    
    colors: dict[str, str] | str, optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be 'black', 'r'...
    
    ls: dict[str, str], optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be '-', '--', '-.'...
    
    fig: plt.Figure, optional
        If created beforehand, the figure to plot on. Otherwise, a new figure is created.
    
    ax: plt.Axes, optional
        If created beforehand, the axe to plot on. Otherwise, a new axe is created.
    
    figsize: tuple[float, float], default=(10, 6)
        If creating a new figure, (width, height) of the figure in inches.
    
    figdpi: float, default=200   
        The resolution of the figure in dots-per-inch.
    
    fontsize: int, default=14
        Fontsize that will be applied to the plot.

    title: str, optional
        Title of the axe.
    
    unit: str, optional
        Unit of the data.
    
    ylabel: str, optional
        Label for the y-axis. Can be "Depth [m]".

    grid: bool, default=True
        Add grid lines to the plot

    nans_to_zero: bool, default=False
        Changes every nan values of the data into zeros before plotting.

    xlim: tuple[float|None, float|None], optional
        Limit of the x-axis.

    ylim: tuple[float|None, float|None], optional
        Limit of the y-axis.
    
    xticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    xticks_minor: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `AutoMinorLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks_minor: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `AutoMinorLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    xticks_formatter: plt.Formatter, optional
        Formatter to use for the x-axis ticks.
    
    yticks_formatter: plt.Formatter, optional
        Formatter to use for the y-axis ticks.
    
    bottom_labels: bool, default=True
        Defines if the labels of the bottom ticks should be displayed or not.
    
    left_labels: bool, default=True
        Defines if the labels of the left ticks should be displayed or not.
    
    top_labels: bool, default=False
        Defines if the labels of the top ticks should be displayed or not.
    
    show_plots: bool, default=False
        Defines if showing the figure after plotting.
    
    save_plot: bool, default=False
        Defines if saving the figure after plotting.
    
    save_path: bool, default=False
        Defines the path to which save the figure
    """

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)
        
        # Should handle single dataset plotting
        if isinstance(vars, dict):
            for var in vars:
                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply labels
                # if labels == 'auto':
                #     opts_args["label"] = var
                # elif labels and var in labels:
                #     opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                if isinstance(depths, dict):
                    depths_ = depths[var]
                else:
                    depths_ = depths
                
                # Finally, plot the line
                ax.plot(vars[var], depths_, **opts_args)
        
        else:
            if unit == None and isinstance(vars, xr.DataArray) and vars.attrs.get("unit"):
                unit = vars.attrs.get("unit")

            # if labels == 'auto': labels=None

            if nans_to_zero:
                vars = np.nan_to_num(vars)

            ax.plot(vars, depths, color=colors, ls=ls, lw=1)  

        
        if ylim == (None, None):
            if isinstance(depths, dict):
                ylim = (
                    np.nanmin([np.nanmin(depths[name]) for name in depths]),
                    np.nanmax([np.nanmax(depths[name]) for name in depths])
                )
            
            else:
                ylim = (np.nanmin(depths), np.nanmax(depths))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)

        if xticks_formatter:
            ax.xaxis.set_major_formatter(xticks_formatter)
        
        # Set manually the ordinate ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        if yticks_minor:
            ylocator = get_locator_from_ticks(yticks_minor, which="minor")
            ax.yaxis.set_minor_locator(ylocator)

        if yticks_formatter:
            ax.yaxis.set_major_formatter(yticks_formatter)
        

        # if not bottom_labels:
        #     ax.set_xticklabels([])
        
        # if not left_labels:
        #     ax.set_yticklabels([])
        # else:
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.tick_params(which='both', top=top_labels, labeltop=top_labels, bottom=bottom_labels, labelbottom=bottom_labels, left=left_labels, labelleft=left_labels)

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if unit:
            ax.set_xlabel(f"[{unit}]")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        plt.gca().invert_yaxis()

        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        # if labels and legend: ax.legend()

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax



def plot_timeserie(
        # Parameter of data
        times: Dict[str, xr.DataArray] | xr.DataArray,
        vars: Dict[str, xr.DataArray] | xr.DataArray,
        labels: Optional[Dict[str, str] | str] = 'auto',
        colors: Optional[Dict[str, str] | str] = None,
        ls: Optional[Dict[str, str] | str] = None,

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (18, 5),
        figdpi: float = 200,

        # Parameter of text
        fontsize: int = 14,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        texts: list[dict] = [],
        legend: bool | dict = True,

        # Parameters of graph
        grid: bool = True,
        nans_to_zero: bool = False,
        xlim: tuple[int|None, int|None] = (None, None),
        ylim: tuple[int|None, int|None] = (None, None),
        xticks: Optional[int | Tuple[float, float] | List[float]] = None,
        xticks_minor: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks_minor: Optional[int | Tuple[float, float] | List[float]] = None,
        yticks_formatter: plt.Formatter = None,
        bottom_labels: bool = True,
        left_labels: bool = True,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    """
    Plots 1D data (time) as a line plot.

    Parameters
    ----------
    times: dict[str, xr.DataArray] | xr.DataArray
        Dictionary attributing to a name a data array representing the time array associated with the variable to plot. For plotting only one variable, a single DataArray can be passed.
    
    vars: dict[str, xr.DataArray] | xr.DataArray
        Dictionary attributing to a name a data array representing the variable data to plot. For plotting only one variable, a single DataArray can be passed.
    
    labels: dict[str, str] | str, optional
        Dictionary attributing to a name the labels to assign to the variable to plot. Pass a single value to apply it to all variables.
    
    colors: dict[str, str] | str, optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be 'black', 'r'...
    
    ls: dict[str, str], optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be '-', '--', '-.'...
    
    fig: plt.Figure, optional
        If created beforehand, the figure to plot on. Otherwise, a new figure is created.
    
    ax: plt.Axes, optional
        If created beforehand, the axe to plot on. Otherwise, a new axe is created.
    
    figsize: tuple[float, float], default=(18, 5)
        If creating a new figure, (width, height) of the figure in inches.
    
    figdpi: float, default=200   
        The resolution of the figure in dots-per-inch.
    
    fontsize: int, default=14
        Fontsize that will be applied to the plot.

    title: str, optional
        Title of the axe.
    
    xlabel: str, optional
        Label for the x-axis. Can be "Time".
    
    ylabel: str, optional
        Label for the y-axis. Can be "Total days [days]".
    
    texts: list[dict], optional
        For each dictionnary of this variable, a text will be added to the plot as: `ax.text(transform=ax.transAxes, **text)`. Can be for example
        ```
            texts = {
                'x': 0.98,
                'y': 0.04,
                's': "1)",
                'ha': 'right',
                'va': 'bottom',
                'fontsize': 18
            }
        ```
    
    legend: bool | dict, default=True
        If True, will add a legend to the plot. If a dict, a legend will be added as: `ax.legend(**legend)`. Can be for example

        ```
            from matplotlib.patches import Patch

            legend = {
                'handles': [
                    Patch(color='C0'),
                    Patch(color='C1'),
                    Patch(color='C2'),
                    Patch(color='C3'),
                ],
                'labels': ["SPC", "BIC", "BS", "NWA"],
                'loc': 'lower right',
                'handlelength': 0.7,
                'handletextpad': 0.4,
                'fontsize': 18,
            }
        ```

    grid: bool, default=True
        Add grid lines to the plot

    nans_to_zero: bool, default=False
        Changes every nan values of the data into zeros before plotting.

    xlim: tuple[float|None, float|None], optional
        Limit of the x-axis.

    ylim: tuple[float|None, float|None], optional
        Limit of the y-axis.
    
    xticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    xticks_minor: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `AutoMinorLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks_minor: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `AutoMinorLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks_formatter: plt.Formatter, optional
        Formatter to use for the y-axis ticks.
    
    bottom_labels: bool, default=True
        Defines if the labels of the bottom ticks should be displayed or not.
    
    left_labels: bool, default=True
        Defines if the labels of the left ticks should be displayed or not.
    
    show_plots: bool, default=False
        Defines if showing the figure after plotting.
    
    save_plot: bool, default=False
        Defines if saving the figure after plotting.
    
    save_path: bool, default=False
        Defines the path to which save the figure
    """

    with plt.rc_context({'font.size': fontsize}):
        # Plotting
        if fig is None or ax is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
            ax = fig.add_subplot(111)
        
        # Should handle single dataset plotting
        if isinstance(vars, dict):
            for var in vars:
                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply labels
                if labels == 'auto':
                    opts_args["label"] = var
                elif labels and var in labels:
                    opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                if isinstance(times, dict):
                    times_ = times[var]
                else:
                    times_ = times
                
                # Finally, plot the line
                ax.plot(times_, vars[var], lw=1, **opts_args)
        
        else:
            # if unit == None and isinstance(vars, xr.DataArray) and vars.attrs.get("unit"):
            #     unit = vars.attrs.get("unit")

            if labels == 'auto': labels=None

            if nans_to_zero:
                vars = np.nan_to_num(vars)

            ax.plot(times, vars, color=colors, label=labels, ls=ls, lw=1)  

        
        if xlim == (None, None):
            if isinstance(times, dict):
                xlim = (
                    np.nanmin([np.nanmin(times[name]) for name in times]),
                    np.nanmax([np.nanmax(times[name]) for name in times])
                )
            
            else:
                xlim = (np.nanmin(times), np.nanmax(times))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)
        
        # Set manually the ordinate ticks
        if yticks:
            ylocator = get_locator_from_ticks(yticks)
            ax.yaxis.set_major_locator(ylocator)

        if yticks_minor:
            ylocator = get_locator_from_ticks(yticks_minor, which="minor")
            ax.yaxis.set_minor_locator(ylocator)

        if yticks_formatter:
            ax.yaxis.set_major_formatter(yticks_formatter)
        

        if not bottom_labels:
            ax.set_xticklabels([])
        
        if not left_labels:
            ax.set_yticklabels([])

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        if labels and legend:
            if isinstance(legend, dict):
                ax.legend(**legend)

            else:
                ax.legend()

        for text in texts:
            ax.text(transform=ax.transAxes, **text)

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax



def plot_bars(
        # Parameter of data
        depths: xr.DataArray,
        vars: Dict[str, xr.DataArray],
        stds: Optional[Dict[str, xr.DataArray]] = None,
        colors: Optional[Dict[str, str] | str] = None,
        ls: Optional[Dict[str, str] | str] = None,
        hatch: Optional[Dict[str, str] | str] = None,

        # Parameter of figure
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (10, 5),
        figdpi: float = 100,

        # Parameter of text
        fontsize: int = 14,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        texts: List[Dict] = [],

        # Parameters of graph
        grid: bool = True,
        nans_to_zero: bool = False,
        xlim: Tuple[Optional[float], Optional[float]] = (None, None),
        xticks: Optional[int | Tuple[float, float] | List[float]] = None,
        xticks_minor = None,
        yticks: Optional[int | Tuple[float, float] | List[float]] = None,
        xticks_formatter = None,
        bottom_labels: bool = True,
        left_labels: bool = True,
        top_labels: bool = False,
        bars_pad: float = 2,

        # Parameter for saving the figure
        show_plots: bool = False,
        save_plot: bool = False,
        save_path: str = "",

        **kwargs
):
    """
    Plots 1D data (depth) as a bar plot.

    Parameters
    ----------
    depths: xr.DataArray
        A DataArray representing the depth array associated with the variables to plot.

    vars: dict[str, xr.DataArray]
        Dictionary attributing to a name a data array representing the variable data to plot. The DataArray must have a `'depth'` dimension matching `depths`.
    
    stds: dict[str, xr.DataArray] | str, optional
        Dictionary attributing to a name the std array to assign to the variable to plot.
    
    colors: dict[str, str] | str, optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be 'black', 'r'...
    
    ls: dict[str, str] | str, optional
        Dictionary attributing to a name the color to assign to the variable to plot. Pass a single value to apply it to all variables. Can be '-', '--', '-.'...
    
    hatch: dict[str, dict[float, str]] | dict[str, str] | str, optional
        Can be:
        - Dictionary attributing to a name (variable) the array giving for the hatch to assign for every depth.
        - Dictionary attributing to a name the hatch to assign to the variable to plot.
        - Pass a single value to apply it to all variables.

        Can be '//', 'x'...
    
    fig: plt.Figure, optional
        If created beforehand, the figure to plot on. Otherwise, a new figure is created.
    
    ax: plt.Axes, optional
        If created beforehand, the axe to plot on. Otherwise, a new axe is created.
    
    figsize: tuple[float, float], default=(10, 6)
        If creating a new figure, (width, height) of the figure in inches.
    
    figdpi: float, default=100   
        The resolution of the figure in dots-per-inch.
    
    fontsize: int, default=14
        Fontsize that will be applied to the plot.

    title: str, optional
        Title of the axe.
    
    xlabel: str, optional
        Label for the x-axis. Can be "Time".
    
    ylabel: str, optional
        Label for the y-axis. Can be "Total days [days]".
    
    texts: list[dict], optional
        For each dictionnary of this variable, a text will be added to the plot as: `ax.text(transform=ax.transAxes, **text)`. Can be for example
        ```
            texts = {
                'x': 0.98,
                'y': 0.04,
                's': "1)",
                'ha': 'right',
                'va': 'bottom',
                'fontsize': 18
            }
        ```grid: bool, default=True
        Add grid lines to the plot

    nans_to_zero: bool, default=False
        Changes every nan values of the data into zeros before plotting.

    xlim: tuple[float|None, float|None], optional
        Limit of the x-axis.

    xticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    xticks_minor: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `AutoMinorLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    yticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the y-axis.
        - Giving an int will use a `MaxNLocator`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    xticks_formatter: plt.Formatter, optional
        Formatter to use for the x-axis ticks.
    
    bottom_labels: bool, default=True
        Defines if the labels of the bottom ticks should be displayed or not.
    
    left_labels: bool, default=True
        Defines if the labels of the left ticks should be displayed or not.
    
    top_labels: bool, default=True
        Defines if the labels of the left ticks should be displayed or not.
    
    bars_pad: float, default=2
        Space added between each depth levels.
    
    show_plots: bool, default=False
        Defines if showing the figure after plotting.
    
    save_plot: bool, default=False
        Defines if saving the figure after plotting.
    
    save_path: bool, default=False
        Defines the path to which save the figure
    """
    
    if not colors:
        colors = {
            var: f'C{i}'
            for i, var in enumerate(vars)
        }

    # Apply fontsize globally
    with plt.rc_context({'font.size': fontsize}):
        # If the figure is not configured, initialise a new one
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=figdpi, layout='tight')
        
        # If the axe is not configured, initialise a new one
        if ax is None:
            ax = fig.add_subplot(111)
        
        last_tick = 0
        ticks_pos = []

        should_define_xlim = xlim == (None, None)

        for depth in depths:
            vars_displayed = 0

            for var in vars:
                value = vars[var].sel(depth=depth)
                
                # If all nan, return
                if value.isnull().all():
                    continue

                opts_args = {}

                # Apply the required color
                if colors:
                    if isinstance(colors, dict) and var in colors:
                        opts_args["color"] = colors[var]
                    elif not isinstance(colors, dict):
                        opts_args["color"] = colors
                
                # Apply the required line style
                if ls:
                    if isinstance(ls, dict) and var in ls:
                        opts_args["ls"] = ls[var]
                    elif not isinstance(ls, dict):
                        opts_args["ls"] = ls
                
                # Apply the required line style
                if hatch:
                    if isinstance(hatch, dict) and var in hatch:
                        if isinstance(hatch[var], dict) and depth in hatch[var]:
                            opts_args["hatch"] = hatch[var][depth]
                        else:
                            opts_args["hatch"] = hatch[var]
                        
                    elif not isinstance(hatch, dict):
                        opts_args["hatch"] = hatch
                
                # Apply labels
                # if labels == 'auto':
                #     opts_args["label"] = var
                # elif labels and var in labels:
                #     opts_args["label"] = labels[var]
                
                # Apply nan filter
                if nans_to_zero:
                    vars[var] = np.nan_to_num(vars[var])
                
                # Choose right time array
                # if isinstance(depths, dict):
                #     depths_ = depths[var]
                # else:
                #     depths_ = depths

                if stds:
                    std = stds[var].sel(depth=depth)
                    opts_args['xerr'] = std
                    # opts_args['ecolor'] = 'k'
                    opts_args['capsize'] = 3
                    opts_args['error_kw'] = {
                        'alpha': 0.4,
                    }
                
                # Finally, plot the line
                # ax.plot(vars[var], depths_, **opts_args)
                a=ax.barh(
                    last_tick + vars_displayed + 0.5,
                    value,
                    **opts_args
                )

                for bc in a:
                    import matplotlib.colors as c
                    bc._hatch_color = c.to_rgba("w")
                    bc.stale = True


                # if stds:
                #     ax.barh(
                #         last_tick + vars_displayed + 0.5,
                #         std,
                #         **opts_args
                #     )

                if should_define_xlim:
                    if xlim == (None, None):
                        xlim = (np.nanmin(value), np.nanmax(value))
                    else:
                        xlim = (min(np.nanmin(value), xlim[0]), max(np.nanmax(value), xlim[1]))
                    
                    if stds:
                        std = stds[var].sel(depth=depth)
                        xlim = (min(np.nanmin(value-std), xlim[0]), max(np.nanmax(value+std), xlim[1]))

                vars_displayed += 1

            ticks_pos.append(last_tick + vars_displayed/2)
            last_tick += vars_displayed + bars_pad
        
        if should_define_xlim:
            vmin, vmax = xlim
            pad = (vmax - vmin)*0.1

            vmin = 0 if vmin*(vmin-pad)<=0 else vmin-pad
            vmax = 0 if vmax*(vmax+pad)<=0 else vmax+pad

            xlim = (vmin, vmax)
        
        # if ylim == (None, None):
        #     if isinstance(depths, dict):
        #         ylim = (
        #             np.nanmin([np.nanmin(depths[name]) for name in depths]),
        #             np.nanmax([np.nanmax(depths[name]) for name in depths])
        #         )
            
        #     else:
        #         ylim = (np.nanmin(depths), np.nanmax(depths))
        
        # Set manually the abscissa ticks
        if xticks:
            xlocator = get_locator_from_ticks(xticks)
            ax.xaxis.set_major_locator(xlocator)

        if xticks_minor:
            xlocator = get_locator_from_ticks(xticks_minor, which="minor")
            ax.xaxis.set_minor_locator(xlocator)

        if xticks_formatter:
            ax.xaxis.set_major_formatter(xticks_formatter)
        
        # Set manually the ordinate ticks
        # if yticks:
        #     ylocator = get_locator_from_ticks(yticks)
        #     ax.yaxis.set_major_locator(ylocator)

        # if yticks_minor:
        #     ylocator = get_locator_from_ticks(yticks_minor, which="minor")
        #     ax.yaxis.set_minor_locator(ylocator)
        
        ticks_pos = np.array(ticks_pos)

        idx = range(len(ticks_pos))

        if isinstance(yticks, int):
            idx = range(0, len(ticks_pos), yticks)

        ax.set_yticks(ticks_pos, minor=True)
        ax.set_yticks(ticks_pos[idx], labels=[f"{depth:.0f} m" for depth in depths[idx]])


        # if not bottom_labels:
        #     ax.set_xticklabels([])
        
        # if not left_labels:
        #     ax.set_yticklabels([])
        # else:
        # if left_labels:
        #     ax.set_ylabel("Depth [m]")

        ax.tick_params(which='both', top=top_labels, labeltop=top_labels, bottom=bottom_labels, labelbottom=bottom_labels, left=left_labels, labelleft=left_labels)
        ax.tick_params(which='both', length=0)

        # Change the ax color
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('green')

        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        # ax.set_ylim(-bars_pad, last_tick)
        ax.set_ylim(-bars_pad/2, last_tick-bars_pad/2)
        ax.invert_yaxis()


        if grid: ax.grid(alpha=0.5)
        if grid: ax.grid(which="minor", alpha=0.3, ls="--")
        # if labels and legend: ax.legend()

        for text in texts:
            ax.text(transform=ax.transAxes, **text)

        if show_plots:
            plt.show()
            plt.clf()
            plt.close("all")
    
    return fig, ax



def get_locator_from_ticks(
        ticks: Optional[int | Tuple[float, float] | List[float]] = None,
        which: str = "major",
):
    """
    Helper function that return a matplotlib Locator from a giving ticks parameter.

    Parameters
    ----------
    xticks: int | tuple[float, float] | list[float], optional
        Defines the ticks of the x-axis.
        - Giving an int will use a `MaxNLocator` or a `AutoMinorLocator`, depending on `which`.
        - Giving a tuple will use a `MultipleLocator`, the first float being the base and the second the offset.
        - Giving a list of float will use a `FixedLocator` with the values given.
    
    which: str, default='major'
        If `ticks` is an int, assign either `MaxNLocator` (if major) or  `AutoMinorLocator` (if minor).

    Returns
    -------
    locator: matplotlib.ticker.Locator
        A matplotlib Locator.
    """
    
    # If an int passed
    if isinstance(ticks, int):
        if which == "major":
            return MaxNLocator(nbins=ticks)
        elif which == "minor":
            return AutoMinorLocator(ticks)
    
    # If a tuple passed
    elif isinstance(ticks, tuple):
        if len(ticks) == 2:
            return MultipleLocator(base=ticks[0], offset=ticks[1])
        else:
            return MultipleLocator(base=ticks[0])
    
    # If a list passed
    elif isinstance(ticks, list):
        return FixedLocator(ticks)

    # Not found locator
    return MaxNLocator()
