# plotting subpackage

Figure generation from the computed MHW datasets, and the helpers needed to prepare the data
before plotting it.

## What is in here?

 - `plot.py` :
     The plotting functions, and the colormaps associated with each MHW statistic
     (`mhws_stats_cmaps`).

 - `utils.py` :
     Helpers used when preparing data to plot: regional masks, transect extraction, the
     Mann-Kendall trend test, and various formatting helpers.

## Plotting functions

 - `subplot(nrows, ncols, subplots_settings, ...)` :
     Builds a figure with subplots, optionally sharing a colorbar over the whole figure or over
     each row.

 - `plot_map(lon, lat, data, ...)` :
     Plots 2D data (lon, lat) onto a map with land features.

 - `plot_transect(depth, abscissa, data, ...)` :
     Plots 2D data (depth, any) onto a transect.

 - `plot_vertical_mean(depths, vars, ...)` :
     Plots 1D data (depth) as a line plot.

 - `plot_timeserie(times, vars, ...)` :
     Plots 1D data (time) as a line plot.

 - `plot_bars(depths, vars, ...)` :
     Plots 1D data (depth) as a bar plot.

## Helpers

 - `apply_regional_mask(ds, region, ds_bathy, ...)` :
     Either applies a regional mask to a dataset, or returns the mask. The regions are the ones
     considered in the study, listed in `config.regions`, and are defined from polygons and from
     the 200m isobath.

 - `extract_transect(ds, pos0, pos1)` :
     Extracts a transect between two positions, adding the distance along it as a coordinate.

 - `apply_mk_test(y)` :
     Applies the Mann-Kendall trend test to a time serie, returning whether a trend is present,
     the p-value and the slope.

The remaining helpers (`lon_to_str`, `nice_range`, `soft_add_values`, `soft_override_value`,
`not_null`, `bold`) format values and merge option dictionnaries for the plotting functions.

## Note

The bathymetry passed to `apply_regional_mask` must use the same grid as the dataset being masked.
