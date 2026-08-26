# processing subpackage

The marine heatwave detection itself, and the wrappers applying it to a whole gridded dataset.

## What is in here?

 - `compute_mhws.py` :
     Computes MHW metrics from a temperature dataset. `compute_mhw_yearly` produces the annual
     metrics, `compute_mhw_all_events` the per-event ones, and `get_mhw_ts_from_ds` extracts a
     time serie from a dataset. The detection is applied point by point through
     `xarray.apply_ufunc`, so it is parallelized by dask over the grid.

 - `marineHeatWaves.py` :
     The *marineHeatWaves* module for python developed by Eric C. J. Oliver, modified for the
     purpose of the thesis (see https://github.com/ecjoliver/marineHeatWaves). This is
     third-party code and is kept as it is.

## Computation

Both wrappers stack the spatial dimensions into a single one, run the detection on every resulting
time serie, then unstack the results. A time serie that is entirely NaN, a land point or a depth
below the local seafloor for instance, gives NaN metrics instead of failing.

The 26 statistics computed are listed in `config.mhws_stats`, together with their short names,
long names and units, which are attached to the output dataset as attributes.

The climatology period must be inside the time range of the input dataset. It is always
`config.CLIMATOLOGY_PERIOD` when called from the pipeline.

## Modifications made to the marineHeatWaves module

The *marineHeatWaves* module for python developed by Eric C. J. Oliver has been modified for the purpose of the thesis. The modifications are the following :

 1. **Add severity metrics**: The severity metric has been added as described in the report.
 2. **Calculate means by days and not by event**: This modification makes longer events have more impact on the annual mean (of intensity or severity).
 3. **Option to cut events between 31st December and 1st January**: This modification makes that for a given year, the annual metrics are only based on what happened this given year. This introduces a bias in the mean duration metric, as some events would be split and thus show a lower duration.

## Note

`marineHeatWaves.blockAverage()` assumes the input covers a continuous range of years. A gap year
in the input makes the number of blocks it computes disagree with the years it finds, and the
computation fails. Backfill the missing year rather than working around it.
