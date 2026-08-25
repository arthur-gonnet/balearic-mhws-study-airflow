# balearic_mhws_OLD folder

This is the original thesis-era code, kept as reference until its remaining parts are ported to the
operational pipeline in `src/balearic_mhws/`. `load_save_dataset.py`, `mhw_computer.py`, `options.py`
and `marineHeatWaves.py` have already been ported (to `balearic_mhws.data.io`/`.download`,
`balearic_mhws.processing.compute_mhws`, `balearic_mhws.config`, and
`balearic_mhws.processing.marineHeatWaves` respectively) and removed from here.

## What is in here?

 - `basic_plotter.py` :
     Functions for generating figures from datasets. Not yet ported.

 - `utils.py` :
     Helper functions for various tasks (regional masks, Mann-Kendall trend test, etc). Not yet ported.

## License

The *marineHeatWaves* module for python developed by Eric C. J. Oliver has been modified for the purpose of the thesis. The modifications are the following :

 1. **Add severity metrics**: The severity metric has been added as described in the report.
 2. **Calculate means by days and not by event**: This modification makes longer events have more impact on the annual mean (of intensity or severity).
 3. **Option to cut events between 31st December and 1st January**: This modification makes that for a given year, the annual metrics are only based on what happened this given year. This introduces a bias in the mean duration metric, as some events would be split and thus show a lower duration.
