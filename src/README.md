# Code folder

The code here computes MHW metrics from temperature data and produces figures displaying these
metrics. The operational path is the Airflow pipeline (see the [root README](../README.md)); the
notebooks kept here are an interactive way to explore the data and produce figures by hand.

## What is in here?

 - `balearic_mhws/` :
     The installable package used by everything else - the pipeline, the sbatch scripts and the
     notebooks all import from it. See [balearic_mhws/README.md](balearic_mhws/README.md).

 - `notebooks/01_check_packages.ipynb` :
     Python notebook meant to help the Python environment setup process.

 - `notebooks/02_download_data.ipynb` :
     Python notebook that automates data downloading of the REP and MEDREA datasets.

 - `notebooks/03_mhws_computing.ipynb` :
     Python notebook computing MHW metrics and saving them to Zarr stores.

 - `notebooks/04_report_plotting.ipynb` :
     Python notebook loading pre-computed MHW datasets in order to produce figures displaying MHW
     metrics (for the report).

 - `notebooks/05_presentation_plotting.ipynb` :
     Python notebook loading pre-computed MHW datasets in order to produce figures displaying MHW
     metrics (for the presentation).

## How should this code be runned?

The notebooks run against the same Zarr stores the pipeline populates, so the data has to exist
first - either by running the pipeline (see the [root README](../README.md)) or by running
`02_download_data.ipynb` itself.

 1. Make sure the Python environment is correctly setup by running `01_check_packages.ipynb`:

> - Python 3.10+
> - **Scientific stack**: numpy, xarray, dask, zarr, scipy
> - **Plotting**: matplotlib, cartopy, geopy, shapely, cmocean, roman
> - **Statistics**: pymannkendall
> - **Dataset downloading** : copernicusmarine
> - **Development Environment**: Jupyter Notebooks, VS Code

 2. Make sure the required data has been downloaded by running `02_download_data.ipynb`.
 3. Run `03_mhws_computing.ipynb` to compute MHW datasets.
 4. Finally, run `04_report_plotting.ipynb` or `05_presentation_plotting.ipynb` to generate the
    desired figures.

*Note: Visual Studio Code was used during development, the workflow being thought to integrate easily with it.*

## Troubleshooting

**> The code can't find my data files.** <br>
The Zarr stores have to be populated first - run the pipeline or `02_download_data.ipynb`. Paths
are resolved from `balearic_mhws.config`, which reads `BALEARIC_DATA_DIR` from the environment and
defaults to `<project>/data`.

**> I get errors with specific packages.** <br>
Run `01_check_packages.ipynb` to check dependencies, ensuring they are installed under a correct
version. Some versions are pinned deliberately (see `requirements.txt` for why).

**For any other query, please contact the author.**
