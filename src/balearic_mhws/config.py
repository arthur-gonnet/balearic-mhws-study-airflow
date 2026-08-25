"""
Centralised configuration for the balearic_mhws package: filesystem paths, Copernicus Marine
credentials, and MHW statistics metadata.

Paths are resolved from environment variables so the same code works unmodified in local dev,
inside the Airflow container, and on an HPC node - each mounts/mounts the data directory
differently.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

########################################################################################################################
##################################### PATHS #############################################################################
########################################################################################################################

# Root of the data directory. Defaults to <repo>/data when running outside a container.
DATA_DIR = Path(os.environ.get(
    "BALEARIC_DATA_DIR",
    Path(__file__).resolve().parents[2] / "data",
))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRODUCTS_DIR = DATA_DIR / "products"

# Raw datasets, ingested as consolidated Zarr stores (see balearic_mhws.data.download)
REP_ZARR = RAW_DIR / "REP" / "rep.zarr"
MEDREA_ZARR = RAW_DIR / "MEDREA" / "medrea.zarr"
BATHY_ZARR = RAW_DIR / "bathymetry" / "medrea_bathy.zarr"

# Chunking used when writing/appending to the raw Zarr stores. A full year per chunk keeps
# appends chunk-aligned (new ingests add whole chunks) while still being cheap to rechunk to
# a single time chunk before running the MHW computation.
ZARR_TIME_CHUNK = 366
ZARR_SPATIAL_CHUNK = 100

# Pattern for computed MHW datasets, mirroring the old NetCDF layout but writing Zarr stores.
MHWS_ZARR_PATTERN = str(PROCESSED_DIR / "mhws" / "{type}" / "{dataset}_mhws_{region}{detrended}_{clim_start}_{clim_end}.zarr")

########################################################################################################################
##################################### DOWNLOAD SOURCES ##################################################################
########################################################################################################################

# Copernicus Marine Service credentials. Read from the environment - no interactive prompts,
# so the download stage can run unattended from Airflow/Slurm.
COPERNICUS_USERNAME = os.environ.get("USERNAME_COPERNICUS") or None
COPERNICUS_PASSWORD = os.environ.get("PASSWORD_COPERNICUS") or None

REP_DATASET_ID = "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021"
MEDREA_DATASET_ID = "med-cmcc-tem-rean-d"
MEDREA_BATHY_DATASET_ID = "cmems_mod_med_phy_my_4.2km_static"

# Spatial/depth extent considered by this study
SPATIAL_EXTENT: Dict[str, float] = {
    "minimum_longitude": -0.9,
    "maximum_longitude": 5.1,
    "minimum_latitude": 37.6,
    "maximum_latitude": 41.1,
    "minimum_depth": 0,
    "maximum_depth": 3000,
}

########################################################################################################################
##################################### MHWS STATISTICS ###################################################################
########################################################################################################################

# Basic MHWs statistics
mhws_basic_stats = [
    'total_days',
    'duration',
    'total_icum',
    'intensity_max_max',
    'intensity_mean_byday',
    'severity_mean_byday',
]

# All MHWs statistics
mhws_stats = [
    # Counts of events/days
    'count',
    'total_days',
    'moderate_days',
    'strong_days',
    'severe_days',
    'extreme_days',

    # Duration statistic
    'duration',

    # Cumulative intensity statistics
    'total_icum',
    'intensity_cumulative',

    # Intensity statistics
    'intensity_max_max',
    'intensity_max',
    'intensity_mean',
    'intensity_mean_byday',
    'intensity_var',

    # Cumulative intensity statistics
    'total_scum',
    'severity_cumulative',

    # Intensity statistics
    'severity_max_max',
    'severity_max',
    'severity_mean',
    'severity_mean_byday',
    'severity_var',

    # Rate onset/decline statistics
    'rate_onset',
    'rate_decline',

    # Temperature statistics
    'temp_min',
    'temp_mean',
    'temp_max',
]

# Short names for MHWs statistics
mhws_stats_shortname = {
    'count':                "Annual MHW events",
    'total_days':           "Total days",
    'moderate_days':        "Annual moderate MHW days",
    'strong_days':          "Annual strong MHW days",
    'severe_days':          "Annual severe MHW days",
    'extreme_days':         "Annual extreme MHW days",

    'duration':             "Mean duration",

    'total_icum':           "Cumulative intensity",
    'intensity_cumulative': "Mean MHW cumulative intensity",

    'intensity_max_max':    "Maximum intensity",
    'intensity_max':        "Mean MHW maximum intensity",
    'intensity_mean':       "Mean MHW event intensity",
    'intensity_mean_byday': "Mean intensity",
    'intensity_var':        "Mean MHW intensity variability",

    'total_scum':           "Annual MHW cumulative severity",
    'severity_cumulative':  "Mean MHW cumulative severity",

    'severity_max_max':     "Maximum MHW severity",
    'severity_max':         "Mean MHW maximum severity",
    'severity_mean':        "Mean MHW event severity",
    'severity_mean_byday':  "Mean severity",
    'severity_var':         "Mean MHW severity variability",

    'rate_onset':           "Mean MHW onset rate",
    'rate_decline':         "Mean MHW decline rate",

    'temp_min':             "Minimum temperature",
    'temp_mean':            "Annual mean temperature",
    'temp_max':             "Maximal temperature",
    'mean_thresh':          "Mean 90th percentile",
}

# Long names for MHWs statistics (from Oliver's code)
mhws_stats_longname = {
    'count':                "Total MHW count per year",
    'total_days':           "Total number of MHW days per year",
    'moderate_days':        "Total number of moderate MHW days per year",
    'strong_days':          "Total number of strong MHW days per year",
    'severe_days':          "Total number of severe MHW days per year",
    'extreme_days':         "Total number of extreme MHW days per year",

    'duration':             "Average MHW duration per year",

    'total_icum':           "Total cumulative intensity over all MHWs per year",
    'intensity_cumulative': "Average MHW \"cumulative intensity\" per year",

    'intensity_max_max':    "Maximum MHW \"maximum (peak) intensity\" per year",
    'intensity_max':        "Average MHW \"maximum (peak) intensity\" per year",
    'intensity_mean':       "Average MHW event \"mean intensity\" per year",
    'intensity_mean_byday': "Average MHW day \"mean intensity\" per year",
    'intensity_var':        "Average MHW \"intensity variability\" per year",

    'total_scum':           "Annual MHW cumulative severity",
    'severity_cumulative':  "Mean MHW cumulative severity",

    'severity_max_max':     "Maximum MHW severity",
    'severity_max':         "Mean MHW maximum severity",
    'severity_mean':        "Mean MHW event severity",
    'severity_mean_byday':  "Mean MHW day severity",
    'severity_var':         "Mean MHW severity variability",

    'rate_onset':           "Average MHW onset rate per year",
    'rate_decline':         "Average MHW decline rate per year",

    'temp_min':             "Minimum temperature per year",
    'temp_mean':            "Mean temperature per year",
    'temp_max':             "Maximum temperature per year",
    'mean_thresh':          "Mean 90th threshold",
}

# MHWs statistics units
mhws_stats_units = {
    'count':                "count",
    'total_days':           "days",
    'moderate_days':        "days",
    'strong_days':          "days",
    'severe_days':          "days",
    'extreme_days':         "days",

    'duration':             "days",

    'total_icum':           "°C·days",
    'intensity_cumulative': "°C·days",

    'intensity_max_max':    "°C",
    'intensity_max':        "°C",
    'intensity_mean':       "°C",
    'intensity_mean_byday': "°C",
    'intensity_var':        "°C",

    'total_scum':           "Severity Index.day",
    'severity_cumulative':  "Severity Index.day",

    'severity_max_max':     "",
    'severity_max':         "",
    'severity_mean':        "",
    'severity_mean_byday':  "",
    'severity_var':         "",

    'rate_onset':           "°C/days",
    'rate_decline':         "°C/days",

    'temp_min':             "°C",
    'temp_mean':            "°C",
    'temp_max':             "°C",
    'mean_thresh':          "°C",
}

# Description to add to generated MHWs dataset
mhw_dataset_description = "MHWs statistics computed using the marineHeatWaves " \
        "module for python developped by Eric C. J. Oliver."
mhw_yearly_dataset_description = "MHWs yearly statistics computed using the marineHeatWaves " \
        "module for python developped by Eric C. J. Oliver."

# Acknowledgment to add to generated MHWs dataset depending on the original dataset used
rep_acknowledgment = 'Generated using E.U. Copernicus Marine Service Information, ' \
        'Mediterranean Sea - High Resolution L4 Sea Surface Temperature Reprocessed (DOI: https://doi.org/10.48670/moi-00173)'
medrea_acknowledgment = 'Generated using E.U. Copernicus Marine Service Information, ' \
        'Mediterranean Sea Physics Reanalysis (DOI: https://doi.org/10.25423/CMCC/MEDSEA_MULTIYEAR_PHY_006_004_E3R1)'

# Subregions considered in the study
regions = [
    'continental_coast',
    'balearic_coast',
    'balearic_sea_deep',
    'west_algerian_deep',
]

# Short names for regions
region_shortname = {
    'continental_coast':    "Continental coast",
    'balearic_coast':       "Balearic Islands coast",
    'balearic_sea_deep':    "Balearic Sea deep",
    'west_algerian_deep':   "West Algerian Basin deep",
}
