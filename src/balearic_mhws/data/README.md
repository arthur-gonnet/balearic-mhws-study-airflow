# data subpackage

Reading and writing the project's Zarr stores, and downloading the raw data they hold from the
Copernicus Marine Service.

## What is in here?

 - `io.py` :
     Opens the raw stores with spatio-temporal subsetting (`open_rep`, `open_medrea`,
     `open_bathy`), reads and writes the computed MHW datasets (`save_mhws`, `load_mhws`), and
     holds `write_zarr_incremental`, the helper the ingestion uses to add data to a store.

 - `download.py` :
     Downloads REP, MEDREA and bathymetry data from the Copernicus Marine Service and ingests it
     into the raw Zarr stores (`ingest_rep`, `ingest_medrea`, `ingest_bathy`).

## Ingestion

Ingestion is idempotent. Each call looks at the time coordinate already in the target store and
only downloads the (year, month) pairs missing from it, so re-running an overlapping range does
nothing for the months already covered.

Data is downloaded and written one month at a time rather than one year at a time, so an
interrupted run loses at most one month of progress.

Credentials are read from the environment, with no interactive prompt, so ingestion can run
unattended from Airflow or from a Slurm job.

## Writing to a store

`write_zarr_incremental` takes one of three paths depending on the data given to it:

 - Nothing new to write, and the call does nothing.
 - Data entirely past the end of the store, appended directly.
 - Data falling before or inside the range already stored, which happens when backfilling older
   months after newer ones were ingested. Zarr can only extend a store at its end, so the store is
   rebuilt sorted, one of its own chunks at a time. It is written to a temporary store and swapped
   in once complete, so an interrupted rebuild leaves the original store untouched.

## Chunking

The chunk sizes used when writing are set in `config.py`. Every dimension other than time uses
dask's automatic chunking rather than a single chunk per dimension. MEDREA has around 111 depth
levels, and holding a whole depth column in one chunk gives chunks of well over a gigabyte, which
is too large to handle comfortably when rebuilding a store.
