# balearic_mhws package

The installable package holding all the logic of the project. The Airflow DAGs, the pipeline CLI,
the sbatch scripts and the notebooks all import from here, so the same code runs whatever the
entrypoint.

## What is in here?

 - `config.py` :
     Single source of truth for the filesystem paths, the Copernicus Marine credentials and
     dataset ids, the chunking used when writing Zarr stores, the climatology period, and the
     metadata of the MHW statistics (short names, long names, units).

 - `data/` :
     Reading and writing the Zarr stores, and downloading the raw data from the Copernicus Marine
     Service. See [data/README.md](data/README.md).

 - `processing/` :
     The MHW detection algorithm and its parallelized wrappers.
     See [processing/README.md](processing/README.md).

 - `plotting/` :
     Figure generation, and the helpers it needs (regional masks, transects, trend tests).
     See [plotting/README.md](plotting/README.md).

## Configuration

Paths are resolved from environment variables so that the same code works unmodified in local
development, inside the Airflow container and on an HPC node, each of which mounts the data
directory differently. `BALEARIC_DATA_DIR` sets the root of the data directory, and defaults to
`<project>/data`.

The climatology period lives in `config.CLIMATOLOGY_PERIOD` and is deliberately not exposed as a
parameter anywhere in the pipeline, so that the results of every run stay comparable.
