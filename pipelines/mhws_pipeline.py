"""
CLI entrypoint for the Balearic MHWs pipeline.

This is the single place download/compute logic gets invoked from - both the Airflow DAG
(`dags/balearic_mhws.py`, for local/dev runs) and the Slurm sbatch scripts (`hpc/slurm/*.sbatch`,
for large runs on a cluster) call into this CLI rather than duplicating logic.

Usage
----------
    python -m pipelines.mhws_pipeline download --dataset rep --years 1982:2024
    python -m pipelines.mhws_pipeline download --dataset medrea --years 1987:2023
    python -m pipelines.mhws_pipeline download --dataset bathy
    python -m pipelines.mhws_pipeline compute --dataset rep --clim-start 1987 --clim-end 2021
"""

import argparse
import sys
from typing import List

from balearic_mhws.data import download, io
from balearic_mhws.processing import compute_mhws

########################################################################################################################
##################################### HELPERS ###########################################################################
########################################################################################################################


def _parse_range(value: str) -> range:
    """Parses a 'start:stop' string (stop exclusive) or a single 'n' into a range."""

    if ':' in value:
        start, stop = value.split(':', 1)
        return range(int(start), int(stop))

    return range(int(value), int(value) + 1)


########################################################################################################################
##################################### SUBCOMMANDS #######################################################################
########################################################################################################################


def cmd_download(args: argparse.Namespace) -> None:
    years = _parse_range(args.years) if args.years else None
    months = _parse_range(args.months) if args.months else None

    if args.dataset == 'rep':
        kwargs = {}
        if years is not None:
            kwargs['years'] = years
        if months is not None:
            kwargs['months'] = months
        download.ingest_rep(**kwargs)

    elif args.dataset == 'medrea':
        kwargs = {}
        if years is not None:
            kwargs['years'] = years
        if months is not None:
            kwargs['months'] = months
        download.ingest_medrea(**kwargs)

    elif args.dataset == 'bathy':
        download.ingest_bathy()

    else:
        raise ValueError(f"Unknown dataset {args.dataset!r}")


def cmd_compute(args: argparse.Namespace) -> None:
    clim_period = (args.clim_start, args.clim_end)

    if args.dataset == 'rep':
        ds = io.open_rep(region_selector=args.region)
        using_dataset = 'rep'
    elif args.dataset == 'medrea':
        ds = io.open_medrea(region_selector=args.region)
        using_dataset = 'medrea'
    else:
        raise ValueError(f"Unknown dataset {args.dataset!r}")

    if args.ds_type == 'yearly':
        ds_mhws = compute_mhws.compute_mhw_yearly(ds, using_dataset=using_dataset, clim_period=clim_period)
    elif args.ds_type == 'all_events':
        ds_mhws = compute_mhws.compute_mhw_all_events(ds, using_dataset=using_dataset, clim_period=clim_period)
    else:
        raise ValueError(f"Unknown ds_type {args.ds_type!r}")

    io.save_mhws(
        ds_mhws,
        ds_type=args.ds_type,
        dataset_used=args.dataset,
        region=args.region,
        clim_period=clim_period,
    )


########################################################################################################################
##################################### CLI ###############################################################################
########################################################################################################################


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Balearic MHWs pipeline CLI")
    sub = parser.add_subparsers(dest='stage', required=True)

    download_parser = sub.add_parser('download', help="Ingest raw data into the raw Zarr stores")
    download_parser.add_argument('--dataset', choices=['rep', 'medrea', 'bathy'], required=True)
    download_parser.add_argument('--years', help="Year range as 'start:stop' (stop exclusive) or a single year")
    download_parser.add_argument('--months', help="Month range as 'start:stop' (stop exclusive) or a single month")
    download_parser.set_defaults(func=cmd_download)

    compute_parser = sub.add_parser('compute', help="Compute MHW metrics and save them to the processed Zarr stores")
    compute_parser.add_argument('--dataset', choices=['rep', 'medrea'], required=True)
    compute_parser.add_argument('--region', default='balears')
    compute_parser.add_argument('--ds-type', choices=['yearly', 'all_events'], default='yearly')
    compute_parser.add_argument('--clim-start', type=int, default=1987)
    compute_parser.add_argument('--clim-end', type=int, default=2021)
    compute_parser.add_argument('--task-id', help="Slurm array task id, unused for now - reserved for future chunked runs")
    compute_parser.set_defaults(func=cmd_compute)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main(sys.argv[1:])
