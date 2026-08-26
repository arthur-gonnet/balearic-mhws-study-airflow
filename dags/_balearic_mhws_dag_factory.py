import pendulum

from airflow.sdk import dag, task


def build_dag(dataset: str):
    """
    Builds the ingest -> compute -> validate -> plot_diagnostic DAG for one dataset ('rep' or
    'medrea'). One DAG per dataset (rather than a single DAG parameterized by `dataset`) since
    REP and MEDREA come from different Copernicus products with different update cadences -
    `schedule` is fixed per DAG, not per-trigger-conf, so a shared DAG couldn't give each its own
    schedule. Not itself a DAG file - imported by balearic_mhws_rep.py/balearic_mhws_medrea.py,
    which are the ones Airflow actually discovers.
    """

    @dag(
        dag_id=f"balearic_mhws_{dataset}",
        schedule=None,
        start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
        catchup=False,
        tags=["oceanography", "mhw", "balearic", dataset],
        doc_md=f"""
        Downloads {dataset.upper()} data into its raw Zarr store, computes MHW metrics from it,
        validates the resulting Zarr store, and saves a diagnostic map as a quick visual sanity
        check. The ingest/compute/validate stages call into `pipelines.mhws_pipeline`, the same
        CLI the Slurm sbatch scripts use, so that logic is defined in exactly one place.

        The climatology period is always config.CLIMATOLOGY_PERIOD - not a trigger param - so
        every run's results stay comparable and never end up self-referential (climatology ==
        analysis year).

        Trigger with e.g. {{"years": "2020:2023"}} to control what gets ingested/computed.
        """,
        params={
            "region": "balears",
            "ds_type": "yearly",
            # Optional 'start:stop' ranges (stop exclusive). Empty means "whatever ingest_* defaults to".
            "years": "",
            "months": "",
        },
    )
    def balearic_mhws_dag():
        @task
        def ingest_data(years: str = "", months: str = ""):
            from balearic_mhws.data import download
            from pipelines.mhws_pipeline import _parse_range

            kwargs = {}
            if years:
                kwargs["years"] = _parse_range(years)
            if months:
                kwargs["months"] = _parse_range(months)

            if dataset == "rep":
                download.ingest_rep(**kwargs)
            else:
                download.ingest_medrea(**kwargs)

        @task
        def compute_mhws_task(region: str = "balears", ds_type: str = "yearly"):
            import os

            from balearic_mhws import config

            # DAG params come through Jinja templating as strings even when their default is an int.
            run_args = {
                "dataset": dataset,
                "region": region,
                "ds_type": ds_type,
                "clim_start": config.CLIMATOLOGY_PERIOD[0],
                "clim_end": config.CLIMATOLOGY_PERIOD[1],
            }

            slurm_host = os.environ.get("SLURM_SSH_HOST")

            if slurm_host:
                # A real cluster is configured: submit the job there instead of computing here.
                from pipelines.slurm import run_compute_job, ssh_target

                remote_dir = os.environ.get("SLURM_REMOTE_PROJECT_DIR", "")
                if not remote_dir:
                    raise ValueError(
                        "SLURM_SSH_HOST is set but SLURM_REMOTE_PROJECT_DIR is empty - "
                        "set it in .env to the project's path on the remote cluster."
                    )

                run_compute_job(
                    target=ssh_target(slurm_host, os.environ.get("SLURM_SSH_USER", "")),
                    remote_dir=remote_dir,
                    run_args=run_args,
                    partition=os.environ.get("SLURM_PARTITION", ""),
                )

            else:
                # No cluster configured (e.g. local Docker Compose dev stack): run in-process.
                import argparse

                from pipelines.mhws_pipeline import cmd_compute

                cmd_compute(argparse.Namespace(**run_args, task_id=None))

            return run_args

        @task
        def validate_result(run_args: dict):
            from balearic_mhws.data import io

            ds_mhws = io.load_mhws(
                ds_type=run_args["ds_type"],
                dataset_used=run_args["dataset"],
                region=run_args["region"],
                clim_period=(run_args["clim_start"], run_args["clim_end"]),
            )

            if "year" not in ds_mhws.coords or ds_mhws.sizes.get("year", 0) == 0:
                raise ValueError("MHWs dataset has no 'year' coordinate - computation likely failed.")

            # Only 'count' is checked, not every variable: duration/intensity/severity/rate stats are
            # legitimately NaN dataset-wide whenever zero MHW events were detected anywhere (e.g. a
            # short climatology period), which is a valid result, not a computation failure. 'count'
            # being entirely NaN, on the other hand, means every grid cell was land/no-data - a real
            # sign the wrong region/dataset was computed.
            if bool(ds_mhws["count"].isnull().all()):
                raise ValueError("MHWs dataset's 'count' variable is entirely NaN - no valid grid cells were computed.")

            print(f"Validated {run_args['dataset']} MHWs dataset: {dict(ds_mhws.sizes)}")

        @task
        def plot_diagnostic(run_args: dict):
            # Only 'yearly' output has the (lat, lon, year) shape a single map can show; 'all_events'
            # is event/time-indexed instead, no natural equivalent for a quick sanity-check plot.
            if run_args["ds_type"] != "yearly":
                print(f"Skipping diagnostic plot for ds_type={run_args['ds_type']!r} (not 'yearly').")
                return None

            import matplotlib
            matplotlib.use("Agg")

            from balearic_mhws import config
            from balearic_mhws.data import io
            from balearic_mhws.plotting.plot import plot_map, mhws_stats_cmaps

            stat = "total_days"

            ds_mhws = io.load_mhws(
                ds_type=run_args["ds_type"],
                dataset_used=run_args["dataset"],
                region=run_args["region"],
                clim_period=(run_args["clim_start"], run_args["clim_end"]),
            )
            da = ds_mhws[stat].isel(year=-1)
            year = int(da.year.item())
            if "depth" in da.dims:
                da = da.sel(depth=0, method="nearest")

            out_dir = config.PRODUCTS_DIR / "diagnostics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_args['dataset']}_{stat}_{year}.png"

            plot_map(
                lon=da.lon,
                lat=da.lat,
                data=da,
                title=f"{run_args['dataset']} {stat} ({year})",
                cbar_unit=f"[{config.mhws_stats_units[stat]}]",
                cmap=mhws_stats_cmaps[stat],
                save_plot=True,
                save_path=str(out_path),
                transparent=False,
            )

            print(f"Saved diagnostic plot to {out_path}")
            return str(out_path)

        ingest = ingest_data(
            years="{{ params.years }}",
            months="{{ params.months }}",
        )

        result = compute_mhws_task(
            region="{{ params.region }}",
            ds_type="{{ params.ds_type }}",
        )
        ingest >> result

        validation = validate_result(result)
        diagnostic = plot_diagnostic(result)
        validation >> diagnostic

    return balearic_mhws_dag()
