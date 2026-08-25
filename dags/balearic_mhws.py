import pendulum

from airflow.sdk import dag, task


@dag(
    dag_id="balearic_mhws",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["oceanography", "mhw", "balearic"],
    params={
        "dataset": "rep",
        "region": "balears",
        "ds_type": "yearly",
        # Optional 'start:stop' ranges (stop exclusive). Empty means "whatever ingest_* defaults to".
        "years": "",
        "months": "",
    },
)
def balearic_mhws():
    """
    Downloads REP or MEDREA data into its raw Zarr store, computes MHW metrics from it, validates
    the resulting Zarr store, and saves a diagnostic map as a quick visual sanity check. The
    ingest/compute/validate stages call into `pipelines.mhws_pipeline`, the same CLI the Slurm
    sbatch scripts use, so that logic is defined in exactly one place.

    The climatology period is always config.CLIMATOLOGY_PERIOD - not a trigger param - so every
    run's results stay comparable and never end up self-referential (climatology == analysis year).

    Trigger with e.g. {"dataset": "rep", "years": "2020:2023"} to control what gets ingested/computed.
    """

    @task
    def ingest_data(dataset: str = "rep", years: str = "", months: str = ""):
        from balearic_mhws.data import download
        from pipelines.mhws_pipeline import _parse_range

        kwargs = {}
        if years:
            kwargs["years"] = _parse_range(years)
        if months:
            kwargs["months"] = _parse_range(months)

        if dataset == "rep":
            download.ingest_rep(**kwargs)
        elif dataset == "medrea":
            download.ingest_medrea(**kwargs)
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

        return dataset

    @task
    def compute_mhws_task(
        dataset: str,
        region: str = "balears",
        ds_type: str = "yearly",
    ):
        import os
        import shlex

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
            # A real cluster is configured: submit the sbatch job and block until it finishes,
            # rather than computing in-process on the Celery worker.
            import subprocess

            remote_dir = os.environ.get("SLURM_REMOTE_PROJECT_DIR", "")
            if not remote_dir:
                raise ValueError(
                    "SLURM_SSH_HOST is set but SLURM_REMOTE_PROJECT_DIR is empty - "
                    "set it in .env to the project's path on the remote cluster."
                )

            ssh_user = os.environ.get("SLURM_SSH_USER", "")
            target = f"{ssh_user}@{slurm_host}" if ssh_user else slurm_host

            # remote_cmd ends up as one string that `ssh` hands to the remote shell to parse -
            # shlex.quote() every interpolated value so a DAG param or env var containing shell
            # metacharacters (';', backticks, '$()', ...) can't inject commands on the remote host.

            # Forward SLURM_REMOTE_PROJECT_DIR explicitly rather than relying on it already being
            # set in the remote environment: `ssh host "command"` runs a non-interactive shell,
            # which doesn't reliably source ~/.bashrc, so --export=ALL alone can't be trusted to
            # carry it through even if it's exported there.
            export_vars = ",".join(
                f"{key.upper()}={shlex.quote(str(value))}"
                for key, value in {**run_args, "slurm_remote_project_dir": remote_dir}.items()
            )

            partition = os.environ.get("SLURM_PARTITION", "")
            partition_flag = f"--partition={shlex.quote(partition)} " if partition else ""

            remote_cmd = (
                f"cd {shlex.quote(remote_dir)} && sbatch --wait {partition_flag}"
                f"--export=ALL,{export_vars} hpc/slurm/compute_mhws.sbatch"
            )

            print("Submitting Slurm job over SSH:", remote_cmd)
            subprocess.run(["ssh", target, remote_cmd], check=True)

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

    dataset = ingest_data(
        dataset="{{ params.dataset }}",
        years="{{ params.years }}",
        months="{{ params.months }}",
    )

    result = compute_mhws_task(
        dataset=dataset,
        region="{{ params.region }}",
        ds_type="{{ params.ds_type }}",
    )

    validation = validate_result(result)
    diagnostic = plot_diagnostic(result)
    validation >> diagnostic


balearic_mhws()
