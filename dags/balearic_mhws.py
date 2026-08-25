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
        "clim_start": 1987,
        "clim_end": 2021,
        # Optional 'start:stop' ranges (stop exclusive). Empty means "whatever ingest_* defaults to".
        "years": "",
        "months": "",
    },
)
def balearic_mhws():
    """
    Downloads REP or MEDREA data into its raw Zarr store, computes MHW metrics from it, and
    validates the resulting Zarr store. All three stages call into `pipelines.mhws_pipeline`,
    the same CLI the Slurm sbatch scripts use, so the logic is defined in exactly one place.

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
        clim_start: int = 1987,
        clim_end: int = 2021,
    ):
        import os

        # DAG params come through Jinja templating as strings even when their default is an int.
        run_args = {
            "dataset": dataset,
            "region": region,
            "ds_type": ds_type,
            "clim_start": int(clim_start),
            "clim_end": int(clim_end),
        }

        slurm_host = os.environ.get("SLURM_SSH_HOST")

        if slurm_host:
            # A real cluster is configured: submit the sbatch job and block until it finishes,
            # rather than computing in-process on the Celery worker.
            import subprocess

            remote_dir = os.environ["SLURM_REMOTE_PROJECT_DIR"]
            ssh_user = os.environ.get("SLURM_SSH_USER", "")
            target = f"{ssh_user}@{slurm_host}" if ssh_user else slurm_host

            export_vars = ",".join(f"{key.upper()}={value}" for key, value in run_args.items())

            remote_cmd = (
                f"cd {remote_dir} && sbatch --wait --export=ALL,{export_vars} hpc/slurm/compute_mhws.sbatch"
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

    dataset = ingest_data(
        dataset="{{ params.dataset }}",
        years="{{ params.years }}",
        months="{{ params.months }}",
    )

    result = compute_mhws_task(
        dataset=dataset,
        region="{{ params.region }}",
        ds_type="{{ params.ds_type }}",
        clim_start="{{ params.clim_start }}",
        clim_end="{{ params.clim_end }}",
    )

    validate_result(result)


balearic_mhws()
