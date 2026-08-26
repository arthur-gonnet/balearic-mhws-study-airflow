"""
Submitting the compute stage to a Slurm cluster over SSH, and following it until it finishes.

Kept out of the DAG file so that it can be tested without Airflow installed. The DAG calls
`run_compute_job`, which submits the job, prints the job's own output into the caller's log as it
is written, and fails if the job does not complete.

The job is polled rather than submitted with `sbatch --wait`, as `--wait` blocks silently over SSH
and streams nothing back, leaving a run of several hours with no visible progress.
"""

import re
import shlex
import subprocess
import time
from typing import Callable, Dict, Optional, Tuple

# Printed by the poll command to separate its three parts in a single SSH round-trip.
_SEP_OUTPUT = "---SLURM-POLL-SEP-1---"
_SEP_QUEUE = "---SLURM-POLL-SEP-2---"


def _run_ssh(target: str, command: str) -> str:
    """Runs one command on the remote host and returns its stdout."""

    result = subprocess.run(["ssh", target, command], check=True, capture_output=True, text=True)
    return result.stdout


def ssh_target(host: str, user: str = "") -> str:
    """Builds the `user@host` argument for ssh, or just the host when no user is given."""

    return f"{user}@{host}" if user else host


def build_submit_command(remote_dir: str, run_args: Dict, partition: str = "") -> str:
    """
    Builds the sbatch command submitting the compute job.

    Every interpolated value is quoted, as the command is handed to the remote shell as a single
    string, so a DAG parameter holding shell metacharacters could otherwise inject commands.

    `SLURM_REMOTE_PROJECT_DIR` is forwarded explicitly rather than relied upon on the remote side,
    as `ssh host "command"` runs a non-interactive shell, which does not reliably source
    `~/.bashrc`.
    """

    export_vars = ",".join(
        f"{key.upper()}={shlex.quote(str(value))}"
        for key, value in {**run_args, "slurm_remote_project_dir": remote_dir}.items()
    )

    partition_flag = f"--partition={shlex.quote(partition)} " if partition else ""

    return (
        f"cd {shlex.quote(remote_dir)} && sbatch {partition_flag}"
        f"--export=ALL,{export_vars} hpc/slurm/compute_mhws.sbatch"
    )


def parse_job_id(sbatch_output: str) -> str:
    """Extracts the job id from sbatch's output."""

    match = re.search(r"Submitted batch job (\d+)", sbatch_output)

    if not match:
        raise RuntimeError(f"Could not parse a Slurm job id from sbatch's output: {sbatch_output!r}")

    return match.group(1)


def job_log_path(remote_dir: str, job_id: str) -> str:
    """
    Builds the path of the job's output file.

    Matches `--output=logs/mhw-compute-%A_%a.out` in `hpc/slurm/compute_mhws.sbatch`. The array
    task id is always this sentinel value, as no `--array` range is set.
    """

    return f"{remote_dir}/logs/mhw-compute-{job_id}_4294967294.out"


def build_poll_command(log_path: str, job_id: str, lines_read: int) -> str:
    """Builds the command reading the new log lines, the log length, and the job's queue state."""

    return (
        f"tail -n +{lines_read + 1} {shlex.quote(log_path)} 2>/dev/null; "
        f"printf -- '{_SEP_OUTPUT}\\n'; "
        f"wc -l < {shlex.quote(log_path)} 2>/dev/null; "
        f"printf -- '{_SEP_QUEUE}\\n'; "
        f"squeue -h -j {shlex.quote(job_id)}"
    )


def parse_poll_output(stdout: str) -> Tuple[str, int, bool]:
    """
    Splits the poll command's output into the new log lines, the log length and whether the job is
    still queued.

    The log length comes from `wc -l` rather than from counting the lines just printed. Counting
    them drifts as soon as a poll finds nothing new, as the separators printed by the command
    itself would be counted too, and every later line would then be skipped.
    """

    new_output, _, rest = stdout.partition(_SEP_OUTPUT + "\n")
    total_lines, _, queue_state = rest.partition(_SEP_QUEUE + "\n")

    return new_output, int(total_lines.strip() or 0), bool(queue_state.strip())


def parse_job_state(scontrol_output: str) -> Optional[str]:
    """Extracts the job state from `scontrol show job`'s output."""

    match = re.search(r"JobState=(\S+)", scontrol_output)

    return match.group(1) if match else None


def run_compute_job(
        target: str,
        remote_dir: str,
        run_args: Dict,
        partition: str = "",
        poll_interval: float = 10,
        run_ssh: Callable[[str, str], str] = _run_ssh,
        sleep: Callable[[float], None] = time.sleep,
) -> str:
    """
    Submits the compute job and follows it until it finishes, printing its output as it is written.

    Raises if the job cannot be submitted, or if it leaves the queue without completing.

    Parameters
    ----------
    target : str
        The `user@host` to submit to.

    remote_dir : str
        Path of the project on the cluster.

    run_args : dict
        Arguments forwarded to the job as environment variables.

    partition : str, optional
        Partition to submit to.

    poll_interval : float, default=10
        Seconds between two polls.

    run_ssh, sleep : callable, optional
        Injected for testing.

    Returns
    ----------
    job_id : str
        The id of the submitted job.
    """

    submit_cmd = build_submit_command(remote_dir, run_args, partition)
    print("Submitting Slurm job over SSH:", submit_cmd)

    sbatch_output = run_ssh(target, submit_cmd)
    print(sbatch_output.strip())

    job_id = parse_job_id(sbatch_output)
    log_path = job_log_path(remote_dir, job_id)

    def poll(lines_read: int) -> Tuple[int, bool]:
        stdout = run_ssh(target, build_poll_command(log_path, job_id, lines_read))
        new_output, total_lines, still_queued = parse_poll_output(stdout)

        if new_output:
            print(new_output, end="")

        return total_lines, still_queued

    lines_read, still_queued = 0, True
    while still_queued:
        sleep(poll_interval)
        lines_read, still_queued = poll(lines_read)

    # Catches whatever was written between the last poll and the job actually exiting.
    poll(lines_read)

    # Leaving the queue only means the job is no longer running, not that it succeeded. `scontrol`
    # reads the live state from slurmctld, where `sacct` needs the accounting database, which a
    # cluster does not necessarily run.
    state = parse_job_state(run_ssh(target, f"scontrol show job {shlex.quote(job_id)}"))

    if state != "COMPLETED":
        raise RuntimeError(f"Slurm job {job_id} did not complete successfully (state: {state}).")

    return job_id
