"""Tests for the Slurm submission logic - no cluster and no SSH needed, the calls are faked."""

import pytest

from pipelines.slurm import (
    build_poll_command,
    build_submit_command,
    job_log_path,
    parse_job_id,
    parse_job_state,
    parse_poll_output,
    run_compute_job,
    ssh_target,
)

RUN_ARGS = {"dataset": "rep", "region": "balears", "ds_type": "yearly", "clim_start": 1987, "clim_end": 2021}


def _poll_output(new_lines, total_lines, queued):
    """Builds what the poll command prints on the remote host."""

    return (
        f"{new_lines}---SLURM-POLL-SEP-1---\n"
        f"{total_lines}\n"
        f"---SLURM-POLL-SEP-2---\n"
        f"{'123 debug mhw-comp user R 0:10 1 node' if queued else ''}"
    )


def test_ssh_target():
    assert ssh_target("cluster", "user") == "user@cluster"
    assert ssh_target("cluster") == "cluster"


def test_build_submit_command_quotes_its_values():
    # A partition holding shell metacharacters must not be able to inject a second command.
    command = build_submit_command("/project", RUN_ARGS, partition="debug; rm -rf /")

    assert "'debug; rm -rf /'" in command
    assert "DATASET=rep" in command
    # The remote project dir is forwarded to the job, as a non-interactive shell may not export it.
    assert "SLURM_REMOTE_PROJECT_DIR=/project" in command


def test_build_submit_command_without_a_partition():
    assert "--partition" not in build_submit_command("/project", RUN_ARGS)


def test_parse_job_id():
    assert parse_job_id("Submitted batch job 4242\n") == "4242"

    with pytest.raises(RuntimeError):
        parse_job_id("sbatch: error: invalid partition")


def test_job_log_path_matches_the_sbatch_output_pattern():
    assert job_log_path("/project", "12") == "/project/logs/mhw-compute-12_4294967294.out"


def test_parse_poll_output():
    new_output, total_lines, still_queued = parse_poll_output(_poll_output("a\nb\n", 7, queued=True))

    assert new_output == "a\nb\n"
    assert total_lines == 7
    assert still_queued is True

    # A finished job leaves the queue, and an empty log reports a length of zero.
    _, total_lines, still_queued = parse_poll_output(_poll_output("", "", queued=False))

    assert (total_lines, still_queued) == (0, False)


def test_parse_job_state():
    assert parse_job_state("JobId=1 JobName=x\n   JobState=COMPLETED Reason=None\n") == "COMPLETED"
    assert parse_job_state("no state here") is None


def test_run_compute_job_streams_the_log_and_never_rereads_a_line(capsys):
    # The log grows between polls, and one poll finds nothing new. Line offsets must come from the
    # log's own length, otherwise every line after the empty poll is skipped.
    polls = [
        _poll_output("first\n", 1, queued=True),
        _poll_output("", 1, queued=True),           # nothing new this time
        _poll_output("second\nthird\n", 3, queued=True),
        _poll_output("", 3, queued=False),          # job left the queue
        _poll_output("last\n", 4, queued=False),    # final catch-up poll
    ]
    asked = []

    def fake_ssh(target, command):
        if command.startswith("cd "):
            return "Submitted batch job 77\n"
        if command.startswith("scontrol"):
            return "JobState=COMPLETED"
        asked.append(command)
        return polls.pop(0)

    job_id = run_compute_job("u@h", "/project", RUN_ARGS, run_ssh=fake_ssh, sleep=lambda _: None)

    assert job_id == "77"
    printed = capsys.readouterr().out
    for line in ("first", "second", "third", "last"):
        assert line in printed

    # Each poll asks for the lines after the length the previous poll reported.
    assert [c.split()[1] for c in asked] == ["-n", "-n", "-n", "-n", "-n"]
    assert "tail -n +1 " in asked[0]
    assert "tail -n +2 " in asked[1]
    assert "tail -n +2 " in asked[2]
    assert "tail -n +4 " in asked[3]


def test_run_compute_job_raises_when_the_job_did_not_complete():
    def fake_ssh(target, command):
        if command.startswith("cd "):
            return "Submitted batch job 78\n"
        if command.startswith("scontrol"):
            return "JobState=FAILED Reason=NonZeroExitCode"
        return _poll_output("", 0, queued=False)

    with pytest.raises(RuntimeError, match="did not complete"):
        run_compute_job("u@h", "/project", RUN_ARGS, run_ssh=fake_ssh, sleep=lambda _: None)
