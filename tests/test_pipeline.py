"""Tests for the pipeline CLI's pure logic - no data, no network."""

import pytest

from pipelines.mhws_pipeline import _parse_range, build_parser


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2020:2023", [2020, 2021, 2022]),  # stop is exclusive
        ("2020", [2020]),                   # a single value is its own range
        ("1:13", list(range(1, 13))),       # the months range used by the DAGs
    ],
)
def test_parse_range(value, expected):
    assert list(_parse_range(value)) == expected


def test_compute_defaults_to_the_configured_climatology():
    # The climatology period must never drift per run - the CLI defaults to config's value.
    from balearic_mhws import config

    args = build_parser().parse_args(["compute", "--dataset", "rep"])

    assert (args.clim_start, args.clim_end) == config.CLIMATOLOGY_PERIOD


def test_parser_rejects_an_unknown_dataset():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["compute", "--dataset", "not-a-dataset"])
