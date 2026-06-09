"""Tests for the CLI entry point."""

from typer.testing import CliRunner

from cng_benchmark import __version__
from cng_benchmark.cli import app

runner = CliRunner()


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_run_is_a_stub_but_exits_cleanly():
    result = runner.invoke(app, ["run", "configs/example.yaml"])
    assert result.exit_code == 0
    assert "configs/example.yaml" in result.stdout


def test_no_args_shows_help():
    # no_args_is_help exits with the usage code (2) after printing help.
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Benchmark cloud-native geospatial formats." in result.stdout
