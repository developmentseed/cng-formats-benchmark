"""Tests for the CLI entry point."""

import json

import pytest
from typer.testing import CliRunner

from cng_benchmark import __version__
from cng_benchmark.cli import app

runner = CliRunner()

BENCHMARK_EXAMPLE = "configs/benchmarks/example_cog.yaml"


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_run_validates_config_without_objects():
    result = runner.invoke(app, ["run", BENCHMARK_EXAMPLE])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_run_rejects_missing_config():
    result = runner.invoke(app, ["run", "does/not/exist.yaml"])
    assert result.exit_code == 1


def test_run_emits_object_profile(tmp_path):
    listing = tmp_path / "objects.json"
    entries = [{"name": "a", "size": 10}, {"name": "b", "size": 30}]
    listing.write_text(json.dumps(entries))

    result = runner.invoke(app, ["run", BENCHMARK_EXAMPLE, "--objects", str(listing)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["object_profile"]["count"] == 2
    assert payload["object_profile"]["total_bytes"] == 40
    assert "tier_fit" in payload["object_profile"]


def test_run_accepts_bare_size_list(tmp_path):
    listing = tmp_path / "sizes.json"
    listing.write_text(json.dumps([100, 200, 300]))

    result = runner.invoke(app, ["run", BENCHMARK_EXAMPLE, "--objects", str(listing)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["object_profile"]["count"] == 3


def test_run_rejects_missing_objects_file():
    result = runner.invoke(
        app, ["run", BENCHMARK_EXAMPLE, "--objects", "no/such/file.json"]
    )
    assert result.exit_code == 1
    assert "Cannot profile objects" in result.output


def test_run_rejects_malformed_objects_entry(tmp_path):
    listing = tmp_path / "bad.json"
    listing.write_text(json.dumps([{"name": "a"}]))  # missing "size"

    result = runner.invoke(app, ["run", BENCHMARK_EXAMPLE, "--objects", str(listing)])
    assert result.exit_code == 1
    assert "Cannot profile objects" in result.output


def test_run_rejects_empty_objects_listing(tmp_path):
    listing = tmp_path / "empty.json"
    listing.write_text(json.dumps([]))

    result = runner.invoke(app, ["run", BENCHMARK_EXAMPLE, "--objects", str(listing)])
    assert result.exit_code == 1


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Benchmark cloud-native geospatial formats." in result.stdout


def test_run_profiles_objects_from_uri(tmp_path):
    objdir = tmp_path / "objects"
    objdir.mkdir()
    (objdir / "a.bin").write_bytes(b"x" * 100)
    (objdir / "b.bin").write_bytes(b"y" * 300)

    result = runner.invoke(
        app, ["run", BENCHMARK_EXAMPLE, "--objects-uri", str(objdir)]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["object_profile"]["count"] == 2
    assert payload["object_profile"]["total_bytes"] == 400


def test_run_writes_artifacts_to_output(tmp_path):
    objfile = tmp_path / "o.bin"
    objfile.write_bytes(b"z" * 50)
    out = tmp_path / "results"

    result = runner.invoke(
        app,
        ["run", BENCHMARK_EXAMPLE, "--objects-uri", str(objfile), "--output", str(out)],
    )
    assert result.exit_code == 0
    assert (out / "result.json").exists()
    assert (out / "summary.md").exists()


def test_run_reports_unreachable_object_source(tmp_path):
    result = runner.invoke(
        app, ["run", BENCHMARK_EXAMPLE, "--objects-uri", str(tmp_path / "absent")]
    )
    assert result.exit_code == 1
    assert "Cannot profile objects" in result.output


def test_seed_generates_fixture_to_local_path(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    dest = tmp_path / "fixture.tif"

    result = runner.invoke(app, ["seed", "--dest", str(dest), "--size", "128"])
    assert result.exit_code == 0
    assert dest.exists() and dest.stat().st_size > 0
