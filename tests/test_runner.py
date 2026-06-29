"""Tests for the orchestration runner (independent of the CLI)."""

import pytest

from cng_benchmark import __version__
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.runner import (
    _safe_display_metrics,
    _safe_read_metrics,
    run_benchmark,
    run_conversion_benchmark,
    tier_policy_from_config,
)

BENCHMARK_EXAMPLE = "configs/benchmarks/example_cog.yaml"
SYNTHETIC = "configs/benchmarks/synthetic_cog.yaml"


def test_run_benchmark_populates_run_context():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    sizes = [10, 20, 30, 40]
    run = run_benchmark(cfg, sizes)

    assert run.dataset_id == cfg.dataset
    assert run.format_id == cfg.formats[0]
    assert run.tool_versions["cng_benchmark"] == __version__
    assert "grouping_lever" in run.params

    expected = profile_object_sizes(sizes, tier_policy_from_config(cfg.tiers))
    assert run.object_profile == expected
    assert {m.name for m in run.metrics} == {"object_count", "total_bytes"}


def test_run_benchmark_selects_named_format():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    run = run_benchmark(cfg, [1, 2, 3], format_id="geozarr")
    assert run.format_id == "geozarr"
    assert "Zarr v3" in run.params["grouping_lever"]


def test_run_benchmark_unknown_format_raises():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    with pytest.raises(KeyError):
        run_benchmark(cfg, [1, 2, 3], format_id="nonesuch")


def test_run_conversion_benchmark_local_publishes_and_collects(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=256, blocksize=256))
    output = tmp_path / "out"

    # Local end-to-end without services: write/object_size/read (no display).
    cfg = load_benchmark_config(SYNTHETIC).model_copy(
        update={"metrics": ["write", "object_size", "read"]}
    )
    run = run_conversion_benchmark(cfg, str(source), str(output))

    assert run.format_id == "cog"
    assert run.object_profile.count == 1
    names = {m.name for m in run.metrics}
    assert {"write_elapsed", "object_count", "read_window_count"} <= names
    # The produced object is always published under the output location.
    assert (output / "cog" / "cog.tif").exists()


def test_run_conversion_benchmark_display_skips_without_endpoint(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=128, blocksize=128))
    cfg = load_benchmark_config(SYNTHETIC).model_copy(update={"metrics": ["display"]})
    # A missing endpoint is caught and surfaced as a skipped metric — the run
    # completes rather than aborting, even in the single-object path.
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))
    names = {m.name for m in run.metrics}
    assert "display_skipped" in names
    skipped = next(m for m in run.metrics if m.name == "display_skipped")
    assert "TiTiler endpoint" in skipped.detail["error"]


def test_safe_read_metrics_returns_skipped_on_failure(monkeypatch):
    import cng_benchmark.runner as _runner

    def _raise(*a, **k):
        raise RuntimeError("timed out")

    monkeypatch.setattr(_runner, "_measure_object_read", _raise)
    result = _safe_read_metrics(adapter=None, object_uri="s3://b/k")
    assert len(result) == 1
    assert result[0].name == "read_skipped"
    assert "timed out" in result[0].detail["error"]


def test_safe_display_metrics_returns_skipped_on_failure(monkeypatch):
    import cng_benchmark.runner as _runner

    def _raise(*a, **k):
        raise RuntimeError("HTTP 504")

    monkeypatch.setattr(_runner, "_measure_display_object", _raise)
    metrics, artifacts = _safe_display_metrics(
        config=None,
        adapter=None,
        local_target="",
        object_uri="",
        artifact_dir="",
        titiler_endpoint=None,
    )
    assert len(metrics) == 1
    assert metrics[0].name == "display_skipped"
    assert "HTTP 504" in metrics[0].detail["error"]
    assert artifacts == []
