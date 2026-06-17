"""Tests for the orchestration runner (independent of the CLI)."""

import pytest

from cng_benchmark import __version__
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.runner import run_benchmark, tier_policy_from_config

BENCHMARK_EXAMPLE = "configs/benchmarks/example_cog.yaml"


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
