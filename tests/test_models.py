"""Tests for the result schema (serialisability and round-tripping)."""

from datetime import UTC, datetime

from cng_benchmark.models import (
    BenchmarkRun,
    HistogramBin,
    MetricResult,
    ObjectSizeProfile,
)


def _profile() -> ObjectSizeProfile:
    return ObjectSizeProfile(
        count=3,
        total_bytes=60,
        mean=20.0,
        median=20.0,
        p50=20.0,
        p90=28.0,
        p95=29.0,
        p99=29.8,
        min_bytes=10,
        max_bytes=30,
        histogram=[HistogramBin(lower=8, upper=32, count=3)],
        tier_fit=["warm"],
        highest_tier="warm",
    )


def test_benchmark_run_json_round_trips():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        tool_versions={"cng_benchmark": "0.0.0"},
        dataset_id="example-raster",
        format_id="cog",
        params={"grouping_lever": "COG internal tiling"},
        object_profile=_profile(),
        metrics=[MetricResult(name="object_count", value=3)],
    )
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded == run
    assert reloaded.object_profile is not None
    assert reloaded.object_profile.highest_tier == "warm"


def test_benchmark_run_defaults_are_empty():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        dataset_id="d",
        format_id="cog",
    )
    assert run.tool_versions == {}
    assert run.params == {}
    assert run.metrics == []
    assert run.object_profile is None
