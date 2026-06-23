"""Tests for the result schema (serialisability and round-tripping)."""

from datetime import UTC, datetime

from cng_benchmark.models import (
    BenchmarkRun,
    CogLayout,
    GeoZarrLayout,
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
    assert run.object_layouts == []


def test_object_layouts_union_round_trips_subclass_fields():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        dataset_id="example-raster",
        format_id="geozarr",
        object_layouts=[
            CogLayout(
                name="FRE_B4",
                size_bytes=100,
                is_tiled=True,
                block_height=512,
                block_width=512,
                overview_decimations=[2, 4],
                internal_tiles=16,
            ),
            GeoZarrLayout(
                name="FRE_B4",
                size_bytes=200,
                chunk_shape=[512, 512],
                shard_shape=[1024, 1024],
                chunks_per_shard=4,
                codec="zstd",
                multiscale_levels=1,
                shard_count=4,
            ),
        ],
    )
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded == run
    # The discriminator preserved each subclass's distinct fields.
    cog, geozarr = reloaded.object_layouts
    assert cog.kind == "cog" and cog.internal_tiles == 16
    assert geozarr.kind == "geozarr" and geozarr.chunks_per_shard == 4
