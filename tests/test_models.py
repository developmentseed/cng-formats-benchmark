"""Tests for the result schema (serialisability and round-tripping)."""

from datetime import UTC, datetime

from cng_benchmark.models import (
    Artifact,
    BenchmarkRun,
    CogLayout,
    ConditionReplicate,
    ConditionResult,
    FlatGeobufLayout,
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
    assert run.artifacts == []
    assert run.conditions == []


def test_artifacts_round_trip():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        dataset_id="example-sentinel2-l2a",
        format_id="cog",
        artifacts=[
            Artifact(
                kind="viewer_vrt",
                name="natural",
                uri="s3://bucket/runs/r1/run-natural.vrt",
                media_type="application/xml",
                detail={"rescale": [0.0, 3000.0], "titiler_url": "/cog/viewer?url=…"},
            ),
            Artifact(
                kind="octree_lod",
                name="copc_octree_lod",
                detail={"skipped_reason": "matplotlib missing"},
            ),
        ],
    )
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded == run
    produced, skipped = reloaded.artifacts
    assert produced.uri == "s3://bucket/runs/r1/run-natural.vrt"
    assert produced.detail["rescale"] == [0.0, 3000.0]
    assert skipped.uri is None
    assert skipped.detail["skipped_reason"] == "matplotlib missing"


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


def test_flatgeobuf_layout_null_geometry_fields_default_to_an_ordinary_run():
    ly = FlatGeobufLayout(
        name="LakeSP_048",
        size_bytes=100,
        geometry_type="Polygon",
        num_features=10,
        has_spatial_index=True,
        index_node_size=16,
    )
    assert ly.features_dropped == 0
    assert ly.content_subset is False
    assert ly.features_sentinel == 0
    assert ly.geometry_fabricated is False


def test_flatgeobuf_layout_null_geometry_fields_round_trip():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        dataset_id="swot-lakesp-prior",
        format_id="flatgeobuf",
        object_layouts=[
            FlatGeobufLayout(
                name="LakeSP_048",
                size_bytes=100,
                geometry_type="Polygon",
                num_features=150,
                has_spatial_index=True,
                index_node_size=16,
                features_dropped=50,
                content_subset=True,
            )
        ],
    )
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded == run
    (ly,) = reloaded.object_layouts
    assert ly.features_dropped == 50
    assert ly.content_subset is True


def test_condition_result_round_trips():
    run = BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        dataset_id="example-raster",
        format_id="cog",
        conditions=[
            ConditionResult(
                phase="read",
                cache="cold",
                concurrency=1,
                replicates=[
                    ConditionReplicate(
                        index=0,
                        metrics=[
                            MetricResult(name="read_latency_mean", value=0.01, unit="s")
                        ],
                    ),
                    ConditionReplicate(
                        index=1,
                        metrics=[
                            MetricResult(name="read_latency_mean", value=0.03, unit="s")
                        ],
                    ),
                ],
                aggregate=[
                    MetricResult(
                        name="read_latency_mean",
                        value=0.02,
                        unit="s",
                        detail={
                            "replicate_values": [0.01, 0.03],
                            "median": 0.02,
                            "stdev": 0.01,
                        },
                    )
                ],
            )
        ],
    )
    reloaded = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert reloaded == run
    (condition,) = reloaded.conditions
    assert condition.cache == "cold"
    assert condition.concurrency == 1
    assert len(condition.replicates) == 2
    assert condition.aggregate[0].detail["median"] == 0.02
