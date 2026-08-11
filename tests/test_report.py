"""Tests for result-artifact rendering and persistence."""

import json
from datetime import UTC, datetime

from cng_benchmark.config import load_benchmark_config, tier_policy_from_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.models import (
    BenchmarkRun,
    CogLayout,
    ConditionReplicate,
    ConditionResult,
    CopcLayout,
    FlatGeobufLayout,
    GeoParquetLayout,
    GeoZarrLayout,
    MetricResult,
)
from cng_benchmark.report import (
    RESULT_FILENAME,
    SUMMARY_FILENAME,
    render_markdown_summary,
    write_artifacts,
)

BENCHMARK_EXAMPLE = "configs/benchmarks/synthetic_cog.yaml"


def _sample_run():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    profile = profile_object_sizes([10, 20, 30], tier_policy_from_config(cfg.tiers))
    return BenchmarkRun(
        timestamp=datetime(2026, 6, 17, tzinfo=UTC),
        tool_versions={"cng_benchmark": "0.0.0"},
        dataset_id=cfg.dataset,
        format_id="cog",
        object_profile=profile,
        metrics=[MetricResult(name="object_count", value=3)],
    )


def test_render_markdown_summary_includes_key_facts():
    md = render_markdown_summary(_sample_run())
    assert "synthetic-cog" in md
    assert "cog" in md
    assert "Object-size profile" in md
    assert "Tier fit" in md
    assert "| object_count |" in md


def test_summary_renders_cog_tiling_layout():
    run = _sample_run()
    run.object_layouts = [
        CogLayout(
            name="FRE_B4",
            size_bytes=100,
            is_tiled=True,
            block_height=512,
            block_width=512,
            overview_decimations=[2, 4],
            internal_tiles=16,
        )
    ]
    md = render_markdown_summary(run)
    assert "## Tiling layout" in md
    assert "Internally tiled:" in md
    assert "512×512" in md


def test_summary_renders_geozarr_chunk_shard_layout():
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="FRE_B4",
            size_bytes=200,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=1,
            shard_count=4,
        )
    ]
    md = render_markdown_summary(run)
    assert "## Chunk/shard layout" in md
    assert "Shard objects:" in md
    assert "| 512×512 | 1024×1024 | 4 | zstd | 1 | 4 |" in md
    # An ordinary (non-batched) run gets no "Bundled" coverage line and an
    # empty Grid column.
    assert "Bundled" not in md
    assert md.count("| — |") >= 1
    # No COG-only table for a GeoZarr run.
    assert "## Tiling layout" not in md


def test_summary_flags_bundled_geozarr_components_sharing_a_grid():
    # #102: components sharing a grid group share its coordinate arrays and
    # pyramid metadata — the report has to say so, not just list arrays.
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="wse",
            size_bytes=200,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=0,
            shard_count=4,
            grid_group="grid0",
        ),
        GeoZarrLayout(
            name="sig0",
            size_bytes=180,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=0,
            shard_count=4,
            grid_group="grid0",
        ),
    ]
    md = render_markdown_summary(run)
    assert "Bundled:** 2 array(s) share 1 grid group(s)" in md
    assert "| wse | 512×512 | 1024×1024 | 4 | zstd | 0 | 4 | — | — | grid0 |" in md
    assert "| sig0 | 512×512 | 1024×1024 | 4 | zstd | 0 | 4 | — | — | grid0 |" in md


def test_summary_flags_bundled_cog_band_names():
    # #102: a multi-band COG bundling several components must still say which
    # band is which component.
    run = _sample_run()
    run.object_layouts = [
        CogLayout(
            name="cog",
            size_bytes=100,
            is_tiled=True,
            block_height=512,
            block_width=512,
            internal_tiles=4,
            band_names=["wse", "sig0", "area"],
        )
    ]
    md = render_markdown_summary(run)
    assert "Bundled:** 1 file(s) hold more than one component" in md
    assert "wse, sig0, area" in md


def test_summary_renders_geoparquet_row_group_layout():
    run = _sample_run()
    run.format_id = "geoparquet"
    run.object_layouts = [
        GeoParquetLayout(
            name="LakeSP_048",
            size_bytes=300,
            geometry_column="geometry",
            num_rows=200,
            num_row_groups=4,
            row_group_rows=50,
            has_bbox_covering=True,
        )
    ]
    md = render_markdown_summary(run)
    assert "## Row-group layout" in md
    assert "Bbox covering:" in md
    assert "| LakeSP_048 | geometry | 200 | 4 | 50 | yes |" in md
    # No raster-only tables for a GeoParquet run.
    assert "## Tiling layout" not in md
    assert "## Chunk/shard layout" not in md


def test_summary_renders_flatgeobuf_spatial_index_layout():
    run = _sample_run()
    run.format_id = "flatgeobuf"
    run.object_layouts = [
        FlatGeobufLayout(
            name="LakeSP_048",
            size_bytes=3000,
            geometry_type="MultiPolygon",
            num_features=200,
            has_spatial_index=True,
            index_node_size=16,
            header_bytes=200,
            index_bytes=800,
            feature_bytes=1988,
        )
    ]
    md = render_markdown_summary(run)
    assert "## Spatial-index layout" in md
    assert "Packed Hilbert R-tree:** 1/1 file(s)" in md
    # What the index costs, in bytes and as a share of the stored object.
    assert "Index cost:** 800 B (26.7% of the stored bytes)" in md
    assert "| LakeSP_048 | MultiPolygon | 200 | 0 | 0 | 0 | yes | 16 | 800 B |" in md
    assert "none | 1.00× |" in md
    # No other-format tables for a FlatGeobuf run.
    assert "## Tiling layout" not in md
    assert "## Row-group layout" not in md
    # An ordinary run (no NULL geometry) gets none of the three caveats.
    assert "Content subset" not in md
    assert "Synthesized geometry" not in md
    assert "Fabricated geometry" not in md


def test_summary_flags_a_flatgeobuf_content_subset_from_dropped_null_geometry():
    # #98: a null_geometry: drop run must be unmistakable from an ordinary one.
    run = _sample_run()
    run.format_id = "flatgeobuf"
    run.object_layouts = [
        FlatGeobufLayout(
            name="LakeSP_048",
            size_bytes=3000,
            geometry_type="MultiPolygon",
            num_features=150,
            has_spatial_index=True,
            index_node_size=16,
            header_bytes=200,
            index_bytes=600,
            feature_bytes=2200,
            features_dropped=50,
            content_subset=True,
        )
    ]
    md = render_markdown_summary(run)
    assert "Content subset:** 1 file(s) dropped 50 feature(s)" in md
    assert "| LakeSP_048 | MultiPolygon | 150 | 50 | 0 | 0 | yes |" in md
    assert "> **Content subset:**" in md
    assert "Synthesized geometry" not in md
    assert "Fabricated geometry" not in md


def test_summary_flags_a_flatgeobuf_fabricated_sentinel_geometry():
    # #98: a null_geometry: sentinel run keeps every row but must say the
    # placeholder bytes are not source content.
    run = _sample_run()
    run.format_id = "flatgeobuf"
    run.object_layouts = [
        FlatGeobufLayout(
            name="LakeSP_048",
            size_bytes=3200,
            geometry_type="MultiPolygon",
            num_features=200,
            has_spatial_index=True,
            index_node_size=16,
            header_bytes=200,
            index_bytes=800,
            feature_bytes=2200,
            features_sentinel=50,
            geometry_fabricated=True,
        )
    ]
    md = render_markdown_summary(run)
    assert "Fabricated geometry:** 1 file(s) hold 50 placeholder" in md
    assert "| LakeSP_048 | MultiPolygon | 200 | 0 | 0 | 50 | yes |" in md
    assert "Synthesized geometry" not in md
    assert "Content subset" not in md


def test_summary_flags_a_flatgeobuf_synthesized_point_from_geometry():
    # #98: a null_geometry: point_from run keeps every row and builds a real
    # geometry — a weaker caveat than the sentinel's "fabricated" one.
    run = _sample_run()
    run.format_id = "flatgeobuf"
    run.object_layouts = [
        FlatGeobufLayout(
            name="LakeSP_048",
            size_bytes=3200,
            geometry_type="MultiPolygon",
            num_features=200,
            has_spatial_index=True,
            index_node_size=16,
            header_bytes=200,
            index_bytes=800,
            feature_bytes=2200,
            features_synthesized=50,
            geometry_synthesized=True,
        )
    ]
    md = render_markdown_summary(run)
    assert "Synthesized geometry:** 1 file(s) built 50 geometry(ies)" in md
    assert "| LakeSP_048 | MultiPolygon | 200 | 0 | 50 | 0 | yes |" in md
    assert "> **Synthesized geometry:**" in md
    assert "Content subset" not in md
    assert "Fabricated geometry" not in md


def test_summary_renders_copc_octree_layout():
    run = _sample_run()
    run.format_id = "copc"
    run.object_layouts = [
        CopcLayout(
            name="pixel_cloud",
            size_bytes=400,
            num_nodes=31,
            max_depth=4,
            point_count=40000,
            points_per_node=2506,
            extra_dimensions=["sig0", "water_frac", "classification_1"],
            compression_ratio=3.5,
        )
    ]
    md = render_markdown_summary(run)
    assert "## Octree layout" in md
    assert "Octree nodes:" in md
    # The carried point variables are reported (content-complete, self-describing).
    assert "Carried point variables:** 3 extra dimension(s)" in md
    assert "classification_1, sig0, water_frac" in md  # sorted
    # LASzip compression ratio is surfaced (the format's saving vs the content).
    assert "LASzip compression:** 3.50×" in md
    assert "| pixel_cloud | 31 | 4 | 40000 | 2506 | 3 | laszip | 3.50× |" in md
    # No other-format tables for a COPC run.
    assert "## Tiling layout" not in md
    assert "## Row-group layout" not in md


def test_write_artifacts_writes_both_files(tmp_path):
    run = _sample_run()
    written = write_artifacts(run, str(tmp_path))

    result_path = tmp_path / RESULT_FILENAME
    summary_path = tmp_path / SUMMARY_FILENAME
    assert result_path.exists() and summary_path.exists()
    assert written["result"].endswith(RESULT_FILENAME)
    assert written["summary"].endswith(SUMMARY_FILENAME)

    payload = json.loads(result_path.read_text())
    assert payload["dataset_id"] == "synthetic-cog"
    assert payload["object_profile"]["count"] == 3


def test_summary_omits_run_protocol_section_without_conditions():
    md = render_markdown_summary(_sample_run())
    assert "## Run protocol" not in md


def test_summary_renders_run_protocol_condition_matrix():
    run = _sample_run()
    run.conditions = [
        ConditionResult(
            phase="read",
            cache="warm",
            concurrency=1,
            replicates=[
                ConditionReplicate(
                    index=0,
                    metrics=[
                        MetricResult(name="read_latency_mean", value=0.01, unit="s")
                    ],
                ),
            ],
            aggregate=[
                MetricResult(
                    name="read_latency_mean",
                    value=0.01,
                    unit="s",
                    detail={"replicate_values": [0.01], "median": 0.01, "stdev": 0.0},
                )
            ],
        ),
        ConditionResult(
            phase="read",
            cache="cold",
            concurrency=4,
            replicates=[],
            aggregate=[
                MetricResult(
                    name="read_latency_mean",
                    value=0.05,
                    unit="s",
                    detail={
                        "replicate_values": [0.04, 0.06],
                        "median": 0.05,
                        "stdev": 0.01,
                    },
                )
            ],
        ),
    ]
    md = render_markdown_summary(run)
    assert "## Run protocol" in md
    assert (
        "| read | warm | isolated | 1 | read_latency_mean | 0.01 | 0.01 | 0 | s |" in md
    )
    assert "| read | cold | concurrent (4) |" in md


def test_summary_flags_the_scale_offset_encoding():
    # The run has to say the codec is active, otherwise the report cannot show
    # the encoding-semantics contrast with COG (#54).
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="FRE_B4",
            size_bytes=200,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=1,
            shard_count=4,
            scale_offset=True,
            stored_dtype="int16",
        )
    ]
    md = render_markdown_summary(run)
    assert "Value encoding" in md
    assert "scale_offset → int16" in md
    assert "without" in md and "unscaling" in md


def test_summary_quotes_what_the_overview_pyramid_costs():
    # Size and display are coupled through the pyramid: it is what serves a
    # zoomed-out tile cheaply, and what the store pays for it.
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="FRE_B4",
            size_bytes=400,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=3,
            shard_count=7,
            overview_bytes=100,
        )
    ]
    md = render_markdown_summary(run)
    assert "**Overviews:**" in md
    assert "25.0%" in md


def test_summary_flags_a_store_that_has_no_pyramid():
    # A single-level store makes the tile server downsample full-resolution
    # chunks, so its display numbers are not comparable with an arm that has one.
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="FRE_B4",
            size_bytes=400,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=0,
            shard_count=4,
        )
    ]
    md = render_markdown_summary(run)
    assert "**No overviews:**" in md


def test_summary_omits_the_scale_offset_note_when_inactive():
    run = _sample_run()
    run.format_id = "geozarr"
    run.object_layouts = [
        GeoZarrLayout(
            name="FRE_B4",
            size_bytes=200,
            chunk_shape=[512, 512],
            shard_shape=[1024, 1024],
            chunks_per_shard=4,
            codec="zstd",
            multiscale_levels=1,
            shard_count=4,
            stored_dtype="int16",
        )
    ]
    md = render_markdown_summary(run)
    assert "scale_offset" not in md
