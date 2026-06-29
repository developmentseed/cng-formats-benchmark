"""Tests for result-artifact rendering and persistence."""

import json
from datetime import UTC, datetime

from cng_benchmark.config import load_benchmark_config, tier_policy_from_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.models import (
    BenchmarkRun,
    CogLayout,
    CopcLayout,
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
    # No COG-only table for a GeoZarr run.
    assert "## Tiling layout" not in md


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
