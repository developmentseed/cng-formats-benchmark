"""Tests for result-artifact rendering and persistence."""

import json
from datetime import UTC, datetime

from cng_benchmark.config import load_benchmark_config, tier_policy_from_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.models import BenchmarkRun, MetricResult
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
