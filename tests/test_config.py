"""Tests for the config schema, loaders, and shipped examples."""

import pytest
from pydantic import ValidationError

from cng_benchmark.config import (
    BenchmarkConfig,
    load_benchmark_config,
    load_dataset_config,
    tier_policy_from_config,
)

DATASET_EXAMPLE = "configs/datasets/example_cog.yaml"
BENCHMARK_EXAMPLE = "configs/benchmarks/example_cog.yaml"


MAJA_DATASET_EXAMPLE = "configs/datasets/example_sentinel2_maja.yaml"
MAJA_BENCHMARK_EXAMPLE = "configs/benchmarks/example_sentinel2_maja_cog.yaml"
MAJA_GEOZARR_BENCHMARK_EXAMPLE = (
    "configs/benchmarks/example_sentinel2_maja_geozarr.yaml"
)


def test_example_dataset_config_validates():
    cfg = load_dataset_config(DATASET_EXAMPLE)
    assert cfg.id == "example-raster"
    assert cfg.baseline_format == "geotiff"
    assert "cog" in cfg.target_formats
    # The single-object reader is the default and needs no options.
    assert cfg.reader == "single-object"
    assert cfg.options == {}


def test_example_maja_dataset_config_validates():
    cfg = load_dataset_config(MAJA_DATASET_EXAMPLE)
    assert cfg.reader == "sentinel2-maja"
    assert cfg.options["masks"] == ["CLM", "EDG", "SAT", "MG2"]


def test_example_maja_benchmark_config_validates():
    cfg = load_benchmark_config(MAJA_BENCHMARK_EXAMPLE)
    assert cfg.params["scope"] == "product-set"
    assert cfg.params["products"]["limit"] == 3


def test_example_maja_geozarr_benchmark_config_validates():
    cfg = load_benchmark_config(MAJA_GEOZARR_BENCHMARK_EXAMPLE)
    assert cfg.formats == ["geozarr"]
    assert cfg.params["shard_shape"] == [2048, 2048]
    assert cfg.params["codec"] == "zstd"
    # The MAJA dataset offers both candidate formats.
    ds = load_dataset_config(MAJA_DATASET_EXAMPLE)
    assert "geozarr" in ds.target_formats


def test_example_benchmark_config_validates():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    assert cfg.dataset == "example-raster"
    assert cfg.formats
    assert [t.name for t in cfg.tiers] == ["warm", "cold"]


def test_tier_policy_orders_by_minimum():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    policy = tier_policy_from_config(cfg.tiers)
    mins = [t.min_object_bytes for t in policy.tiers]
    assert mins == sorted(mins)
    assert policy.highest_fit(200 * 1024 * 1024) == "cold"


def test_invalid_benchmark_config_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate({"id": "x"})  # missing required fields


def test_non_mapping_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        load_benchmark_config(bad)
