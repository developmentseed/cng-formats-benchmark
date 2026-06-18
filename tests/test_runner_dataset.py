"""Tests for the dataset fan-out runner (multi-object products + roll-up)."""

from pathlib import Path

import pytest

from cng_benchmark.config import DatasetConfig, load_benchmark_config
from cng_benchmark.datasets.base import Dataset, Product, SourceObject
from cng_benchmark.registry import DATASETS
from cng_benchmark.report import write_product_set_artifacts
from cng_benchmark.runner import run_dataset_benchmark

SYNTHETIC = "configs/benchmarks/synthetic_cog.yaml"


@DATASETS.register("test-multifile")
class _MultiFileDataset(Dataset):
    """Test reader: each subdir of ``source`` is a product of its ``*.tif``."""

    def products(self, *, prefix=None, limit=None):
        root = Path(self.source_uri)
        products = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            if prefix and prefix not in sub.name:
                continue
            components = [
                SourceObject(name=f.stem, uri=str(f)) for f in sorted(sub.glob("*.tif"))
            ]
            products.append(Product(id=sub.name, components=components))
        if limit is not None:
            products = products[:limit]
        return products


def _write_product(root: Path, product_id: str, n_components: int) -> None:
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    d = root / product_id
    d.mkdir(parents=True)
    for i in range(n_components):
        (d / f"band{i}.tif").write_bytes(generate_cog_bytes(size=128, blocksize=128))


def _dataset_config(source: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "id": "multi",
            "reader": "test-multifile",
            "source": str(source),
            "baseline_format": "geotiff",
            "target_formats": ["cog"],
        }
    )


def _benchmark(metrics, params):
    return load_benchmark_config(SYNTHETIC).model_copy(
        update={"metrics": metrics, "params": params}
    )


def test_single_product_aggregates_all_objects(tmp_path):
    src = tmp_path / "src"
    _write_product(src, "sceneA", n_components=4)
    output = tmp_path / "out"

    cfg = _benchmark(["write", "object_size", "read"], {"scope": "product"})
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))

    assert len(result.per_product) == 1
    run = result.per_product[0]
    assert run.object_profile.count == 4  # one object per component
    assert run.params["product_id"] == "sceneA"
    assert run.params["scope"] == "product"
    # write is aggregated to a single pair of metrics over the 4 components.
    write_names = [m.name for m in run.metrics if m.name.startswith("write_")]
    assert write_names == ["write_elapsed", "write_throughput"]
    throughput = next(m for m in run.metrics if m.name == "write_throughput")
    assert throughput.detail["components"] == 4
    # read sampled to the default 1 component.
    assert sum(m.name == "read_window_count" for m in run.metrics) == 1
    # Roll-up over a single product mirrors it.
    assert result.rollup.object_profile.count == 4
    assert result.rollup.params["scope"] == "rollup"


def test_object_layouts_captured_and_pooled(tmp_path):
    src = tmp_path / "src"
    _write_product(src, "sceneA", n_components=3)
    output = tmp_path / "out"

    cfg = _benchmark(["object_size"], {"scope": "product"})
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))

    run = result.per_product[0]
    # One layout per produced object, and the synthetic COGs are tiled.
    assert len(run.object_layouts) == 3
    assert all(ly.is_tiled for ly in run.object_layouts)
    assert all(ly.internal_tiles >= 1 for ly in run.object_layouts)
    # The roll-up pools every object's layout.
    assert len(result.rollup.object_layouts) == 3


def test_product_set_summary_shows_tiling(tmp_path):
    from cng_benchmark.report import render_markdown_summary, render_product_set_summary

    src = tmp_path / "src"
    _write_product(src, "sceneA", n_components=2)
    output = tmp_path / "out"
    cfg = _benchmark(["object_size"], {"scope": "product"})
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))

    per_product_md = render_markdown_summary(result.per_product[0])
    assert "## Tiling layout" in per_product_md
    assert "Internally tiled:" in per_product_md
    assert "| Tiled |" in render_product_set_summary(result)


def test_product_set_rollup_pools_all_products(tmp_path):
    src = tmp_path / "src"
    _write_product(src, "scene1", n_components=3)
    _write_product(src, "scene2", n_components=2)
    output = tmp_path / "out"

    cfg = _benchmark(["object_size"], {"scope": "product-set"})
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))

    assert len(result.per_product) == 2
    per_counts = sorted(r.object_profile.count for r in result.per_product)
    assert per_counts == [2, 3]
    # The roll-up count equals the sum of the per-product counts.
    assert result.rollup.object_profile.count == 5
    assert result.rollup.params["product_count"] == 2


def test_product_set_bounded_by_limit(tmp_path):
    src = tmp_path / "src"
    _write_product(src, "scene1", n_components=1)
    _write_product(src, "scene2", n_components=1)
    _write_product(src, "scene3", n_components=1)
    output = tmp_path / "out"

    cfg = _benchmark(
        ["object_size"], {"scope": "product-set", "products": {"limit": 2}}
    )
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))
    assert len(result.per_product) == 2


def test_write_product_set_tree(tmp_path):
    src = tmp_path / "src"
    _write_product(src, "sceneA", n_components=2)
    _write_product(src, "sceneB", n_components=2)
    output = tmp_path / "out"

    cfg = _benchmark(["object_size"], {"scope": "product-set"})
    result = run_dataset_benchmark(cfg, _dataset_config(src), str(output))
    write_product_set_artifacts(result, str(output))

    assert (output / "product" / "sceneA" / "result.json").exists()
    assert (output / "product" / "sceneB" / "summary.md").exists()
    assert (output / "rollup" / "result.json").exists()
    top = (output / "summary.md").read_text()
    assert "roll-up" in top
    assert "sceneA" in top and "sceneB" in top


def test_no_products_raises(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    cfg = _benchmark(["object_size"], {"scope": "product-set"})
    with pytest.raises(ValueError, match="no products"):
        run_dataset_benchmark(cfg, _dataset_config(src), str(tmp_path / "out"))
