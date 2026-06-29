"""Tests for the dataset fan-out runner (multi-object products + roll-up)."""

import io
import zipfile
from pathlib import Path

import pytest

from cng_benchmark.config import DatasetConfig, load_benchmark_config
from cng_benchmark.datasets.base import Dataset, Product, SourceObject
from cng_benchmark.registry import DATASETS, FORMATS
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


@DATASETS.register("test-binfiles")
class _BinFileDataset(Dataset):
    """Test reader: one product whose components are the ``*.bin`` files in source."""

    def products(self, *, prefix=None, limit=None):
        root = Path(self.source_uri)
        comps = [
            SourceObject(name=f.stem, uri=str(f)) for f in sorted(root.glob("*.bin"))
        ]
        return [Product(id="sceneZ", components=comps)] if comps else []


@DATASETS.register("test-zip-delivery")
class _ZipDeliveryDataset(Dataset):
    """Test reader: each *.zip under source is a product whose members are components.

    Member URIs are composed as ``/vsizip/<zip_path>/<member>`` — the same shape
    that :mod:`cng_benchmark.datasets.zip_delivery` uses for S1/S2/LakeSP.
    """

    def products(self, *, prefix=None, limit=None):
        root = Path(self.source_uri)
        zips = sorted(root.glob("*.zip"))
        if limit is not None:
            zips = zips[:limit]
        products = []
        for zip_path in zips:
            with zipfile.ZipFile(zip_path) as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
            components = [
                SourceObject(name=Path(m).stem, uri=f"/vsizip/{zip_path}/{m}")
                for m in sorted(members)
            ]
            products.append(Product(id=zip_path.stem, components=components))
        return products


def _make_zip(path: Path, members: dict[str, bytes]) -> None:
    """Write a zip archive at ``path`` with ``members`` (name → content)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    path.write_bytes(buf.getvalue())


def _register_fake_passthrough():
    """Register a no-op adapter that writes a fixed dummy output, ignoring source.

    Lets the runner's bytes_in / source-size path be tested without needing
    rasterio to actually open the source (the source_path is passed to convert
    but never opened here).
    """
    import os

    from cng_benchmark.formats.base import FormatAdapter, ObjectKind

    if "fake-passthrough" in FORMATS:
        return

    @FORMATS.register("fake-passthrough")
    class _PassthroughAdapter(FormatAdapter):
        name = "fake-passthrough"
        object_kind = ObjectKind.RASTER_FILE

        def target_basename(self):
            return "out.bin"

        def convert(self, source, target, params):
            Path(target).write_bytes(b"dummy-output" * 4)

        def describe_grouping_lever(self):
            return "none"

        def enumerate_objects(self, target):
            return [os.path.getsize(target)]

        def describe_layout(self, target, *, name=None):
            return []


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
    summary = render_product_set_summary(result)
    assert "| Layout |" in summary
    assert "tiled" in summary  # COG products report their tiled fraction


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


# --- zarr-store object kind flows through the per-component path --------------


def _register_fake_zarr():
    """Register a store-kind adapter that writes a tiny sharded store per source.

    Lets the runner's store branch (directory target, ``upload_tree``,
    store-walking ``enumerate_objects``, a ``GeoZarrLayout`` describer, directory
    cleanup) be exercised with only zarr + numpy — no rioxarray source read.
    """
    from cng_benchmark.formats.base import FormatAdapter, ObjectKind
    from cng_benchmark.formats.geozarr import (
        describe_store_layout,
        enumerate_store_objects,
    )
    from cng_benchmark.registry import FORMATS

    if "fake-zarr" in FORMATS:
        return

    @FORMATS.register("fake-zarr")
    class _FakeZarrAdapter(FormatAdapter):
        name = "fake-zarr"
        object_kind = ObjectKind.ZARR_STORE

        def target_basename(self):
            return "geozarr.zarr"

        def convert(self, source, target, params):
            import numpy as np

            from cng_benchmark.formats.geozarr import _write_sharded

            data = (np.arange(1024 * 1024, dtype="uint16") % 100).reshape(1024, 1024)
            _write_sharded(
                target, data, chunk=(512, 512), shard=(512, 512), codec="none"
            )

        def describe_grouping_lever(self):
            return "Zarr v3 chunk and shard shape"

        def enumerate_objects(self, target):
            return enumerate_store_objects(target)

        def describe_layout(self, target, *, name=None):
            return [describe_store_layout(target, name or self.name)]


def test_zarr_store_object_flows_through_per_component_path(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("xarray")
    # The runner reads every source through a GDAL session (rasterio), regardless
    # of the produced object kind — so this integration test needs the geo stack,
    # exactly like the COG runner tests. The store-writing logic itself is covered
    # without rasterio in test_geozarr.py.
    pytest.importorskip("rasterio")
    _register_fake_zarr()

    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        (src / f"band{i}.bin").write_bytes(b"x" * 1000)
    output = tmp_path / "out"

    cfg = _benchmark(["write", "object_size"], {"scope": "product"}).model_copy(
        update={"formats": ["fake-zarr"]}
    )
    ds_cfg = DatasetConfig.model_validate(
        {
            "id": "z",
            "reader": "test-binfiles",
            "source": str(src),
            "baseline_format": "geotiff",
            "target_formats": ["fake-zarr"],
        }
    )
    result = run_dataset_benchmark(cfg, ds_cfg, str(output))

    run = result.per_product[0]
    # 2 components × 4 shards (1024/512 = 2 per side) = 8 stored objects.
    assert run.object_profile.count == 8
    # One GeoZarrLayout per component (per produced array), pooled into the roll-up.
    assert [ly.kind for ly in run.object_layouts] == ["geozarr", "geozarr"]
    assert len(result.rollup.object_layouts) == 2
    # Published as a tree under each component dir, not a single file.
    assert (
        output / "objects" / "sceneZ" / "band0" / "geozarr.zarr" / "zarr.json"
    ).exists()


# --- vector-file object kind flows through the per-component path -------------


def _register_fake_geoparquet():
    """Register a vector-file adapter that writes a tiny GeoParquet per source.

    Lets the runner's vector branch (single-file target + publish, file-size
    ``enumerate_objects``, a ``GeoParquetLayout`` describer, the bbox/row-group
    read collector, file cleanup) be exercised with only geopandas + pyarrow +
    shapely — no OGR source read.
    """
    from cng_benchmark.formats.base import FormatAdapter, ObjectKind
    from cng_benchmark.formats.geoparquet import (
        _write_geoparquet,
        describe_geoparquet_layout,
    )
    from cng_benchmark.registry import FORMATS

    if "fake-geoparquet" in FORMATS:
        return

    @FORMATS.register("fake-geoparquet")
    class _FakeGeoParquetAdapter(FormatAdapter):
        name = "fake-geoparquet"
        object_kind = ObjectKind.VECTOR_FILE

        def target_basename(self):
            return "geoparquet.parquet"

        def convert(self, source, target, params):
            import geopandas as gpd
            from shapely.geometry import Point

            gdf = gpd.GeoDataFrame(
                {"id": list(range(200))},
                geometry=[Point(x % 50, x // 50) for x in range(200)],
                crs="EPSG:4326",
            )
            _write_geoparquet(gdf, target, row_group_rows=50, spatial_partitioning=True)

        def describe_grouping_lever(self):
            return "GeoParquet row-group size and spatial partitioning"

        def enumerate_objects(self, target):
            import os

            return [os.path.getsize(target)]

        def describe_layout(self, target, *, name=None):
            return [describe_geoparquet_layout(target, name or self.name)]


def test_vector_file_object_flows_through_per_component_path(tmp_path):
    pytest.importorskip("geopandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("shapely")
    # The runner reads every source through a GDAL session (rasterio) before the
    # convert, regardless of object kind — so this integration test needs the geo
    # stack, like the COG/GeoZarr runner tests. The write/read logic itself is
    # covered without rasterio in test_geoparquet.py.
    pytest.importorskip("rasterio")
    _register_fake_geoparquet()

    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        (src / f"pass{i}.bin").write_bytes(b"x" * 1000)
    output = tmp_path / "out"

    cfg = _benchmark(["write", "object_size", "read"], {"scope": "product"}).model_copy(
        update={"formats": ["fake-geoparquet"]}
    )
    ds_cfg = DatasetConfig.model_validate(
        {
            "id": "v",
            "reader": "test-binfiles",
            "source": str(src),
            "baseline_format": "shapefile",
            "target_formats": ["fake-geoparquet"],
        }
    )
    result = run_dataset_benchmark(cfg, ds_cfg, str(output))

    run = result.per_product[0]
    # One single-file object per component (2 passes).
    assert run.object_profile.count == 2
    assert [ly.kind for ly in run.object_layouts] == ["geoparquet", "geoparquet"]
    # The vector read metric ran (bbox/row-group query, not a raster window).
    read_names = {m.name for m in run.metrics}
    assert "read_query_count" in read_names
    assert "read_window_count" not in read_names
    # Published as a single file under each component dir, not a tree.
    assert (output / "objects" / "sceneZ" / "pass0" / "geoparquet.parquet").is_file()


# --- point-cloud object kind flows through the per-component path -------------


def _register_fake_copc():
    """Register a point-cloud adapter that writes a tiny COPC per source.

    Lets the runner's point-cloud branch (single-file target + publish, file-size
    ``enumerate_objects``, a ``CopcLayout`` describer, the octree-node read
    collector, file cleanup) be exercised with only copclib + laspy + numpy — no
    netCDF source read.
    """
    import numpy as np

    from cng_benchmark.formats.base import FormatAdapter, ObjectKind
    from cng_benchmark.formats.copc import _build_copc, describe_copc_layout
    from cng_benchmark.registry import FORMATS

    if "fake-copc" in FORMATS:
        return

    @FORMATS.register("fake-copc")
    class _FakeCopcAdapter(FormatAdapter):
        name = "fake-copc"
        object_kind = ObjectKind.POINT_CLOUD_FILE

        def target_basename(self):
            return "copc.laz"

        def convert(self, source, target, params):
            rng = np.random.default_rng(0)
            n = 40_000
            _build_copc(
                target,
                rng.uniform(300000, 300500, n),
                rng.uniform(4900000, 4900500, n),
                rng.uniform(0, 100, n),
                span=32,
                max_depth=4,
            )

        def describe_grouping_lever(self):
            return "COPC octree depth and per-node point budget"

        def enumerate_objects(self, target):
            import os

            return [os.path.getsize(target)]

        def describe_layout(self, target, *, name=None):
            return [describe_copc_layout(target, name or self.name)]


def test_point_cloud_object_flows_through_per_component_path(tmp_path):
    pytest.importorskip("copclib")
    pytest.importorskip("laspy")
    # The runner reads every source through a GDAL session (rasterio) before the
    # convert, regardless of object kind — like the other runner tests. The
    # write/read logic itself is covered without rasterio in test_copc.py.
    pytest.importorskip("rasterio")
    _register_fake_copc()

    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        (src / f"granule{i}.bin").write_bytes(b"x" * 1000)
    output = tmp_path / "out"

    cfg = _benchmark(["write", "object_size", "read"], {"scope": "product"}).model_copy(
        update={"formats": ["fake-copc"]}
    )
    ds_cfg = DatasetConfig.model_validate(
        {
            "id": "pc",
            "reader": "test-binfiles",
            "source": str(src),
            "baseline_format": "netcdf",
            "target_formats": ["fake-copc"],
        }
    )
    result = run_dataset_benchmark(cfg, ds_cfg, str(output))

    run = result.per_product[0]
    # One single-file object per component (2 granules).
    assert run.object_profile.count == 2
    assert [ly.kind for ly in run.object_layouts] == ["copc", "copc"]
    # The COPC read metric ran (octree-node query, not a raster window).
    read_names = {m.name for m in run.metrics}
    assert "read_query_count" in read_names
    assert "read_window_count" not in read_names
    # Published as a single file under each component dir, not a tree.
    assert (output / "objects" / "sceneZ" / "granule0" / "copc.laz").is_file()


# --- bytes_in source-size coverage across source layouts -----------------------


def _write_ds_cfg(reader: str, source: Path) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "id": "bytes-in-test",
            "reader": reader,
            "source": str(source),
            "baseline_format": "geotiff",
            "target_formats": ["fake-passthrough"],
        }
    )


def test_flat_source_bytes_in_is_sum_of_component_sizes(tmp_path):
    """bytes_in for flat (non-zip) sources equals the sum of per-component file sizes."""
    pytest.importorskip("rasterio")
    _register_fake_passthrough()

    src = tmp_path / "src"
    src.mkdir()
    sizes = [100, 200, 150]
    for i, sz in enumerate(sizes):
        (src / f"band{i}.bin").write_bytes(b"x" * sz)
    output = tmp_path / "out"

    cfg = _benchmark(["write"], {"scope": "product"}).model_copy(
        update={"formats": ["fake-passthrough"]}
    )
    result = run_dataset_benchmark(cfg, _write_ds_cfg("test-binfiles", src), str(output))

    run = result.per_product[0]
    throughput = next(m for m in run.metrics if m.name == "write_throughput")
    assert "bytes_in" in throughput.detail
    assert throughput.detail["bytes_in"] == sum(sizes)


def test_zip_delivered_bytes_in_equals_zip_size_once(tmp_path):
    """bytes_in for zip-delivered sources equals the zip size, not N×zip_size."""
    pytest.importorskip("rasterio")
    _register_fake_passthrough()

    src = tmp_path / "src"
    src.mkdir()
    members = {"band1.bin": b"a" * 500, "band2.bin": b"b" * 500, "band3.bin": b"c" * 500}
    _make_zip(src / "scene.zip", members)
    zip_size = (src / "scene.zip").stat().st_size
    output = tmp_path / "out"

    cfg = _benchmark(["write"], {"scope": "product"}).model_copy(
        update={"formats": ["fake-passthrough"]}
    )
    result = run_dataset_benchmark(
        cfg, _write_ds_cfg("test-zip-delivery", src), str(output)
    )

    run = result.per_product[0]
    assert len(run.metrics) > 0, "expected write metrics"
    throughput = next(m for m in run.metrics if m.name == "write_throughput")
    assert "bytes_in" in throughput.detail, "bytes_in missing for zip-delivered source"
    # Must equal the zip size exactly — not 3× (one per band).
    assert throughput.detail["bytes_in"] == zip_size
    assert throughput.detail["components"] == len(members)
