"""Tests for the GeoParquet (vector, single-file) adapter.

The write core (row-group lever, enumerate, layout) and the bbox/row-group vector
read metric are exercised on a synthetic in-memory GeoDataFrame with only
geopandas + pyarrow + shapely, so they run in CI. The ``convert`` source-read path
needs an OGR driver (pyogrio) and is guarded with ``importorskip``.
"""

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyarrow")
shapely_geometry = pytest.importorskip("shapely.geometry")

from shapely.geometry import Point  # noqa: E402

from cng_benchmark.formats.geoparquet import (  # noqa: E402
    DEFAULT_ROW_GROUP_ROWS,
    GeoParquetParams,
    _row_group_rows,
    _spatial_sort,
    _write_geoparquet,
    describe_geoparquet_layout,
)


def _gdf(n=200):
    """A small point GeoDataFrame on a 50-wide grid (deterministic geometry)."""
    return gpd.GeoDataFrame(
        {"id": list(range(n))},
        geometry=[Point(x % 50, x // 50) for x in range(n)],
        crs="EPSG:4326",
    )


def _parquet(tmp_path, name="lakes.parquet", *, n=200, **kw):
    target = str(tmp_path / name)
    opts = dict(row_group_rows=50, spatial_partitioning=True, compression="zstd")
    opts.update(kw)
    _write_geoparquet(_gdf(n), target, **opts)
    return target


def test_row_group_rows_normalises():
    # None/empty -> default; swept list -> first; scalar -> int.
    assert _row_group_rows(None) == DEFAULT_ROW_GROUP_ROWS
    assert _row_group_rows([]) == DEFAULT_ROW_GROUP_ROWS
    assert _row_group_rows([1000, 2000]) == 1000
    assert _row_group_rows("250") == 250


def test_params_default_and_tolerate_extra_keys():
    opts = GeoParquetParams.model_validate(
        {"row_group_rows": 1000, "scope": "product-set"}
    )
    assert opts.row_group_rows == 1000
    assert opts.spatial_partitioning is True
    assert opts.compression == "zstd"


def test_write_produces_one_file_with_row_groups(tmp_path):
    target = _parquet(tmp_path, n=200, row_group_rows=50)
    import os

    from cng_benchmark.formats.geoparquet import GeoParquetAdapter

    adapter = GeoParquetAdapter()
    sizes = adapter.enumerate_objects(target)
    assert sizes == [os.path.getsize(target)]  # a single addressable object
    assert sizes[0] > 0


def test_describe_layout_reports_row_groups_and_bbox_covering(tmp_path):
    target = _parquet(tmp_path, n=200, row_group_rows=50)
    ly = describe_geoparquet_layout(target, "lakes")
    assert ly.kind == "geoparquet"
    assert ly.name == "lakes"
    assert ly.num_rows == 200
    assert ly.num_row_groups == 4  # 200 rows / 50 per group
    assert ly.row_group_rows == 50
    assert ly.geometry_column == "geometry"
    assert ly.has_bbox_covering is True  # written with write_covering_bbox=True


def test_row_group_lever_changes_group_count(tmp_path):
    coarse = describe_geoparquet_layout(
        _parquet(tmp_path, name="coarse.parquet", n=200, row_group_rows=200), "x"
    )
    fine = describe_geoparquet_layout(
        _parquet(tmp_path, name="fine.parquet", n=200, row_group_rows=25), "x"
    )
    assert coarse.num_row_groups == 1
    assert fine.num_row_groups == 8


def test_spatial_sort_preserves_features_and_reorders():
    gdf = _gdf(100)
    out = _spatial_sort(gdf)
    assert len(out) == len(gdf)
    # The Hilbert ordering is not the identity for a 2D grid of points.
    assert list(out["id"]) != list(gdf["id"])


def test_vector_read_metric_round_trips(tmp_path):
    target = _parquet(tmp_path, n=200, row_group_rows=50)
    from cng_benchmark.metrics.read import measure_vector_read

    metrics = {m.name: m for m in measure_vector_read(target, role="sink", queries=4)}
    assert metrics["read_query_count"].value == 4
    assert metrics["read_latency_mean"].value >= 0
    # The bbox queries cover the full extent, so they return features in total.
    assert metrics["read_decoded_throughput"].detail["features"] > 0


def test_convert_reads_a_shapefile_and_writes_geoparquet(tmp_path):
    pytest.importorskip("pyogrio")
    from cng_benchmark.formats.geoparquet import GeoParquetAdapter

    source = str(tmp_path / "lakes.shp")
    _gdf(120).to_file(source, driver="ESRI Shapefile")

    target = str(tmp_path / "out.parquet")
    GeoParquetAdapter().convert(source, target, {"row_group_rows": 40})
    ly = describe_geoparquet_layout(target, "lakes")
    assert ly.num_rows == 120
    assert ly.num_row_groups == 3  # 120 / 40
    assert ly.has_bbox_covering is True
