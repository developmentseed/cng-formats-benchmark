"""Tests for the FlatGeobuf (vector, single-file) adapter.

The whole path runs in CI on a synthetic in-memory GeoDataFrame: pyogrio's wheel
bundles the GDAL FlatGeobuf driver, so the indexed write, the header/index layout
and the bbox read through the index are all exercised — with only the ESRI
Shapefile source read guarded, as in the GeoParquet tests.
"""

import struct

import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyogrio")
pytest.importorskip("shapely.geometry")

from shapely.geometry import Point  # noqa: E402

from cng_benchmark.formats.flatgeobuf import (  # noqa: E402
    DEFAULT_INDEX_NODE_SIZE,
    NODE_ITEM_BYTES,
    FlatGeobufAdapter,
    FlatGeobufParams,
    _write_flatgeobuf,
    describe_flatgeobuf_layout,
    packed_rtree_bytes,
    read_flatgeobuf_header,
)


def _gdf(n=200):
    """A small point GeoDataFrame on a 50-wide grid (deterministic geometry)."""
    return gpd.GeoDataFrame(
        {"id": list(range(n)), "label": [f"lake-{i}" for i in range(n)]},
        geometry=[Point(x % 50, x // 50) for x in range(n)],
        crs="EPSG:4326",
    )


def _fgb(tmp_path, name="lakes.fgb", *, n=200, spatial_index=True):
    target = str(tmp_path / name)
    _write_flatgeobuf(_gdf(n), target, spatial_index=spatial_index)
    return target


def test_params_default_to_an_indexed_write():
    # The index is the candidate under evaluation, so it is the default; unknown
    # keys (the shared run-shape params) are tolerated.
    opts = FlatGeobufParams.model_validate({"scope": "product-set"})
    assert opts.spatial_index is True
    off = FlatGeobufParams.model_validate({"spatial_index": False})
    assert off.spatial_index is False


def test_packed_rtree_bytes_follows_the_spec_formula():
    # 50 items at a branching factor of 16: 50 leaves + 4 + 1 internal nodes.
    assert packed_rtree_bytes(50, 16) == 55 * NODE_ITEM_BYTES
    assert packed_rtree_bytes(1, 16) == NODE_ITEM_BYTES
    # No features, or no index (node size 0), means no tree on disk.
    assert packed_rtree_bytes(0, 16) == 0
    assert packed_rtree_bytes(50, 0) == 0


def test_write_produces_one_indexed_file(tmp_path):
    target = _fgb(tmp_path, n=200)
    import os

    adapter = FlatGeobufAdapter()
    sizes = adapter.enumerate_objects(target)
    assert sizes == [os.path.getsize(target)]  # a single addressable object
    header = read_flatgeobuf_header(target)
    assert header.features_count == 200
    assert header.index_node_size == DEFAULT_INDEX_NODE_SIZE
    assert header.geometry_type == "Point"


def test_describe_layout_splits_header_index_and_features(tmp_path):
    target = _fgb(tmp_path, n=200)
    ly = describe_flatgeobuf_layout(target, "lakes")
    assert ly.kind == "flatgeobuf"
    assert ly.name == "lakes"
    assert ly.num_features == 200
    assert ly.has_spatial_index is True
    assert ly.index_node_size == DEFAULT_INDEX_NODE_SIZE
    assert ly.index_bytes == packed_rtree_bytes(200, DEFAULT_INDEX_NODE_SIZE)
    # The three parts plus the 12-byte preamble account for the whole object, so
    # "what does the index cost" is answerable against the size it is paid in.
    assert 12 + ly.header_bytes + ly.index_bytes + ly.feature_bytes == ly.size_bytes
    assert ly.feature_bytes > 0
    # FlatGeobuf defines no compression: stored bytes are the uncompressed bytes.
    assert ly.codec == "none"
    assert ly.compression_ratio == 1.0


def test_index_costs_exactly_its_nodes(tmp_path):
    indexed = describe_flatgeobuf_layout(_fgb(tmp_path, n=200), "x")
    plain = describe_flatgeobuf_layout(
        _fgb(tmp_path, name="plain.fgb", n=200, spatial_index=False), "x"
    )
    assert plain.has_spatial_index is False
    assert plain.index_node_size == 0
    assert plain.index_bytes == 0
    # The indexed file is larger by exactly the tree, and holds the same features.
    assert indexed.size_bytes - plain.size_bytes == indexed.index_bytes
    assert plain.num_features == indexed.num_features
    assert plain.feature_bytes == indexed.feature_bytes


def test_indexed_write_orders_features_along_the_hilbert_curve(tmp_path):
    # Writing the index also sorts the features, which is what keeps the features
    # a bbox query selects contiguous in the file.
    import pyogrio

    indexed = pyogrio.read_dataframe(_fgb(tmp_path, n=100))
    plain = pyogrio.read_dataframe(
        _fgb(tmp_path, name="plain.fgb", n=100, spatial_index=False)
    )
    assert list(plain["id"]) == list(range(100))  # source order
    assert list(indexed["id"]) != list(plain["id"])
    assert sorted(indexed["id"]) == sorted(plain["id"])  # same features


def test_geometry_and_attributes_round_trip(tmp_path):
    import pyogrio

    source = _gdf(120)
    back = pyogrio.read_dataframe(_fgb(tmp_path, n=120))
    assert len(back) == len(source)
    assert set(back.columns) == set(source.columns)
    assert back.crs == source.crs
    by_id = back.set_index("id").sort_index()
    original = source.set_index("id").sort_index()
    assert list(by_id["label"]) == list(original["label"])
    assert by_id.geometry.geom_equals_exact(original.geometry, tolerance=0).all()


def test_read_header_rejects_a_non_flatgeobuf(tmp_path):
    bogus = tmp_path / "not.fgb"
    bogus.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    with pytest.raises(ValueError, match="not a FlatGeobuf"):
        read_flatgeobuf_header(str(bogus))


def test_header_parser_reads_a_written_node_size(tmp_path):
    # The default node size is absent from the header vtable (flatbuffers omit a
    # default), so parse a file where it *is* written: the unindexed write records
    # 0 explicitly, which is how "no index" is told from "the default 16".
    target = _fgb(tmp_path, name="plain.fgb", n=32, spatial_index=False)
    with open(target, "rb") as fh:
        (header_bytes,) = struct.unpack_from("<I", fh.read(12), 8)
    assert header_bytes > 0
    assert read_flatgeobuf_header(target).index_node_size == 0


def test_vector_read_metric_goes_through_the_index(tmp_path):
    target = _fgb(tmp_path, n=200)
    from cng_benchmark.metrics.read import measure_vector_read

    metrics = {m.name: m for m in measure_vector_read(target, role="sink", queries=4)}
    # The same metric names as the GeoParquet path, so the two are comparable.
    assert metrics["read_query_count"].value == 4
    assert metrics["read_latency_mean"].value >= 0
    assert metrics["read_latency_spread"].value >= 0
    detail = metrics["read_decoded_throughput"].detail
    assert detail["features"] > 0
    # The driver reports a fast spatial filter, i.e. the bbox really is pushed
    # through the R-tree rather than emulated by a scan.
    assert detail["spatial_index"] is True


def test_vector_read_reports_an_unindexed_file_as_such(tmp_path):
    target = _fgb(tmp_path, name="plain.fgb", n=200, spatial_index=False)
    from cng_benchmark.metrics.read import measure_vector_read

    metrics = {m.name: m for m in measure_vector_read(target, role="sink", queries=2)}
    assert metrics["read_decoded_throughput"].detail["spatial_index"] is False


def test_bbox_query_returns_only_the_features_it_covers(tmp_path):
    import pyogrio

    target = _fgb(tmp_path, n=200)
    sub = pyogrio.read_dataframe(target, bbox=(0, 0, 9, 0))
    # Row 0 of the 50-wide grid, columns 0..9 — 10 of the 200 features.
    assert len(sub) == 10
    assert sorted(sub["id"]) == list(range(10))


def test_convert_reads_a_shapefile_and_writes_flatgeobuf(tmp_path):
    source = str(tmp_path / "lakes.shp")
    _gdf(120).to_file(source, driver="ESRI Shapefile")

    target = str(tmp_path / "out.fgb")
    FlatGeobufAdapter().convert(source, target, {"spatial_index": True})
    ly = describe_flatgeobuf_layout(target, "lakes")
    assert ly.num_features == 120
    assert ly.has_spatial_index is True
