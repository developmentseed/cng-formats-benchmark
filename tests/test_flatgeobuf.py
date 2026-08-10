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

from shapely.geometry import Point, Polygon  # noqa: E402

from cng_benchmark.formats.flatgeobuf import (  # noqa: E402
    DEFAULT_INDEX_NODE_SIZE,
    NODE_ITEM_BYTES,
    FlatGeobufAdapter,
    FlatGeobufParams,
    _apply_null_geometry_policy,
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


def _gdf_with_nulls(n=200, *, null_every=4):
    """A polygon GeoDataFrame where every ``null_every``-th feature is NULL.

    Mirrors the SWOT LakeSP Prior shape (#98): a prior-database delivery whose
    unobserved features carry no geometry at all.
    """
    geoms = []
    for i in range(n):
        if i % null_every == 0:
            geoms.append(None)
        else:
            geoms.append(Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]))
    return gpd.GeoDataFrame(
        {"id": list(range(n)), "label": [f"lake-{i}" for i in range(n)]},
        geometry=geoms,
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


# --- NULL geometry policy (#98): a prior-database delivery like SWOT LakeSP
# Prior leaves the geometry NULL for features it did not observe, and
# FlatGeobuf's packed Hilbert R-tree cannot index a NULL geometry. ----------


def _geojson_source(tmp_path, gdf, name="source.geojson"):
    path = str(tmp_path / name)
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_params_null_geometry_defaults_to_error():
    assert FlatGeobufParams.model_validate({}).null_geometry == "error"
    for value in ("error", "drop", "sentinel"):
        assert (
            FlatGeobufParams.model_validate({"null_geometry": value}).null_geometry
            == value
        )


def test_null_geometry_policy_is_a_noop_without_nulls():
    gdf = _gdf(50)
    out, dropped, sentinel, synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(null_geometry="drop")
    )
    assert dropped == 0
    assert sentinel == 0
    assert synthesized == 0
    assert len(out) == len(gdf)


def test_null_geometry_error_names_the_count_and_share_before_writing(tmp_path):
    gdf = _gdf_with_nulls(200, null_every=4)  # 50 of 200 are NULL (25%)
    with pytest.raises(ValueError, match=r"50 of 200 features \(25%\)"):
        _apply_null_geometry_policy(gdf, FlatGeobufParams(null_geometry="error"))


def test_null_geometry_error_is_fine_when_unindexed():
    # GDAL only refuses a NULL geometry when the spatial index is on; without
    # it, a NULL geometry is written without complaint.
    gdf = _gdf_with_nulls(200, null_every=4)
    out, dropped, sentinel, synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(spatial_index=False, null_geometry="error")
    )
    assert dropped == 0
    assert sentinel == 0
    assert synthesized == 0
    assert out.geometry.isna().sum() == 50


def test_null_geometry_all_null_raises_regardless_of_policy():
    gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[None, None], crs="EPSG:4326")
    with pytest.raises(ValueError, match="every one of 2 features"):
        _apply_null_geometry_policy(gdf, FlatGeobufParams(null_geometry="drop"))


def test_null_geometry_drop_removes_only_the_null_rows():
    gdf = _gdf_with_nulls(200, null_every=4)
    out, dropped, sentinel, synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(null_geometry="drop")
    )
    assert dropped == 50
    assert sentinel == 0
    assert synthesized == 0
    assert len(out) == 150
    assert not out.geometry.isna().any()


def test_convert_with_drop_writes_the_subset_and_flags_the_layout(tmp_path):
    source = _geojson_source(tmp_path, _gdf_with_nulls(200, null_every=4))
    target = str(tmp_path / "out.fgb")
    FlatGeobufAdapter().convert(
        source, target, {"spatial_index": True, "null_geometry": "drop"}
    )

    header = read_flatgeobuf_header(target)
    assert header.features_count == 150  # 200 - 50 NULL

    ly = describe_flatgeobuf_layout(target, "lakes", features_dropped=50)
    assert ly.num_features == 150
    assert ly.features_dropped == 50
    assert ly.content_subset is True
    assert ly.has_spatial_index is True  # the index writes fine over the subset
    assert ly.features_sentinel == 0
    assert ly.geometry_fabricated is False


def test_null_geometry_sentinel_keeps_every_row_and_flags_placeholders():
    gdf = _gdf_with_nulls(200, null_every=4)
    out, dropped, sentinel, synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(null_geometry="sentinel")
    )
    assert dropped == 0
    assert sentinel == 50
    assert synthesized == 0
    assert len(out) == 200  # no row dropped
    assert not out.geometry.isna().any()  # every geometry is now real
    assert int(out["null_geometry"].sum()) == 50
    # The real rows keep their real geometry; only the null ones changed.
    real = out.loc[~out["null_geometry"]]
    assert real.geom_equals_exact(gdf.loc[real.index].geometry, tolerance=0).all()


def test_null_geometry_sentinel_flag_column_is_collision_safe():
    gdf = _gdf_with_nulls(20, null_every=4)
    gdf["null_geometry"] = "pre-existing column"
    out, _dropped, sentinel, _synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(null_geometry="sentinel")
    )
    assert sentinel == 5
    assert out["null_geometry"].tolist() == gdf["null_geometry"].tolist()  # untouched
    assert "null_geometry_" in out.columns


def test_null_geometry_sentinel_placeholder_sits_outside_the_real_extent():
    gdf = _gdf_with_nulls(200, null_every=4)
    out, _dropped, _sentinel, _synthesized = _apply_null_geometry_policy(
        gdf, FlatGeobufParams(null_geometry="sentinel")
    )
    real_bounds = gdf.loc[~gdf.geometry.isna()].total_bounds
    placeholders = out.loc[out["null_geometry"]]
    minx, miny, maxx, maxy = real_bounds
    # Every placeholder's bounds fall outside the real content's bbox.
    for geom in placeholders.geometry:
        gminx, gminy, _gmaxx, _gmaxy = geom.bounds
        assert gminx < minx or gminy < miny


def test_convert_with_sentinel_writes_all_features_and_flags_fabrication(tmp_path):
    source = _geojson_source(tmp_path, _gdf_with_nulls(200, null_every=4))
    target = str(tmp_path / "out.fgb")
    FlatGeobufAdapter().convert(
        source, target, {"spatial_index": True, "null_geometry": "sentinel"}
    )

    header = read_flatgeobuf_header(target)
    assert header.features_count == 200  # every feature kept

    ly = describe_flatgeobuf_layout(target, "lakes", features_sentinel=50)
    assert ly.num_features == 200
    assert ly.features_sentinel == 50
    assert ly.geometry_fabricated is True
    assert ly.features_dropped == 0
    assert ly.content_subset is False
    # Polygon + a hairline-Polygon sentinel promotes cleanly, unlike a bare
    # Point sentinel among polygons (which reports "Unknown").
    assert ly.geometry_type == "Polygon"


def _gdf_with_nulls_and_priors(n=200, *, null_every=4):
    """Like :func:`_gdf_with_nulls`, plus ``p_lon``/``p_lat`` on every row.

    Mirrors what investigating the real LakeSP product found (#98): the
    NULL-geometry rows are not positionless — they carry the prior lake's own
    reference coordinates on every row, geometry or not.
    """
    gdf = _gdf_with_nulls(n, null_every=null_every)
    gdf["p_lon"] = [(-10.0 + 0.1 * i) for i in range(n)]
    gdf["p_lat"] = [(40.0 + 0.1 * i) for i in range(n)]
    return gdf


def test_null_geometry_point_from_requires_the_param():
    gdf = _gdf_with_nulls_and_priors(200, null_every=4)
    with pytest.raises(ValueError, match="requires null_geometry_point_from"):
        _apply_null_geometry_policy(gdf, FlatGeobufParams(null_geometry="point_from"))


def test_null_geometry_point_from_requires_the_named_columns_to_exist():
    gdf = _gdf_with_nulls(200, null_every=4)  # no p_lon/p_lat
    with pytest.raises(ValueError, match="'p_lon'.*not in the source"):
        _apply_null_geometry_policy(
            gdf,
            FlatGeobufParams(
                null_geometry="point_from", null_geometry_point_from=("p_lon", "p_lat")
            ),
        )


def test_null_geometry_point_from_builds_real_points_from_named_columns():
    gdf = _gdf_with_nulls_and_priors(200, null_every=4)
    out, dropped, sentinel, synthesized = _apply_null_geometry_policy(
        gdf,
        FlatGeobufParams(
            null_geometry="point_from", null_geometry_point_from=("p_lon", "p_lat")
        ),
    )
    assert dropped == 0
    assert sentinel == 0
    assert synthesized == 50
    assert len(out) == 200  # no row dropped
    assert not out.geometry.isna().any()
    assert int(out["null_geometry"].sum()) == 50

    synthesized_rows = out.loc[out["null_geometry"]]
    for _, row in synthesized_rows.iterrows():
        assert row.geometry.x == pytest.approx(row["p_lon"])
        assert row.geometry.y == pytest.approx(row["p_lat"])
    # The real rows keep their real (polygon) geometry, untouched.
    real = out.loc[~out["null_geometry"]]
    assert real.geom_equals_exact(gdf.loc[real.index].geometry, tolerance=0).all()


def test_null_geometry_point_from_raises_when_a_row_lacks_lon_lat_too():
    gdf = _gdf_with_nulls_and_priors(200, null_every=4)
    null_mask = gdf.geometry.isna()
    first_null_idx = gdf.index[null_mask][0]
    gdf.loc[first_null_idx, "p_lon"] = None

    with pytest.raises(ValueError, match=r"1 of 50 NULL-geometry features"):
        _apply_null_geometry_policy(
            gdf,
            FlatGeobufParams(
                null_geometry="point_from", null_geometry_point_from=("p_lon", "p_lat")
            ),
        )


def test_convert_with_point_from_writes_all_features_and_flags_synthesis(tmp_path):
    source = _geojson_source(tmp_path, _gdf_with_nulls_and_priors(200, null_every=4))
    target = str(tmp_path / "out.fgb")
    FlatGeobufAdapter().convert(
        source,
        target,
        {
            "spatial_index": True,
            "null_geometry": "point_from",
            "null_geometry_point_from": ["p_lon", "p_lat"],
        },
    )

    header = read_flatgeobuf_header(target)
    assert header.features_count == 200  # every feature kept

    ly = describe_flatgeobuf_layout(target, "lakes", features_synthesized=50)
    assert ly.num_features == 200
    assert ly.features_synthesized == 50
    assert ly.geometry_synthesized is True
    assert ly.features_dropped == 0
    assert ly.content_subset is False
    assert ly.features_sentinel == 0
    assert ly.geometry_fabricated is False


def test_convert_with_error_policy_fails_before_writing_any_bytes(tmp_path):
    source = _geojson_source(tmp_path, _gdf_with_nulls(200, null_every=4))
    target = str(tmp_path / "out.fgb")
    with pytest.raises(ValueError, match=r"50 of 200 features \(25%\)"):
        FlatGeobufAdapter().convert(
            source, target, {"spatial_index": True, "null_geometry": "error"}
        )
    import os

    assert not os.path.exists(target)
