"""Tests for the GeoZarr v3 (2D, per-component) adapter.

The store-writing core (chunk/shard lever, enumerate, layout) is exercised on
synthetic in-memory arrays with only zarr + xarray + numpy, so it runs in CI. The
``convert`` source-read path needs rioxarray and is guarded with ``importorskip``.
"""

import pytest

pytest.importorskip("zarr")
pytest.importorskip("xarray")
np = pytest.importorskip("numpy")

from cng_benchmark.formats import geozarr_multiscales as ms  # noqa: E402
from cng_benchmark.formats.geozarr import (  # noqa: E402
    DATA_VAR,
    GeoZarrParams,
    _fit_shard,
    _shard_data_files,
    _spatial_pair,
    _write_sharded,
    describe_store_layout,
    enumerate_store_objects,
    finest_level_group,
)

#: A real UTM 31N WKT — the CRS a MAJA tile carries. Verbatim (rather than
#: derived from rasterio) so the metadata tests run without the geo stack, and
#: nested: the inner geographic `AUTHORITY["EPSG","4326"]` must not be mistaken
#: for the CRS's own code.
UTM31N_WKT = (
    'PROJCS["WGS 84 / UTM zone 31N",GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",3],'
    'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32631"]]'
)
#: 10 m pixels with their origin at a MAJA-like tile corner (GDAL order).
GEOTRANSFORM = "300000.0 10.0 0.0 4900020.0 0.0 -10.0"


def _store(tmp_path, name="g.zarr", **kw):
    store = str(tmp_path / name)
    data = (np.arange(2048 * 2048, dtype="uint16") % 1000).reshape(2048, 2048)
    opts = dict(
        chunk=(512, 512),
        shard=(1024, 1024),
        codec="zstd",
        crs_wkt=UTM31N_WKT,
        geotransform=GEOTRANSFORM,
    )
    opts.update(kw)
    _write_sharded(store, data, **opts)
    return store


def _flat_store(tmp_path, name="flat.zarr", **kw):
    """A store with the pyramid switched off — the single-level comparison case."""
    return _store(tmp_path, name, multiscale_levels=0, **kw)


def test_spatial_pair_normalises_shapes():
    # scalar -> square; swept list of shapes -> first; 3D -> trailing two;
    # 2D -> as is; fallback for empty/None.
    assert _spatial_pair(1024, (9, 9)) == (1024, 1024)
    assert _spatial_pair([[1, 2048, 2048], [1, 1024, 1024]], (9, 9)) == (2048, 2048)
    assert _spatial_pair([1, 2048, 1024], (9, 9)) == (2048, 1024)
    assert _spatial_pair([256, 512], (9, 9)) == (256, 512)
    assert _spatial_pair(None, (9, 9)) == (9, 9)
    assert _spatial_pair([], (9, 9)) == (9, 9)


def test_fit_shard_aligns_to_chunk_multiple_and_clamps():
    # A shard must be a whole multiple of the chunk and may not exceed the array.
    assert _fit_shard((1500, 1500), (512, 512), (2048, 2048)) == (1024, 1024)
    assert _fit_shard((4096, 4096), (512, 512), (2048, 2048)) == (2048, 2048)


def test_enumerate_returns_shard_data_including_coordinate_chunks(tmp_path):
    store = _flat_store(tmp_path)
    sizes = enumerate_store_objects(store)
    # 2048/1024 = 2 shards per side -> 4 data shard objects, plus the shared
    # x and y coordinate arrays' own chunk (one each, well under a default
    # chunk) -- real physical objects the store puts on S3, not excluded
    # (#102: undercounting them was the harness/reality mismatch the issue
    # is about; CNES's own S3 inspection counts them too).
    assert len(sizes) == 6
    assert all(s > 0 for s in sizes)
    import os

    names = {f for _r, _d, fs in os.walk(store) for f in fs}
    assert "zarr.json" in names  # present in the store, excluded from enumeration


def test_shard_data_files_scoped_to_one_array_excludes_coordinates(tmp_path):
    # Scoped to the data array (what describe_layout's per-array size uses),
    # the coordinate arrays' chunks are excluded -- they are shared overhead,
    # never any one array's own cost.
    store = _flat_store(tmp_path, name="coords.zarr")
    scoped = _shard_data_files(store, var_name=DATA_VAR)
    assert scoped
    assert all(f"/{DATA_VAR}/c/" in p for p in scoped)
    assert len(scoped) < len(_shard_data_files(store))  # unscoped also finds x/y


def test_describe_layout_reports_chunk_shard_codec(tmp_path):
    store = _flat_store(tmp_path)
    ly = describe_store_layout(store, "FRE_B4")
    assert ly.kind == "geozarr"
    assert ly.name == "FRE_B4"
    assert ly.chunk_shape == [512, 512]
    assert ly.shard_shape == [1024, 1024]
    assert ly.chunks_per_shard == 4  # (1024/512) ** 2
    assert ly.codec == "zstd"
    assert ly.multiscale_levels == 0
    assert ly.shard_count == 4  # this array's own shards only
    assert ly.overview_bytes == 0
    assert ly.grid_group is None
    # size_bytes is the array's own bytes; enumerate_store_objects additionally
    # counts the (shared, non-attributable) x/y coordinate chunks, so the two
    # are no longer equal now that coordinate chunks are real objects (#102).
    assert ly.size_bytes < sum(enumerate_store_objects(store))


def test_codec_none_is_uncompressed(tmp_path):
    store = _flat_store(tmp_path, name="raw.zarr", codec="none")
    ly = describe_store_layout(store, "x")
    assert ly.codec == "none"


def test_multiscale_levels_build_a_pyramid(tmp_path):
    store = _store(tmp_path, name="ms.zarr", multiscale_levels=2)
    ly = describe_store_layout(store, "x")
    assert ly.multiscale_levels == 2
    # The base array plus two coarsened levels each add shard objects.
    assert ly.shard_count > 4


def test_unknown_codec_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown geozarr codec"):
        _store(tmp_path, name="bad.zarr", codec="lz4-nope")


def test_params_default_and_tolerate_extra_keys():
    opts = GeoZarrParams.model_validate({"codec": "zstd", "scope": "product-set"})
    assert opts.codec == "zstd"
    # A pyramid by default: an arm that omits the lever still gets overviews.
    assert opts.multiscale_levels == "auto"


def test_convert_reads_a_raster_and_writes_a_store(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import rasterio
    from rasterio.transform import from_origin

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=1,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(300000, 4900020, 10, 10),
    ) as dst:
        band = (np.arange(1024 * 1024, dtype="uint16") % 1000).reshape(1024, 1024)
        dst.write(band, 1)

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source,
        target,
        {"chunk_shape": [256, 256], "shard_shape": [512, 512], "multiscale_levels": 0},
    )
    ly = describe_store_layout(target, "B4")
    assert ly.chunk_shape == [256, 256]
    assert ly.shard_shape == [512, 512]
    assert ly.shard_count == 4  # 1024/512 = 2 per side

    # The store round-trips through the zarr-native read collector.
    from cng_benchmark.metrics.read import measure_zarr_read

    metrics = {m.name: m.value for m in measure_zarr_read(target, role="sink")}
    assert metrics["read_window_count"] >= 1
    assert metrics["read_latency_spread"] >= 0
    assert metrics["read_decoded_throughput"] > 0


def test_finest_array_is_readable_by_name(tmp_path):
    # The single-level store exposes the data variable at the root.
    import zarr

    store = _flat_store(tmp_path)
    group = zarr.open_group(store, mode="r")
    assert DATA_VAR in group


def _nodata_store(tmp_path, name, **kw):
    """A store whose int16 data holds MAJA's -10000 fill in its first column."""
    store = str(tmp_path / name)
    data = np.full((256, 256), 1234, dtype="int16")
    data[:, 0] = -10000
    _write_sharded(store, data, chunk=(128, 128), shard=(256, 256), codec="zstd", **kw)
    return store


def test_fill_value_is_declared_both_as_zarr_fill_and_cf_attribute(tmp_path):
    # `fill_value` alone leaves the no-data undeclared to a CF reader; the
    # `_FillValue` attribute is what xarray/rioxarray/GDAL actually mask on.
    import zarr

    store = _nodata_store(tmp_path, "nodata.zarr", fill_value=-10000.0)
    arr = zarr.open_group(store, mode="r")[DATA_VAR]
    assert arr.fill_value == -10000
    assert arr.attrs["_FillValue"] == -10000


def test_no_fill_value_defaults_to_zero(tmp_path):
    import zarr

    store = str(tmp_path / "default.zarr")
    data = np.zeros((256, 256), dtype="uint16")
    _write_sharded(store, data, chunk=(128, 128), shard=(256, 256), codec="zstd")
    arr = zarr.open_group(store, mode="r")[DATA_VAR]
    assert arr.fill_value == 0
    assert "_FillValue" not in arr.attrs
    assert "scale_factor" not in arr.attrs


def test_scale_factor_is_a_cf_attribute_and_leaves_the_counts_verbatim(tmp_path):
    # The stored DNs must stay int16 and byte-comparable with the COG arm; only
    # a decoding reader sees physical reflectance.
    import xarray as xr
    import zarr

    store = _nodata_store(
        tmp_path, "scaled.zarr", fill_value=-10000.0, scale_factor=1e-4
    )

    arr = zarr.open_group(store, mode="r")[DATA_VAR]
    assert arr.attrs["scale_factor"] == 1e-4
    assert arr.dtype == np.dtype("int16")
    assert arr[0, 1] == 1234  # raw DN, not rescaled on write

    decoded = xr.open_zarr(store, consolidated=False)[DATA_VAR]
    assert decoded.values[0, 1] == pytest.approx(0.1234)  # physical reflectance
    assert np.isnan(decoded.values[0, 0])  # the fill masked out


def test_multiscale_coarsening_excludes_the_fill_value(tmp_path):
    # Averaging -10000 as though it were data drags every overview pixel that
    # touches a swath edge towards the fill; GDAL excludes nodata when it builds
    # COG overviews, so the GeoZarr arm must too.
    import zarr

    store = _nodata_store(
        tmp_path, "pyramid.zarr", fill_value=-10000.0, multiscale_levels=1
    )
    level1 = zarr.open_group(store, mode="r")["1"][DATA_VAR]

    # Column 0 of level 1 averages a 2x2 block of one fill + one valid column.
    assert level1[0, 0] == 1234
    assert level1[0, 5] == 1234


def test_multiscale_coarsening_keeps_all_fill_blocks_as_fill(tmp_path):
    import zarr

    store = str(tmp_path / "allfill.zarr")
    data = np.full((256, 256), 1234, dtype="int16")
    data[:, :2] = -10000  # a full 2x2 block-column of fill
    _write_sharded(
        store,
        data,
        chunk=(128, 128),
        shard=(256, 256),
        codec="zstd",
        fill_value=-10000.0,
        multiscale_levels=1,
    )
    level1 = zarr.open_group(store, mode="r")["1"][DATA_VAR]
    assert level1[0, 0] == -10000


# --- the default pyramid and its metadata (#71) -----------------------------


def _root_attrs(store) -> dict:
    import zarr

    return dict(zarr.open_group(store, mode="r").attrs)


def test_auto_depth_coarsens_by_two_down_to_about_one_tile():
    # The rule GDAL follows for COG overviews: keep halving while the shorter
    # side stays at or above a tile, so the two arms get comparable pyramids.
    assert ms.auto_depth((2048, 2048)) == 3  # 2048 -> 1024 -> 512 -> 256
    assert ms.auto_depth((10980, 10980)) == 5  # a MAJA 10 m band
    assert ms.auto_depth((256, 256)) == 0  # already one tile
    assert ms.auto_depth((4096, 300)) == 0  # bounded by the shorter side
    assert ms.auto_depth((512, 512), min_dimension=128) == 2


def test_a_store_carries_a_pyramid_by_default(tmp_path):
    # The regression #71 is about: an arm that sets no lever used to get a flat,
    # full-resolution store, which made the tile server downsample on every
    # zoomed-out tile and made the display comparison meaningless.
    import zarr

    store = _store(tmp_path, name="default.zarr")
    group = zarr.open_group(store, mode="r")

    assert sorted(group.group_keys(), key=int) == ["0", "1", "2", "3"]
    assert [group[k][DATA_VAR].shape for k in ("0", "1", "2", "3")] == [
        (2048, 2048),
        (1024, 1024),
        (512, 512),
        (256, 256),
    ]
    assert describe_store_layout(store, "x").multiscale_levels == 3


def test_each_level_georeferences_on_its_own(tmp_path):
    # A coarser level covers the same ground with bigger cells. Copying the
    # native transform onto every level would mislocate each overview by its
    # decimation factor.
    import zarr

    group = zarr.open_group(_store(tmp_path, name="gt.zarr"), mode="r")
    x0, y0 = 300000.0, 4900020.0

    assert group["0"].attrs["spatial:transform"] == [10.0, 0.0, x0, 0.0, -10.0, y0]
    assert group["1"].attrs["spatial:transform"] == [20.0, 0.0, x0, 0.0, -20.0, y0]
    assert group["2"].attrs["spatial:transform"] == [40.0, 0.0, x0, 0.0, -40.0, y0]
    # The extent is the same at every level; only the cells grow.
    assert group["3"].attrs["spatial:bbox"] == group["0"].attrs["spatial:bbox"]
    # The CRS travels on every level too, so an overview stands alone.
    assert group["3"].attrs["proj:code"] == "EPSG:32631"


def test_no_cf_grid_mapping_shadows_the_conventions(tmp_path):
    # The conventions are the only place the georeferencing lives: no
    # `spatial_ref` variable duplicating the CRS and transform, no
    # `grid_mapping` pointing at one, and no `_ARRAY_DIMENSIONS` — Zarr v3 names
    # the axes itself, in `dimension_names`.
    import json
    import os

    import zarr

    store = _store(tmp_path, name="nocf.zarr")
    group = zarr.open_group(store, mode="r")

    assert "spatial_ref" not in list(group["0"].array_keys())
    for node in (group, group["1"], group["1"][DATA_VAR], group["1"]["x"]):
        assert "_ARRAY_DIMENSIONS" not in node.attrs
        assert "grid_mapping" not in node.attrs
        assert "GeoTransform" not in node.attrs

    meta = json.load(open(os.path.join(store, "1", DATA_VAR, "zarr.json")))
    assert meta["dimension_names"] == ["y", "x"]


def test_root_declares_the_multiscales_layout(tmp_path):
    import zarr_cm

    attrs = _root_attrs(_store(tmp_path, name="layout.zarr"))

    # The declared conventions are `zarr_cm`'s to define — including which
    # revision's schema each entry pins.
    uuids = {c["uuid"] for c in attrs["zarr_conventions"]}
    assert uuids == {
        zarr_cm.multiscales.UUID,
        zarr_cm.spatial.UUID,
        zarr_cm.proj.UUID,
    }
    zarr_cm.multiscales.validate(attrs["multiscales"])

    layout = attrs["multiscales"]["layout"]
    assert [e["asset"] for e in layout] == ["0", "1", "2", "3"]
    # Level 0 is the source of the chain and derives from nothing.
    assert "derived_from" not in layout[0]
    assert [e["derived_from"] for e in layout[1:]] == ["0", "1", "2"]
    # `scale` is relative to the level it was derived from, so it is 2 at every
    # step — not the absolute decimation.
    assert [e["transform"]["scale"] for e in layout[1:]] == [[2.0, 2.0]] * 3
    assert attrs["multiscales"]["resampling_method"] == "average"
    assert all(e["resampling_method"] == "average" for e in layout[1:])


def test_layout_entries_carry_each_level_absolute_position(tmp_path):
    # `transform` is the relative step between levels; where a level actually
    # sits belongs outside it, per the convention.
    layout = _root_attrs(_store(tmp_path, name="abs.zarr"))["multiscales"]["layout"]

    assert [e["spatial:shape"] for e in layout] == [
        [2048, 2048],
        [1024, 1024],
        [512, 512],
        [256, 256],
    ]
    # Rasterio/affine coefficient order [a, b, c, d, e, f]: pixel size first,
    # origin third and sixth — *not* GDAL's [c, a, b, f, d, e]. The origin is
    # the same on every level; only the cell size doubles.
    x0, y0 = 300000.0, 4900020.0
    assert layout[0]["spatial:transform"] == [10.0, 0.0, x0, 0.0, -10.0, y0]
    assert layout[1]["spatial:transform"] == [20.0, 0.0, x0, 0.0, -20.0, y0]


def test_root_declares_the_native_crs_and_grid(tmp_path):
    # Native CRS, not web mercator: nothing is reprojected to build the pyramid.
    attrs = _root_attrs(_store(tmp_path, name="crs.zarr"))

    assert attrs["proj:code"] == "EPSG:32631"
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:shape"] == [2048, 2048]
    assert attrs["spatial:registration"] == "pixel"
    assert attrs["spatial:bbox"] == [300000.0, 4879540.0, 320480.0, 4900020.0]


def test_the_store_metadata_reads_back_as_declared_conventions(tmp_path):
    # The conformance check, in the terms the conventions define: every
    # attribute the group carries belongs to a convention it declares, each
    # document validates, and nothing is left dangling.
    import zarr_cm

    attrs = _root_attrs(_store(tmp_path, name="valid.zarr"))
    remaining, extracted = zarr_cm.extract_many(
        attrs, ["multiscales", "spatial", "geo-proj"]
    )

    assert zarr_cm.multiscales.detect(attrs) is not None
    zarr_cm.multiscales.validate(extracted["multiscales"])
    zarr_cm.spatial.validate(extracted["spatial"])
    zarr_cm.proj.validate(extracted["geo-proj"])
    assert remaining == {}


def test_each_level_group_and_array_declare_what_they_use(tmp_path):
    # A convention's keys are only meaningful where the node declares it, and a
    # level's grid is its own — so the declaration travels all the way down to
    # the array, which is what a client ends up holding.
    import zarr
    import zarr_cm

    group = zarr.open_group(_store(tmp_path, name="decl.zarr"), mode="r")
    declared = {zarr_cm.spatial.UUID, zarr_cm.proj.UUID}

    for node in (group["1"], group["1"][DATA_VAR]):
        attrs = dict(node.attrs)
        assert {c["uuid"] for c in attrs["zarr_conventions"]} == declared
        assert attrs["spatial:dimensions"] == ["y", "x"]
        assert attrs["spatial:shape"] == [1024, 1024]
        assert attrs["proj:code"] == "EPSG:32631"
        _remaining, got = zarr_cm.extract_many(attrs, ["spatial", "geo-proj"])
        zarr_cm.spatial.validate(got["spatial"])
        zarr_cm.proj.validate(got["geo-proj"])


def test_a_crs_without_an_epsg_code_travels_as_wkt():
    wkt = 'PROJCRS["Local grid",BASEGEOGCRS["Unknown"]]'
    assert ms.proj_attrs(wkt) == {"proj:wkt2": wkt}
    assert ms.proj_attrs("") == {}


def test_every_array_names_its_axes_and_quantity(tmp_path):
    import zarr

    store = _store(tmp_path, name="axes.zarr", standard_name="toa_reflectance")
    group = zarr.open_group(store, mode="r")["1"]

    data = group[DATA_VAR]
    # Which array dimensions are spatial, ordered Y then X — what resolves
    # `spatial:transform` against the array's axes.
    assert data.attrs["spatial:dimensions"] == ["y", "x"]
    assert data.attrs["standard_name"] == "toa_reflectance"
    assert group["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert group["x"].attrs["units"] == "m"


def test_levels_carry_cell_centre_coordinates(tmp_path):
    import zarr

    group = zarr.open_group(_store(tmp_path, name="coords.zarr"), mode="r")

    # The first 10 m cell's centre is half a pixel in from the tile corner.
    assert group["0"]["x"][0] == pytest.approx(300005.0)
    assert group["0"]["y"][0] == pytest.approx(4900015.0)
    # Level 1's cells are 20 m, so its first centre is 10 m in.
    assert group["1"]["x"][0] == pytest.approx(300010.0)
    assert group["1"]["x"][1] - group["1"]["x"][0] == pytest.approx(20.0)


def test_a_geographic_crs_gets_longitude_latitude_coordinates(tmp_path):
    import zarr

    wkt = 'GEOGCRS["WGS 84",ID["EPSG",4326]]'
    store = str(tmp_path / "geo.zarr")
    _write_sharded(
        store,
        np.zeros((256, 256), dtype="uint16"),
        chunk=(128, 128),
        shard=(256, 256),
        codec="zstd",
        crs_wkt=wkt,
        geotransform="-10.0 0.01 0.0 45.0 0.0 -0.01",
    )
    group = zarr.open_group(store, mode="r")
    assert group["x"].attrs["standard_name"] == "longitude"
    assert group["x"].attrs["units"] == "degrees_east"
    assert group["y"].attrs["standard_name"] == "latitude"
    assert group.attrs["proj:code"] == "EPSG:4326"


def test_a_rotated_grid_skips_coordinates_but_keeps_its_transform(tmp_path):
    # 1D coordinate variables cannot describe a sheared grid; the affine still can.
    import zarr

    store = str(tmp_path / "rot.zarr")
    _write_sharded(
        store,
        np.zeros((256, 256), dtype="uint16"),
        chunk=(128, 128),
        shard=(256, 256),
        codec="zstd",
        crs_wkt=UTM31N_WKT,
        geotransform="300000.0 10.0 2.0 4900020.0 3.0 -10.0",
    )
    group = zarr.open_group(store, mode="r")
    assert "x" not in list(group.array_keys())
    # [a, b, c, d, e, f]: the rotation terms survive where 1D coordinates cannot.
    x0, y0 = 300000.0, 4900020.0
    assert group.attrs["spatial:transform"] == [10.0, 2.0, x0, 3.0, -10.0, y0]


def test_bbox_bounds_a_rotated_grid_by_all_four_corners():
    # A sheared grid's other diagonal can fall outside the box the first one
    # makes, so bounding by two corners understates the extent. Here the corner
    # at (0, h) is the southernmost and (w, 0) the northernmost — neither is on
    # the (0,0)–(w,h) diagonal.
    gt = (300000.0, 10.0, 2.0, 4900020.0, 3.0, -10.0)

    assert ms.bounds(gt, (256, 256)) == [300000.0, 4897460.0, 303072.0, 4900788.0]
    # An unrotated grid is unaffected: the diagonal already bounds it.
    north_up = (300000.0, 10.0, 0.0, 4900020.0, 0.0, -10.0)
    assert ms.bounds(north_up, (256, 256)) == [300000.0, 4897460.0, 302560.0, 4900020.0]


def test_overview_bytes_account_for_the_pyramid_alone(tmp_path):
    # Overviews cost bytes, the same way a COG's do; the report quotes this to
    # keep the size story readable next to the display one.
    pyramid = describe_store_layout(_store(tmp_path, name="p.zarr"), "p")
    flat = describe_store_layout(_flat_store(tmp_path, name="f.zarr"), "f")

    assert flat.overview_bytes == 0
    assert 0 < pyramid.overview_bytes < pyramid.size_bytes
    assert pyramid.size_bytes - pyramid.overview_bytes == flat.size_bytes


def test_compression_ratio_measures_compression_not_the_pyramid(tmp_path):
    # The stored size counts every level, so the uncompressed baseline must too
    # — otherwise the pyramid's cost would read as poor compression.
    pyramid = describe_store_layout(_store(tmp_path, name="p.zarr"), "p")
    flat = describe_store_layout(_flat_store(tmp_path, name="f.zarr"), "f")

    assert pyramid.compression_ratio == pytest.approx(flat.compression_ratio, rel=0.2)


def test_the_pyramid_is_visible_to_the_display_tile_selector(tmp_path):
    # The display metric picks tiles against the available decimations; the
    # levels are what make a zoomed-out tile cheap instead of a full-res read.
    pytest.importorskip("rasterio")
    pytest.importorskip("morecantile")
    from cng_benchmark.metrics.display_tiles import _read_zarr_grid

    grid = _read_zarr_grid(_store(tmp_path, name="disp.zarr"), role="sink")
    assert grid.decimations == [1, 2, 4, 8]
    assert (grid.width, grid.height) == (2048, 2048)


def test_the_read_metric_addresses_the_native_level(tmp_path):
    # A partial read must land on level 0, not on an overview — otherwise the
    # read metric would quietly compare full-resolution COG reads against
    # coarsened Zarr ones.
    from cng_benchmark.metrics.read import _open_zarr_array

    assert _open_zarr_array(_store(tmp_path, name="read.zarr"), "sink").shape == (
        2048,
        2048,
    )


def test_rioxarray_georeferences_every_level_from_the_conventions(tmp_path):
    # The store carries no CF grid mapping, so this is the whole georeferencing
    # story: rioxarray 0.22+ reads the `spatial` and `geo-proj` conventions.
    # Checked on the *array*, not just the dataset — xarray drops dataset
    # attributes when a variable is selected out, and a tile server addresses a
    # variable, so each array repeats its own grid.
    pytest.importorskip("rioxarray")
    import xarray as xr

    store = _store(tmp_path, name="rio.zarr")
    extent = (300000.0, 4879540.0, 320480.0, 4900020.0)
    for level, cell in (("0", 10.0), ("1", 20.0), ("3", 80.0)):
        da = xr.open_dataset(store, engine="zarr", group=level, consolidated=False)[
            DATA_VAR
        ]
        assert da.rio.crs.to_epsg() == 32631
        assert da.rio.transform()[0] == pytest.approx(cell)
        # Every level covers the same ground; only the cells grow.
        assert da.rio.bounds() == pytest.approx(extent)


def test_finest_level_group_finds_the_native_level(tmp_path):
    # A pyramid store: DATA_VAR lives under numbered level groups, "0" native.
    assert finest_level_group(_store(tmp_path, name="pyramid.zarr")) == "0"


def test_finest_level_group_is_none_for_a_flat_store(tmp_path):
    # No pyramid: DATA_VAR sits directly at the root, no group to select.
    assert finest_level_group(_flat_store(tmp_path)) is None


def test_a_pyramid_store_opens_as_a_datatree(tmp_path):
    # What a GeoZarr reader does with a multiscale store: open the hierarchy and
    # find every level under its own group, decoded and georeferenced.
    import xarray as xr

    tree = xr.open_datatree(
        _store(tmp_path, name="tree.zarr"), engine="zarr", consolidated=False
    )
    assert sorted(tree.children) == ["0", "1", "2", "3"]
    level1 = tree["1"].ds
    assert level1[DATA_VAR].dims == ("y", "x")
    assert level1[DATA_VAR].shape == (1024, 1024)
    assert float(level1["x"][0]) == pytest.approx(300010.0)


def _write_source(path, *, nodata=None, scales=None):
    """A minimal georeferenced int16 GeoTIFF baseline."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=1,
        dtype="int16",
        crs="EPSG:32631",
        transform=from_origin(300000, 4900020, 10, 10),
        **({} if nodata is None else {"nodata": nodata}),
    ) as dst:
        dst.write(np.full((256, 256), 1234, dtype="int16"), 1)
        if scales is not None:
            dst.scales = scales


def test_convert_propagates_nodata_and_scale_params(tmp_path):
    # The MAJA case: the source declares neither, the params carry both.
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    _write_source(source)

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source,
        target,
        {
            "chunk_shape": [128, 128],
            "shard_shape": [256, 256],
            "nodata": -10000.0,
            "scale_factor": 1e-4,
        },
    )
    arr = zarr.open_group(target, mode="r")[DATA_VAR]
    assert arr.fill_value == -10000
    assert arr.attrs["_FillValue"] == -10000
    assert arr.attrs["scale_factor"] == 1e-4


def test_convert_falls_back_to_what_the_source_declares(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    _write_source(source, nodata=-9999, scales=(0.01,))

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source, target, {"chunk_shape": [128, 128], "shard_shape": [256, 256]}
    )
    arr = zarr.open_group(target, mode="r")[DATA_VAR]
    assert arr.fill_value == -9999
    assert arr.attrs["scale_factor"] == pytest.approx(0.01)


def test_convert_omits_the_identity_scale_of_an_unpacked_source(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    _write_source(source)

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source, target, {"chunk_shape": [128, 128], "shard_shape": [256, 256]}
    )
    arr = zarr.open_group(target, mode="r")[DATA_VAR]
    assert "scale_factor" not in arr.attrs
    assert "add_offset" not in arr.attrs


# --- scale_offset codec arm (#54) -------------------------------------------


def _packed_store(tmp_path, name, **kw):
    """A store from int16 counts (1234 DN) with -10000 fill in the first column."""
    store = str(tmp_path / name)
    data = np.full((256, 256), 1234, dtype="int16")
    data[:, :64] = -10000
    _write_sharded(
        store,
        data,
        chunk=(128, 128),
        shard=(256, 256),
        codec="zstd",
        fill_value=-10000.0,
        scale_factor=1e-4,
        **kw,
    )
    return store


def test_scale_offset_codec_lands_in_the_array_pipeline(tmp_path):
    import zarr

    store = _packed_store(tmp_path, "codec.zarr", scale_offset=True)
    arr = zarr.open_group(store, mode="r")[DATA_VAR]

    names = [f.to_dict()["name"] for f in arr.filters]
    assert names == ["scale_offset", "cast_value"]
    # The array declares physical units; the packed integer is what is stored.
    assert arr.dtype == np.dtype("float32")
    assert arr.filters[1].to_dict()["configuration"]["data_type"] == "int16"


def test_scale_offset_gives_every_reader_physical_units(tmp_path):
    # The point of #54: a plain zarr reader — no CF decoding — already sees
    # reflectance, where the attribute arm hands back a raw count to unscale.
    import zarr

    codec = zarr.open_group(_packed_store(tmp_path, "c.zarr", scale_offset=True))
    attrs = zarr.open_group(_packed_store(tmp_path, "a.zarr", scale_offset=False))

    assert codec[DATA_VAR][0, 100] == pytest.approx(0.1234, abs=1e-6)
    assert attrs[DATA_VAR][0, 100] == 1234


def test_scale_offset_omits_the_cf_scale_attributes(tmp_path):
    # The codec already unpacks; a CF reader applying scale_factor on top would
    # scale a second time.
    import zarr

    arr = zarr.open_group(_packed_store(tmp_path, "codec.zarr", scale_offset=True))[
        DATA_VAR
    ]
    assert "scale_factor" not in arr.attrs
    assert "add_offset" not in arr.attrs


def test_scale_offset_states_the_fill_in_physical_units(tmp_path):
    import zarr

    arr = zarr.open_group(_packed_store(tmp_path, "codec.zarr", scale_offset=True))[
        DATA_VAR
    ]
    # -10000 DN x 1/10000 = -1.0 reflectance.
    assert arr.fill_value == pytest.approx(-1.0)
    assert arr[0, 0] == pytest.approx(-1.0)


def test_scale_offset_round_trips_the_full_dn_range(tmp_path):
    import zarr

    store = str(tmp_path / "rt.zarr")
    probe = np.arange(-10000, 10001, 7, dtype="int16")
    data = np.zeros((256, 256), dtype="int16")
    data.flat[: probe.size] = probe
    _write_sharded(
        store,
        data,
        chunk=(256, 256),
        shard=(256, 256),
        codec="zstd",
        scale_factor=1e-4,
        scale_offset=True,
    )
    back = zarr.open_group(store, mode="r")[DATA_VAR][:]
    recovered = np.rint(back.astype("float64") / 1e-4).astype("int16")
    assert np.array_equal(recovered.flat[: probe.size], probe)


def test_scale_offset_keeps_the_stored_size_and_ratio_comparable(tmp_path):
    # The codec must not inflate the payload: the declared dtype is float32 but
    # int16 reaches disk, so the size and compression ratio stay comparable with
    # the COG arm. Sizing "uncompressed" off the declared dtype would double it.
    codec = describe_store_layout(
        _packed_store(tmp_path, "c.zarr", scale_offset=True), "c"
    )
    attrs = describe_store_layout(
        _packed_store(tmp_path, "a.zarr", scale_offset=False), "a"
    )

    assert codec.scale_offset is True
    assert attrs.scale_offset is False
    assert codec.stored_dtype == "int16"
    assert attrs.stored_dtype == "int16"
    assert codec.size_bytes == attrs.size_bytes
    assert codec.compression_ratio == pytest.approx(attrs.compression_ratio)


def test_scale_offset_without_a_scale_factor_is_a_no_op(tmp_path):
    import zarr

    store = str(tmp_path / "noscale.zarr")
    _write_sharded(
        store,
        np.full((256, 256), 7, dtype="int16"),
        chunk=(128, 128),
        shard=(256, 256),
        codec="zstd",
        scale_offset=True,
    )
    arr = zarr.open_group(store, mode="r")[DATA_VAR]
    assert arr.dtype == np.dtype("int16")
    assert not arr.filters


def test_scale_offset_multiscale_coarsens_without_the_fill(tmp_path):
    import zarr

    store = _packed_store(tmp_path, "pyr.zarr", scale_offset=True, multiscale_levels=1)
    level1 = zarr.open_group(store, mode="r")["1"][DATA_VAR]
    assert level1[0, 100] == pytest.approx(0.1234, abs=1e-6)
    assert level1[0, 0] == pytest.approx(-1.0)  # an all-fill block stays fill


def test_convert_honours_the_scale_offset_param(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    _write_source(source)

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source,
        target,
        {
            "chunk_shape": [128, 128],
            "shard_shape": [256, 256],
            "nodata": -10000.0,
            "scale_factor": 1e-4,
            "scale_offset": True,
        },
    )
    arr = zarr.open_group(target, mode="r")[DATA_VAR]
    assert [f.to_dict()["name"] for f in arr.filters] == ["scale_offset", "cast_value"]
    assert arr[0, 0] == pytest.approx(0.1234, abs=1e-6)


# --- Bundled multi-component writes (#102) --------------------------------


def _write_source_at(
    path, *, value=1234, width=256, height=256, origin=(300000, 4900020)
):
    """A georeferenced int16 GeoTIFF at ``origin`` — same shape/CRS convention
    as :func:`_write_source`, but with a controllable grid so two calls can be
    made to share a grid (defaults) or deliberately not (different ``origin``
    or ``width``/``height``)."""
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="int16",
        crs="EPSG:32631",
        transform=from_origin(origin[0], origin[1], 10, 10),
    ) as dst:
        dst.write(np.full((height, width), value, dtype="int16"), 1)


def test_shard_data_files_var_name_scopes_to_one_array(tmp_path):
    # Write two arrays as siblings in one group (the shape a bundled store's
    # grid-group takes) and confirm var_name isolates each one's own shards,
    # while the unscoped call finds both -- and the coordinate arrays too.
    import xarray as xr

    store = str(tmp_path / "sibling.zarr")
    # Non-zero, non-default-fill values throughout: an all-fill chunk is
    # never written at all (zarr skips it), which would silently defeat this
    # test's file-count assertions.
    y = np.arange(1, 5, dtype="float64")
    x = np.arange(1, 5, dtype="float64")
    da_a = xr.DataArray(np.full((4, 4), 5, dtype="int16"), dims=("y", "x"))
    ds = xr.Dataset({"a": da_a}).assign_coords(y=("y", y), x=("x", x))
    ds.to_zarr(
        store,
        mode="w",
        group="grid0",
        zarr_format=3,
        consolidated=False,
        encoding={"a": {"chunks": (2, 2), "shards": (4, 4)}},
    )
    da_b = xr.DataArray(np.full((4, 4), 9, dtype="int16"), dims=("y", "x"))
    xr.Dataset({"b": da_b}).to_zarr(
        store,
        mode="a",
        group="grid0",
        zarr_format=3,
        consolidated=False,
        encoding={"b": {"chunks": (2, 2), "shards": (4, 4)}},
    )

    a_files = _shard_data_files(store, group="grid0", var_name="a")
    b_files = _shard_data_files(store, group="grid0", var_name="b")
    assert len(a_files) == 1 and len(b_files) == 1
    assert a_files != b_files

    unscoped = _shard_data_files(store)
    assert len(unscoped) == 4  # "a", "b", and the shared "x"/"y" coordinates


def test_convert_batch_bundles_components_sharing_a_grid(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    for name, value in [("wse", 10), ("sig0", 20), ("area", 30)]:
        _write_source_at(str(tmp_path / f"{name}.tif"), value=value)
    sources = [
        SourceObject(name=n, uri=str(tmp_path / f"{n}.tif"))
        for n in ("wse", "sig0", "area")
    ]

    target = str(tmp_path / "bundle.zarr")
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        target,
        {"chunk_shape": [64, 64], "shard_shape": [128, 128], "multiscale_levels": 0},
    )

    root = zarr.open_group(target, mode="r")
    assert list(root.group_keys()) == ["grid0"]
    grid0 = root["grid0"]
    assert set(grid0.array_keys()) == {"wse", "sig0", "area", "x", "y"}
    assert (grid0["wse"][:] == 10).all()
    assert (grid0["sig0"][:] == 20).all()
    assert (grid0["area"][:] == 30).all()
    assert root.attrs["cng_benchmark:components"] == {
        "wse": "grid0",
        "sig0": "grid0",
        "area": "grid0",
    }

    for name in ("wse", "sig0", "area"):
        assert adapter.component_locator(target, name) == "grid0"
    assert adapter.component_locator(target, "nonexistent") is None
    assert adapter.component_locator("/some/other/store", "wse") is None


def test_convert_batch_object_count_beats_per_component_conversion(tmp_path):
    # The whole point of #102: bundling must produce meaningfully fewer
    # physical shard objects than converting each component independently
    # would (each independent store would duplicate the x/y coordinate chunks).
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    names = [f"var{i}" for i in range(6)]
    for n in names:
        _write_source_at(str(tmp_path / f"{n}.tif"), value=1)
    sources = [SourceObject(name=n, uri=str(tmp_path / f"{n}.tif")) for n in names]

    bundled_target = str(tmp_path / "bundled.zarr")
    adapter = GeoZarrAdapter()
    params = {
        "chunk_shape": [64, 64],
        "shard_shape": [128, 128],
        "multiscale_levels": 0,
    }
    adapter.convert_batch(sources, bundled_target, params)
    bundled_count = len(adapter.enumerate_objects(bundled_target))

    per_component_count = 0
    for i, src in enumerate(sources):
        t = str(tmp_path / f"solo{i}.zarr")
        solo_adapter = GeoZarrAdapter()
        solo_adapter.convert(src.uri, t, params)
        per_component_count += len(solo_adapter.enumerate_objects(t))

    # 6 components independently: 6 * (4 data shards + 2 coordinate chunks) = 36.
    # Bundled: 6*4 = 24 data shards + one grid's own 2 coordinate chunks = 26.
    assert bundled_count < per_component_count


def test_convert_batch_splits_components_by_grid(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import zarr

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    _write_source_at(str(tmp_path / "wse.tif"), value=1)
    _write_source_at(str(tmp_path / "sig0.tif"), value=2)
    # A different shape -> a different grid, must not be merged into grid0.
    _write_source_at(str(tmp_path / "other.tif"), value=3, width=128, height=128)

    sources = [
        SourceObject(name="wse", uri=str(tmp_path / "wse.tif")),
        SourceObject(name="sig0", uri=str(tmp_path / "sig0.tif")),
        SourceObject(name="other", uri=str(tmp_path / "other.tif")),
    ]
    target = str(tmp_path / "multigrid.zarr")
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        target,
        {"chunk_shape": [32, 32], "shard_shape": [64, 64], "multiscale_levels": 0},
    )

    root = zarr.open_group(target, mode="r")
    assert sorted(root.group_keys()) == ["grid0", "grid1"]
    assert adapter.component_locator(target, "wse") == "grid0"
    assert adapter.component_locator(target, "sig0") == "grid0"
    assert adapter.component_locator(target, "other") == "grid1"
    assert set(root["grid0"].array_keys()) == {"wse", "sig0", "x", "y"}
    assert set(root["grid1"].array_keys()) == {"other", "x", "y"}


def test_describe_layout_reports_one_layout_per_bundled_component(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    for name in ("wse", "sig0"):
        _write_source_at(str(tmp_path / f"{name}.tif"))
    sources = [
        SourceObject(name=n, uri=str(tmp_path / f"{n}.tif")) for n in ("wse", "sig0")
    ]

    target = str(tmp_path / "bundle.zarr")
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        target,
        {"chunk_shape": [64, 64], "shard_shape": [128, 128], "multiscale_levels": 0},
    )

    layouts = adapter.describe_layout(target, name="ignored-for-batched")
    assert {ly.name for ly in layouts} == {"wse", "sig0"}
    for ly in layouts:
        assert ly.grid_group == "grid0"
        assert ly.kind == "geozarr"
        assert ly.size_bytes > 0
        assert ly.chunk_shape == [64, 64]

    # A non-batched target keeps today's single-layout, grid_group=None shape.
    solo_target = str(tmp_path / "solo.zarr")
    GeoZarrAdapter().convert(
        str(tmp_path / "wse.tif"),
        solo_target,
        {"chunk_shape": [64, 64], "shard_shape": [128, 128], "multiscale_levels": 0},
    )
    (solo_layout,) = GeoZarrAdapter().describe_layout(solo_target, name="wse")
    assert solo_layout.grid_group is None
