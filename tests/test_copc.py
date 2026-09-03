"""Tests for the COPC (point-cloud, single-file) adapter.

The octree builder, enumerate, layout (copclib) and the octree-node read metric
(laspy) are exercised on a synthetic in-memory point cloud — all pip wheels, so
they run in CI. The ``convert`` source-read paths (a SWOT PIXC netCDF group via
xarray/h5netcdf; a LAS/LAZ tile via laspy) are guarded with ``importorskip``.
"""

import pytest

pytest.importorskip("copclib")
pytest.importorskip("laspy")
np = pytest.importorskip("numpy")

from cng_benchmark.formats.copc import (  # noqa: E402
    _SAFETY_MAX_DEPTH,
    DEFAULT_SPAN,
    PIXC_SCHEME,
    CopcParams,
    _build_copc,
    _first,
    describe_copc_layout,
)


def _cloud(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(300000, 300500, n),
        rng.uniform(4900000, 4900500, n),
        rng.uniform(0, 100, n),
    )


def _copc(
    tmp_path, name="out.copc.laz", *, n=40_000, span=32, max_depth=4, extras=None
):
    target = str(tmp_path / name)
    x, y, z = _cloud(n)
    _build_copc(target, x, y, z, extras, span=span, max_depth=max_depth)
    return target


def test_first_normalises():
    assert _first(None, 7) == 7
    assert _first([], 7) == 7
    assert _first([4, 5], 7) == 4
    assert _first(3, 7) == 3


def test_octree_bounds_node_size_on_a_skewed_cloud(tmp_path):
    # Memory control: a heavily skewed cloud (most points in one small corner,
    # like PIXC lon/lat clustered against a wide height axis) must still bin into
    # nodes no larger than the per-node budget (span**3). That bound is what keeps
    # the content-complete build from materialising a large fraction of the cloud
    # in one node — the regression that OOM-killed the full-granule run.
    pytest.importorskip("laspy")

    rng = np.random.default_rng(0)
    n, span = 50_000, 8
    budget = span**3  # 512
    # 90% of the (all-distinct) points in a small corner of a wide domain.
    dense = rng.uniform(0.0, 0.1, size=(45_000, 3))
    sparse = rng.uniform(0.0, 1.0, size=(n - 45_000, 3))
    pts = np.vstack([dense, sparse])

    target = str(tmp_path / "skewed.copc.laz")
    _build_copc(
        target, pts[:, 0], pts[:, 1], pts[:, 2], span=span, max_depth=_SAFETY_MAX_DEPTH
    )

    ly = describe_copc_layout(target, "skewed")
    assert ly.point_count == n  # every point written, none dropped
    assert ly.num_nodes > 1  # subdivided, not dumped into one bucket
    assert ly.points_per_node <= budget  # no giant node -> bounded peak memory


def test_pixc_octree_root_is_a_spatial_sample_not_a_height_slice(tmp_path):
    # Issue #134: a PIXC group's raw lon/lat (degrees) sits next to height
    # (metres) -- Z's raw extent is ~10,000x LARGER than X/Y's before
    # projection. Un-reprojected, the octree cube comes out sized on height
    # alone, and the root node is `span` points from one narrow height band
    # (flat water), not a spatial sample of the tile. `_read_pixc_group`
    # reprojects lon/lat to a local UTM zone before the octree is built, so
    # the root must instead be a genuine, roughly footprint-covering sample.
    pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    laspy = pytest.importorskip("laspy")
    from cng_benchmark.formats.copc import CopcAdapter

    n = 20_000
    granule, source_ds = _pixc_netcdf(tmp_path, n)
    target = str(tmp_path / "root.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud", target, {"span": 32, "max_depth": 8}
    )

    reader = laspy.CopcReader.open(target)
    root = reader.query(level=0)
    assert len(root) > 32  # well above `span`, not a single height-band leaf

    full = reader.query()
    full_x_range = float(np.asarray(full.x).max() - np.asarray(full.x).min())
    full_y_range = float(np.asarray(full.y).max() - np.asarray(full.y).min())
    root_x_range = float(np.asarray(root.x).max() - np.asarray(root.x).min())
    root_y_range = float(np.asarray(root.y).max() - np.asarray(root.y).min())
    # The root's own footprint covers most of the full cloud's footprint --
    # a spatial overview, not points clustered from one narrow height slice.
    assert root_x_range >= 0.5 * full_x_range
    assert root_y_range >= 0.5 * full_y_range


def test_octree_level_growth_is_gradual_not_a_single_dump(tmp_path):
    # Issue #134's second defect: a node under the per-node budget used to be
    # written whole (no voxel sampling), so a "coarse" level could actually
    # hold nearly the entire cloud. On a uniform cloud, no single level should
    # add anywhere near the whole cloud once every node is voxel-sampled.
    laspy = pytest.importorskip("laspy")
    import copclib as copc

    rng = np.random.default_rng(4)
    n, span = 60_000, 8
    pts = rng.uniform(0, 1000, size=(n, 3))
    target = str(tmp_path / "uniform.copc.laz")
    _build_copc(target, pts[:, 0], pts[:, 1], pts[:, 2], span=span, max_depth=8)

    depth = copc.FileReader(target).GetMaxDepth()
    reader = laspy.CopcReader.open(target)
    level_counts = [len(reader.query(level=lvl)) for lvl in range(depth + 1)]
    assert sum(level_counts) == n  # every point accounted for, once
    assert depth >= 2  # this cloud is large enough to actually subdivide
    for count in level_counts:
        assert count <= 0.5 * n  # no single level dumps most of the cloud


def test_pixc_output_declares_a_crs_and_xy_round_trips(tmp_path):
    # Issue #134: the produced COPC used to declare no CRS at all
    # (`header.parse_crs()` was always `None`) since the octree was built on
    # raw, unprojected lon/lat. `_build_copc` now writes the UTM CRS
    # `_read_pixc_group` reprojected onto, so the file is self-describing,
    # and inverting that projection must land back on the source lon/lat.
    pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    laspy = pytest.importorskip("laspy")
    pyproj = pytest.importorskip("pyproj")
    from cng_benchmark.formats.copc import CopcAdapter

    n = 10_000
    granule, source_ds = _pixc_netcdf(tmp_path, n)
    target = str(tmp_path / "crs.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud", target, {"span": 32, "max_depth": 6}
    )

    # The CRS lives in the EVLR section (COPC keeps its large VLRs, the
    # hierarchy and, now, the WKT, there); `laspy.CopcReader`'s header does not
    # surface EVLRs at all (`.header.evlrs` is always `None`), so the CRS check
    # needs the plain reader. This is a real, complete round trip either way:
    # any standard LAS/LAZ reader (PDAL, GDAL, ...) reads a file's EVLRs.
    with laspy.open(target) as plain_reader:
        crs = plain_reader.header.parse_crs()
    assert crs is not None

    reader = laspy.CopcReader.open(target)
    full = reader.query()
    inverse = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_back, lat_back = inverse.transform(np.asarray(full.x), np.asarray(full.y))
    lon_source = np.asarray(source_ds["longitude"].values)
    lat_source = np.asarray(source_ds["latitude"].values)
    # Statistical round-trip (point order is not preserved by the octree
    # bucketing): inverting the output's own CRS lands back on the source
    # lon/lat footprint within a fraction of a metre (~1e-6 deg), not the
    # un-reprojected bug's ~0.01 deg (~1 km) footprint distortion.
    assert abs(float(lon_back.min()) - float(lon_source.min())) < 1e-6
    assert abs(float(lon_back.max()) - float(lon_source.max())) < 1e-6
    assert abs(float(lat_back.min()) - float(lat_source.min())) < 1e-6
    assert abs(float(lat_back.max()) - float(lat_source.max())) < 1e-6


def test_params_default_and_tolerate_extra_keys():
    opts = CopcParams.model_validate({"span": 64, "scope": "product-set"})
    assert opts.span == 64
    assert opts.max_depth is None  # None -> high safety cap; span bounds nodes
    assert opts.scale is None


def test_build_produces_one_file_object(tmp_path):
    import os

    from cng_benchmark.formats.copc import CopcAdapter

    target = _copc(tmp_path)
    sizes = CopcAdapter().enumerate_objects(target)
    assert sizes == [os.path.getsize(target)]  # a single addressable object
    assert sizes[0] > 0


def test_layout_reports_octree_and_preserves_all_points(tmp_path):
    target = _copc(tmp_path, n=40_000, span=32, max_depth=4)
    ly = describe_copc_layout(target, "pixel_cloud")
    assert ly.kind == "copc"
    assert ly.name == "pixel_cloud"
    # Every input point lands in exactly one node, so the cloud round-trips.
    assert ly.point_count == 40_000
    assert ly.num_nodes > 1  # a real octree, not a single bucket
    assert 1 <= ly.max_depth <= 4
    assert 0 < ly.points_per_node <= 40_000
    assert ly.compression_ratio > 1.0  # the LAS point block is LASzip-compressed


def test_octree_lever_changes_node_structure(tmp_path):
    # A smaller per-node span (tighter budget) forces more, smaller nodes.
    coarse = describe_copc_layout(
        _copc(tmp_path, name="coarse.copc.laz", span=64, max_depth=6), "x"
    )
    fine = describe_copc_layout(
        _copc(tmp_path, name="fine.copc.laz", span=16, max_depth=6), "x"
    )
    assert fine.num_nodes > coarse.num_nodes
    assert fine.points_per_node < coarse.points_per_node


def test_octree_node_read_metric_round_trips(tmp_path):
    target = _copc(tmp_path)
    from cng_benchmark.metrics.read import measure_copc_read

    metrics = {m.name: m for m in measure_copc_read(target, role="sink", queries=4)}
    assert metrics["read_query_count"].value == 4
    assert metrics["read_latency_mean"].value >= 0
    assert metrics["read_latency_spread"].value >= 0
    # Random (seeded) boxes keep the grid's per-box footprint but overlap/gap
    # differently, so the total is no longer an exact full-extent tiling.
    assert 0 < metrics["read_decoded_throughput"].detail["points"] <= 40_000 * 4


def _pixc_netcdf(tmp_path, n=20_000, *, nan_first=False):
    """Write a synthetic PIXC pixel_cloud netCDF: geometry + a few point variables.

    lon/lat are realistic geographic degrees (a ~1 km tile footprint) and height
    is realistic metres -- deliberately mismatched in numeric scale (0.01 deg
    against 100 m), the exact shape that exposed the octree-cube-sized-on-height
    defect (issue #134): a naive cube built from ``max(x_extent, y_extent,
    z_extent)`` on unprojected lon/lat next to metric height comes out driven by
    height alone. `_read_pixc_group` reprojects lon/lat to a local UTM zone
    before the octree is built, so this fixture is what that reprojection has
    to correct for the fix to be tested honestly, not sidestepped by picking
    lon/lat magnitudes that happen not to trigger it.
    """
    import xarray as xr

    rng = np.random.default_rng(7)
    lon = 1.40 + rng.uniform(0, 0.01, n)
    lat = 43.60 + rng.uniform(0, 0.01, n)
    height = rng.uniform(0, 100, n).astype("float64")
    if nan_first:
        height = height.copy()
        height[0] = np.nan
    ds = xr.Dataset(
        {
            "longitude": ("points", lon),
            "latitude": ("points", lat),
            "height": ("points", height.astype("float32")),
            "sig0": ("points", rng.uniform(0, 1, n).astype("float32")),
            "water_frac": ("points", rng.uniform(0, 1, n).astype("float64")),
            # 'classification' collides with a standard LAS pf6 dimension name.
            "classification": ("points", rng.integers(0, 5, n).astype("uint8")),
            "geolocation_qual": ("points", rng.integers(-3, 3, n).astype("int16")),
        }
    )
    granule = str(tmp_path / "SWOT_L2_HR_PIXC_048.nc")
    ds.to_netcdf(granule, group="pixel_cloud", engine="h5netcdf")
    return granule, ds


def test_convert_carries_all_pixc_point_variables(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    from cng_benchmark.formats.copc import CopcAdapter

    n = 20_000
    granule, source_ds = _pixc_netcdf(tmp_path, n, nan_first=True)
    target = str(tmp_path / "out.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud", target, {"span": 32, "max_depth": 4}
    )

    ly = describe_copc_layout(target, "pixel_cloud")
    assert ly.point_count == n - 1  # the NaN point was dropped
    # Every non-geometry point variable is carried; 'classification' is a
    # standard dimension of the target format too, so it lands in that native
    # field, not as a renamed extra dimension (only names with no home in the
    # target format become extras).
    assert set(ly.extra_dimensions) == {
        "sig0",
        "water_frac",
        "geolocation_qual",
    }

    import laspy

    back = laspy.CopcReader.open(target).query()
    # The native 'classification' field actually carries the source values
    # (not left empty by a name collision with a renamed extra dimension).
    expected_classification = np.asarray(source_ds["classification"].values)[1:]
    assert set(np.unique(np.asarray(back.classification))) <= set(
        np.unique(expected_classification)
    )
    assert np.asarray(back.classification).max() > 0


def test_extra_dimension_values_and_dtypes_round_trip(tmp_path):
    pytest.importorskip("laspy")
    from cng_benchmark.formats.copc import _build_copc

    n = 15_000
    x, y, z = _cloud(n)
    rng = np.random.default_rng(3)
    extras = {
        "sig0": rng.uniform(0, 1, n).astype("float32"),
        "qual": rng.integers(-100, 100, n).astype("int16"),
        "pid": np.arange(n, dtype="uint32"),  # unique id to pair points
    }
    # _build_copc consumes ``extras`` to keep peak memory down, so snapshot the
    # expected values before the call.
    expected = {k: v.copy() for k, v in extras.items()}
    target = str(tmp_path / "rt.copc.laz")
    _build_copc(target, x, y, z, extras, span=32, max_depth=4)

    import laspy

    back = laspy.CopcReader.open(target).query()
    order = np.argsort(np.asarray(back["pid"]))
    assert np.array_equal(np.asarray(back["pid"])[order], np.arange(n))
    # float32 stays float32 and int16 stays int16 — values preserved exactly.
    assert np.asarray(back["sig0"]).dtype == np.float32
    assert np.array_equal(np.asarray(back["sig0"])[order], expected["sig0"])
    assert np.array_equal(
        np.asarray(back["qual"])[order].astype("int16"), expected["qual"]
    )


def test_include_and_exclude_select_the_carried_set(tmp_path):
    pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    pytest.importorskip("h5py")
    from cng_benchmark.formats.copc import CopcAdapter

    granule, _ = _pixc_netcdf(tmp_path, 8_000)
    inc = str(tmp_path / "inc.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud?include=sig0,geolocation_qual",
        inc,
        {"span": 32},
    )
    assert set(describe_copc_layout(inc, "x").extra_dimensions) == {
        "sig0",
        "geolocation_qual",
    }

    exc = str(tmp_path / "exc.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud?exclude=water_frac", exc, {"span": 32}
    )
    assert "water_frac" not in describe_copc_layout(exc, "x").extra_dimensions
    assert "sig0" in describe_copc_layout(exc, "x").extra_dimensions


def test_default_span_is_a_per_node_budget():
    assert DEFAULT_SPAN**3 > 1  # span**3 is the per-node point budget


def test_render_copc_lod_writes_png(tmp_path):
    # The octree level-of-detail figure is a systematic structural artifact of a
    # COPC run (the point-cloud analogue of the COG chunk-layout image).
    pytest.importorskip("matplotlib")
    import os

    from cng_benchmark.formats.copc import render_copc_lod

    target = _copc(tmp_path, n=30_000, span=16, max_depth=6)
    out = str(tmp_path / "lod.png")
    assert render_copc_lod(target, out) == out
    assert os.path.getsize(out) > 0


def _rgb_extras(n, seed=7):
    # LAS RGB is nominally 16-bit; use the full range so auto-detection has
    # something real to normalise against.
    rng = np.random.default_rng(seed)
    return {
        "red": rng.integers(0, 65536, n, dtype="uint16"),
        "green": rng.integers(0, 65536, n, dtype="uint16"),
        "blue": rng.integers(0, 65536, n, dtype="uint16"),
    }


def test_render_copc_lod_color_by_auto_uses_rgb_when_carried(tmp_path):
    # A flat colour makes a photogrammetric/colour-LiDAR cloud unreadable as a
    # cloud (no terrain, no structure); auto should reach for the cloud's own
    # colour when it has one, the CO3D/CARS case (issue: LOD figure request).
    pytest.importorskip("matplotlib")
    import os

    from cng_benchmark.formats.copc import render_copc_lod

    target = _copc(tmp_path, n=20_000, span=16, max_depth=5, extras=_rgb_extras(20_000))
    out = str(tmp_path / "lod_rgb.png")
    assert render_copc_lod(target, out) == out
    assert os.path.getsize(out) > 0


def test_render_copc_lod_color_by_auto_falls_back_to_height_without_rgb(tmp_path):
    pytest.importorskip("matplotlib")
    import os

    from cng_benchmark.formats.copc import render_copc_lod

    target = _copc(tmp_path, n=20_000, span=16, max_depth=5)  # no extras -> no RGB
    out = str(tmp_path / "lod_z.png")
    assert render_copc_lod(target, out) == out
    assert os.path.getsize(out) > 0


def test_render_copc_lod_color_by_a_carried_extra_dimension(tmp_path):
    pytest.importorskip("matplotlib")
    import os

    from cng_benchmark.formats.copc import render_copc_lod

    n = 20_000
    rng = np.random.default_rng(3)
    extras = {"scan_angle_rank": rng.uniform(-30, 30, n).astype("float32")}
    target = _copc(tmp_path, n=n, span=16, max_depth=5, extras=extras)
    out = str(tmp_path / "lod_dim.png")
    assert render_copc_lod(target, out, color_by="scan_angle_rank") == out
    assert os.path.getsize(out) > 0


def test_render_copc_lod_color_by_rgb_without_rgb_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from cng_benchmark.formats.copc import render_copc_lod

    target = _copc(tmp_path, n=5_000, span=16, max_depth=3)  # geometry only
    with pytest.raises(ValueError, match="rgb"):
        render_copc_lod(target, str(tmp_path / "lod.png"), color_by="rgb")


def test_render_copc_lod_color_by_unknown_dimension_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from cng_benchmark.formats.copc import render_copc_lod

    target = _copc(tmp_path, n=5_000, span=16, max_depth=3)
    with pytest.raises(ValueError, match="not a dimension"):
        render_copc_lod(target, str(tmp_path / "lod.png"), color_by="not_a_real_dim")


def test_render_copc_lod_dot_count_scales_with_panel_point_share(tmp_path, monkeypatch):
    # Issue #134's third defect: every panel used to draw up to the same fixed
    # `max_points` dots regardless of how many points it actually held, so a
    # sparse coarse level looked as dense as the full cloud. A skewed cloud
    # (dense core + sparse halo) forces multiple octree levels with clearly
    # different point counts, so the coarse panel's dot budget must come out
    # smaller than the full-detail panel's.
    pytest.importorskip("matplotlib")
    import matplotlib.axes

    from cng_benchmark.formats.copc import render_copc_lod

    rng = np.random.default_rng(1)
    dense = rng.uniform(0, 50, size=(40_000, 3))
    sparse = rng.uniform(0, 500, size=(4_000, 3))
    pts = np.vstack([dense, sparse])
    target = str(tmp_path / "skew.copc.laz")
    _build_copc(target, pts[:, 0], pts[:, 1], pts[:, 2], span=16, max_depth=6)

    scatter_sizes: list[int] = []
    orig_scatter = matplotlib.axes.Axes.scatter

    def spy(self, x, *a, **kw):
        scatter_sizes.append(len(x))
        return orig_scatter(self, x, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy)
    render_copc_lod(target, str(tmp_path / "lod.png"), max_points=2000, color_by="flat")

    assert len(scatter_sizes) >= 2
    assert scatter_sizes[0] < scatter_sizes[-1]  # coarse panel is visibly sparser


def _cars_tile(tmp_path, name="0_0.laz", n=20_000):
    """Write a synthetic CARS tile: geometry + colour + a CARS-style extra dim."""
    import laspy

    x, y, z = _cloud(n)
    rng = np.random.default_rng(11)
    # Point format 3 carries RGB — CARS colours its cloud from the source image.
    header = laspy.LasHeader(point_format=3)
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.intensity = rng.integers(0, 4000, n).astype("uint16")
    las.red = rng.integers(0, 65535, n).astype("uint16")
    las.green = rng.integers(0, 65535, n).astype("uint16")
    las.blue = rng.integers(0, 65535, n).astype("uint16")
    las.confidence = rng.uniform(0, 1, n).astype("float32")
    tile = str(tmp_path / name)
    las.write(tile)
    return tile


def test_convert_carries_a_cars_tile_point_record(tmp_path):
    # The CO3D CARS arm: a delivered LAZ tile → one COPC, reusing the PIXC
    # adapter path. The conversion is content-complete — the tile's colour,
    # intensity and its own extra dimensions travel with the geometry, so the
    # produced object's size is a like-for-like basis for comparison with the
    # source tile, not a geometry-only fraction.
    from cng_benchmark.formats.copc import CopcAdapter

    n = 20_000
    tile = _cars_tile(tmp_path, n=n)
    target = str(tmp_path / "out.copc.laz")
    CopcAdapter().convert(tile, target, {"span": 32, "max_depth": 4})

    ly = describe_copc_layout(target, "0_0")
    assert ly.point_count == n
    carried = set(ly.extra_dimensions)
    # RGB has no home in the target format (LAS 1.4 point format 6 carries no
    # colour), so it is a genuine extra dimension, same as the tile's own
    # 'confidence'. 'intensity' IS a standard dimension of the target format,
    # so it lands in that native field instead, not as a renamed extra.
    assert {"red", "green", "blue", "confidence"} <= carried
    assert "intensity" not in carried
    assert "intensity_1" not in carried
    # The raw scaled-integer geometry is not carried twice.
    assert carried.isdisjoint({"X", "Y", "Z"})


def test_cars_tile_values_round_trip(tmp_path):
    import laspy

    from cng_benchmark.formats.copc import CopcAdapter

    tile = _cars_tile(tmp_path, name="1_0.laz", n=5_000)
    source = laspy.read(tile)
    target = str(tmp_path / "rt.copc.laz")
    CopcAdapter().convert(tile, target, {"span": 16, "max_depth": 4})

    back = laspy.CopcReader.open(target).query()
    # Pair source and produced points by their (unique) confidence value: the
    # octree reorders points, so compare per point, not by position.
    order = np.argsort(np.asarray(back.confidence))
    expected = np.argsort(np.asarray(source.confidence))
    assert np.array_equal(np.asarray(back.red)[order], np.asarray(source.red)[expected])
    # 'intensity' is a standard dimension of the target format, so it round-trips
    # through that native field, not a renamed extra dimension.
    assert np.array_equal(
        np.asarray(back.intensity)[order], np.asarray(source.intensity)[expected]
    )
    # The geometry survives the octree's own coordinate quantisation (the COPC is
    # written on a scale derived from the cloud extent, not the tile's 0.01 step).
    assert np.allclose(
        np.sort(np.asarray(back.z)), np.sort(np.asarray(source.z)), atol=1e-4
    )


def test_object_size_resolves_pixc_scheme(tmp_path):
    # The point-cloud component URI wraps the granule; object_size sizes the
    # underlying granule so the write metric records bytes_in for the PIXC arm.
    from cng_benchmark import storage

    granule = tmp_path / "SWOT_L2_HR_PIXC_048.nc"
    granule.write_bytes(b"x" * 4321)
    uri = f"{PIXC_SCHEME}{granule}::pixel_cloud?include=sig0,water_frac"
    assert storage.object_size(uri) == 4321
