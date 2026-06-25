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
    DEFAULT_SPAN,
    PIXC_SCHEME,
    CopcParams,
    _build_copc,
    _derive_max_depth,
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


def _copc(tmp_path, name="out.copc.laz", *, n=40_000, span=32, max_depth=4):
    target = str(tmp_path / name)
    x, y, z = _cloud(n)
    _build_copc(target, x, y, z, span=span, max_depth=max_depth)
    return target


def test_first_normalises():
    assert _first(None, 7) == 7
    assert _first([], 7) == 7
    assert _first([4, 5], 7) == 4
    assert _first(3, 7) == 3


def test_derive_max_depth_scales_with_density():
    # A cloud below the per-node budget stays a single level; a big one goes deeper.
    assert _derive_max_depth(10, span=128) == 1
    assert _derive_max_depth(100_000_000, span=32) > 1


def test_params_default_and_tolerate_extra_keys():
    opts = CopcParams.model_validate({"span": 64, "scope": "product-set"})
    assert opts.span == 64
    assert opts.max_depth is None  # None -> derived
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
    # The grid of boxes tiles the full extent, so every point is fetched in total.
    assert metrics["read_decoded_throughput"].detail["points"] == 40_000


def _pixc_netcdf(tmp_path, n=20_000, *, nan_first=False):
    """Write a synthetic PIXC pixel_cloud netCDF: geometry + a few point variables."""
    import xarray as xr

    lon, lat, height = _cloud(n)
    lon = lon / 1000.0
    lat = lat / 1000.0
    if nan_first:
        height = height.copy()
        height[0] = np.nan
    rng = np.random.default_rng(7)
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
    granule, _ = _pixc_netcdf(tmp_path, n, nan_first=True)
    target = str(tmp_path / "out.copc.laz")
    CopcAdapter().convert(
        f"{PIXC_SCHEME}{granule}::pixel_cloud", target, {"span": 32, "max_depth": 4}
    )

    ly = describe_copc_layout(target, "pixel_cloud")
    assert ly.point_count == n - 1  # the NaN point was dropped
    # Every non-geometry point variable is carried as an extra dimension; the
    # 'classification' name collides with a standard LAS dim and is suffixed.
    assert set(ly.extra_dimensions) == {
        "sig0",
        "water_frac",
        "classification_1",
        "geolocation_qual",
    }


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
