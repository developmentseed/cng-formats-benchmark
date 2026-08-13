"""Tests for chunk-aware tile selection + layout rendering (requires `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("morecantile")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.metrics.display_tiles import (  # noqa: E402
    _read_zarr_grid,
    render_chunk_layout,
    render_zarr_chunk_layout,
    select_chunk_tiles,
    select_resolution_tiles,
    select_zarr_chunk_tiles,
    select_zarr_resolution_tiles,
)


@pytest.fixture
def cog_path(tmp_path):
    """A small, valid, overview-bearing COG with a known block size on disk."""
    path = tmp_path / "cog.tif"
    path.write_bytes(generate_cog_bytes(size=1024, blocksize=256, overview_levels=2))
    return str(path)


def test_select_chunk_tiles_returns_buckets_with_matching_counts(cog_path):
    tiles = select_chunk_tiles(cog_path)
    assert tiles, "expected at least one reachable chunk scenario"

    labels = {t.label for t in tiles}
    assert "1chunk" in labels  # a single-block tile is always reachable

    for t in tiles:
        target = int(t.label.removesuffix("chunk"))
        if t.approx:
            continue
        if target >= 9:
            assert t.chunks >= 9
        else:
            assert t.chunks == target
        assert t.z >= 0 and t.x >= 0 and t.y >= 0


def test_select_chunk_tiles_custom_targets(cog_path):
    tiles = select_chunk_tiles(cog_path, targets=(1,))
    assert [t.label for t in tiles] == ["1chunk"]


def test_render_chunk_layout_writes_png(cog_path, tmp_path):
    pytest.importorskip("matplotlib")
    tiles = select_chunk_tiles(cog_path)
    out = tmp_path / "layout.png"
    render_chunk_layout(cog_path, tiles, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


def test_render_chunk_layout_handles_empty_tiles(cog_path, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "empty.png"
    render_chunk_layout(cog_path, [], str(out))
    assert out.exists() and out.stat().st_size > 0


# --- GeoZarr store: same chunk-crossing geometry, read from the chunk grid -----


@pytest.fixture
def zarr_store(tmp_path):
    """A small sharded GeoZarr store with a real UTM CRS + transform."""
    pytest.importorskip("zarr")
    pytest.importorskip("xarray")
    import numpy as np
    from rasterio.crs import CRS

    from cng_benchmark.formats.geozarr import _write_sharded

    store = str(tmp_path / "g.zarr")
    data = (np.arange(1024 * 1024, dtype="uint16") % 1000).reshape(1024, 1024)
    _write_sharded(
        store,
        data,
        chunk=(256, 256),
        shard=(512, 512),
        codec="zstd",
        crs_wkt=CRS.from_epsg(32631).to_wkt(),
        # GDAL order: c a b f d e — origin (300000, 4900020), 10 m, north-up.
        geotransform="300000.0 10.0 0.0 4900020.0 0.0 -10.0",
    )
    return store


def test_select_zarr_chunk_tiles_returns_buckets(zarr_store):
    tiles = select_zarr_chunk_tiles(zarr_store)
    assert tiles, "expected at least one reachable chunk scenario"
    assert "1chunk" in {t.label for t in tiles}


def test_render_zarr_chunk_layout_writes_png(zarr_store, tmp_path):
    pytest.importorskip("matplotlib")
    tiles = select_zarr_chunk_tiles(zarr_store)
    out = tmp_path / "zlayout.png"
    render_zarr_chunk_layout(zarr_store, tiles, str(out))
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_ungeoreferenced_store_raises_clear_error(tmp_path):
    """A store with no GeoTransform fails with a clear message, not an unpack error."""
    pytest.importorskip("zarr")
    pytest.importorskip("xarray")
    import numpy as np

    from cng_benchmark.formats.geozarr import _write_sharded

    store = str(tmp_path / "plain.zarr")
    data = (np.arange(512 * 512, dtype="uint16") % 1000).reshape(512, 512)
    _write_sharded(store, data, chunk=(256, 256), shard=(512, 512), codec="none")
    with pytest.raises(RuntimeError, match="not georeferenced for display"):
        select_zarr_chunk_tiles(store)


# --- real pyramid discovery + resolution-targeted selection (#116) ----------


def test_read_zarr_grid_reads_real_decimations_for_a_uniform_pyramid(zarr_store):
    # The non-#114 case: `_write_sharded`'s own multiscales doc, still the
    # source of truth (not re-derived by walking group_keys()).
    grid = _read_zarr_grid(zarr_store)
    assert grid.native_resolution == pytest.approx(10.0)
    assert grid.decimations == [1, 2, 4]
    assert grid.level_paths == ["0", "1", "2"]


@pytest.fixture
def unified_bundle(tmp_path):
    """A #114-shaped multi-resolution bundle: real 10 m + 20 m tiers, plus a
    synthetic 60 m level -- every level a sibling top-level group at the
    store root, not nested under either component's own group."""
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    def _write_source(path, value, pixel_size, size=256):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="int16",
            crs="EPSG:32631",
            transform=from_origin(300000, 4900020, pixel_size, pixel_size),
        ) as dst:
            dst.write(np.full((size, size), value, dtype="int16"), 1)

    _write_source(str(tmp_path / "b2.tif"), 1, 10)
    _write_source(str(tmp_path / "clm_r2.tif"), 2, 20)
    sources = [
        SourceObject(name="b2", uri=str(tmp_path / "b2.tif")),
        SourceObject(name="clm_r2", uri=str(tmp_path / "clm_r2.tif")),
    ]
    store = str(tmp_path / "unified.zarr")
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        store,
        {
            "chunk_shape": [32, 32],
            "shard_shape": [64, 64],
            "multiscale_levels": [20, 60],
        },
    )
    return store, adapter


def test_read_zarr_grid_scopes_decimations_to_one_component_of_a_unified_bundle(
    unified_bundle,
):
    store, adapter = unified_bundle

    # b2 (10 m native): the whole ladder, 10 -> 20 -> 60 m.
    b2_grid = _read_zarr_grid(
        store, group=adapter.component_locator(store, "b2"), var_name="b2"
    )
    assert b2_grid.native_resolution == pytest.approx(10.0)
    assert b2_grid.decimations == [1, 2, 6]
    assert b2_grid.level_paths == ["0", "1", "2"]

    # clm_r2 (20 m native): its own chain starts at 20 m -- never the 10 m
    # level, where it has no data (the bug #116 found and fixed: walking
    # group_keys() under "1" would have found nothing at all, not even this).
    clm_grid = _read_zarr_grid(
        store, group=adapter.component_locator(store, "clm_r2"), var_name="clm_r2"
    )
    assert clm_grid.native_resolution == pytest.approx(20.0)
    assert clm_grid.decimations == [1, 3]  # relative to its OWN 20 m native
    assert clm_grid.level_paths == ["1", "2"]


def test_select_zarr_resolution_tiles_one_per_target_resolution(unified_bundle):
    store, adapter = unified_bundle

    tiles = select_zarr_resolution_tiles(
        store,
        target_resolutions=[10, 20, 60],
        group=adapter.component_locator(store, "b2"),
        var_name="b2",
    )
    assert [t.label for t in tiles] == ["res_10m", "res_20m", "res_60m"]
    assert [t.group for t in tiles] == ["0", "1", "2"]
    # Zoom strictly decreases as the target resolution coarsens.
    assert tiles[0].z > tiles[1].z > tiles[2].z


def test_select_zarr_resolution_tiles_clamps_to_a_components_own_native(
    unified_bundle,
):
    # clm_r2 has no 10 m data at all -- requesting it must clamp to its own
    # finest available level (20 m), never construct a request for a
    # resolution/level it doesn't have (the shape of request that trips a
    # real GeoZarrReader gap, verified separately against titiler.eopf).
    store, adapter = unified_bundle

    tiles = select_zarr_resolution_tiles(
        store,
        target_resolutions=[10, 20, 60],
        group=adapter.component_locator(store, "clm_r2"),
        var_name="clm_r2",
    )
    assert [t.label for t in tiles] == ["res_10m", "res_20m", "res_60m"]
    assert [t.group for t in tiles] == ["1", "1", "2"]  # 10m clamps to 20m's own group


@pytest.fixture
def unified_bundle_full_ladder(tmp_path):
    """A #114-shaped bundle on the real production ladder (#107/#108's S1/S2
    cross-mission chain, ``[20, 60, 120, 360, 720]`` off a 10 m native tier)
    with *several* components sharing each tier -- MAJA's real fan-out
    (several 10 m reflectance bands plus several 20 m masks, #118), not the
    one-component-per-tier shape every other fixture here uses."""
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from cng_benchmark.datasets.base import SourceObject
    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    def _write_source(path, value, pixel_size, size=256):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="int16",
            crs="EPSG:32631",
            transform=from_origin(300000, 4900020, pixel_size, pixel_size),
        ) as dst:
            dst.write(np.full((size, size), value, dtype="int16"), 1)

    fine_names = ["b2", "b3", "b4"]
    coarse_names = ["clm_r2", "edg_r2", "sat_r2"]
    for i, name in enumerate(fine_names):
        _write_source(str(tmp_path / f"{name}.tif"), i + 1, 10)
    for i, name in enumerate(coarse_names):
        _write_source(str(tmp_path / f"{name}.tif"), i + 10, 20)
    sources = [
        SourceObject(name=n, uri=str(tmp_path / f"{n}.tif"))
        for n in fine_names + coarse_names
    ]
    store = str(tmp_path / "unified-full.zarr")
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        store,
        {
            "chunk_shape": [32, 32],
            "shard_shape": [64, 64],
            "multiscale_levels": [20, 60, 120, 360, 720],
        },
    )
    return store, adapter


def test_select_zarr_resolution_tiles_full_ladder_requests_distinct_zooms(
    unified_bundle_full_ladder,
):
    # #118: whatever the `.group` a target resolution nominally maps to
    # (informational only once #121 lands -- it no longer drives either
    # router's query), what actually matters is that each target resolution
    # produces a genuinely different request: `z` comes straight from the
    # target resolution (`tms.zoom_for_res`), independent of `.group`, so
    # every one of the production ladder's 5 steps must land on its own
    # zoom -- multiple zooms really do get requested and measured.
    store, adapter = unified_bundle_full_ladder

    fine_tiles = select_zarr_resolution_tiles(
        store,
        target_resolutions=[20, 60, 120, 360, 720],
        group=adapter.component_locator(store, "b2"),
        var_name="b2",
    )
    assert [t.label for t in fine_tiles] == [
        "res_20m",
        "res_60m",
        "res_120m",
        "res_360m",
        "res_720m",
    ]
    fine_zooms = [t.z for t in fine_tiles]
    assert fine_zooms == sorted(fine_zooms, reverse=True)
    assert len(set(fine_zooms)) == len(fine_zooms)

    # Same for a component whose own chain is a slice of the root layout
    # (the shape #118 was filed against) -- still distinct zooms per
    # resolution, regardless of whatever `.group` it nominally reports.
    coarse_tiles = select_zarr_resolution_tiles(
        store,
        target_resolutions=[20, 60, 120, 360, 720],
        group=adapter.component_locator(store, "clm_r2"),
        var_name="clm_r2",
    )
    coarse_zooms = [t.z for t in coarse_tiles]
    assert coarse_zooms == sorted(coarse_zooms, reverse=True)
    assert len(set(coarse_zooms)) == len(coarse_zooms)


def test_select_resolution_tiles_cog_has_no_group(cog_path):
    # COG has no server-side group concept -- rio-tiler resolves its own
    # overview by zoom -- so every tile's `.group` stays None regardless of
    # target_resolutions.
    tiles = select_resolution_tiles(cog_path, target_resolutions=[10, 20, 40])
    assert tiles
    assert all(t.group is None for t in tiles)
    assert [t.label for t in tiles] == ["res_10m", "res_20m", "res_40m"]


def test_select_resolution_tiles_defaults_to_the_objects_own_decimations(zarr_store):
    # `target_resolutions=None` (unset) derives resolutions from the grid's
    # own decimations -- today's implicit per-format behaviour, for an arm
    # where cross-format resolution alignment isn't the point.
    tiles = select_zarr_resolution_tiles(zarr_store)
    assert [t.label for t in tiles] == ["res_10m", "res_20m", "res_40m"]
