"""Tests for chunk-aware tile selection + layout rendering (requires `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("morecantile")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.metrics.display_tiles import (  # noqa: E402
    render_chunk_layout,
    render_zarr_chunk_layout,
    select_chunk_tiles,
    select_zarr_chunk_tiles,
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
