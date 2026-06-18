"""Tests for chunk-aware tile selection + layout rendering (requires `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("morecantile")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.metrics.display_tiles import (  # noqa: E402
    render_chunk_layout,
    select_chunk_tiles,
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
