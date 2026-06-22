"""Tests for the tiling-layout collector (internal structure of an object)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.metrics.layout import describe_cog_layout  # noqa: E402


def _write(tmp_path, name: str, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_tiled_cog_layout(tmp_path):
    path = _write(tmp_path, "cog.tif", generate_cog_bytes(size=512, blocksize=128))
    ly = describe_cog_layout("cog", path, 123)
    assert ly.is_tiled is True
    assert (ly.block_width, ly.block_height) == (128, 128)
    assert ly.size_bytes == 123
    # 512/128 = 4 per side -> 16 internal tiles, and overviews are present.
    assert ly.internal_tiles == 16
    assert len(ly.overview_decimations) >= 1


def test_striped_geotiff_is_not_tiled(tmp_path):
    import numpy as np
    import rasterio

    path = str(tmp_path / "striped.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=1,
        dtype="uint8",
        tiled=False,  # striped: a block spans the full width
    ) as dst:
        dst.write(np.zeros((256, 256), dtype="uint8"), 1)

    ly = describe_cog_layout("striped", path, 999)
    assert ly.is_tiled is False
    assert ly.block_width == 256  # block spans the full raster width
