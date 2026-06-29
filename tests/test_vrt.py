"""Tests for the hand-built RGB mosaic VRT (requires the `cog` extra)."""

import pytest

pytest.importorskip("rasterio")

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from cng_benchmark.vrt import build_rgb_vrt_xml, read_grid  # noqa: E402

_RES = 10.0
_SIZE = 4
_CRS = "EPSG:32631"


def _write_tif(path, origin_x, origin_y, *, value=1, nodata=None):
    """Write a tiny single-band UInt16 GeoTIFF at the given upper-left origin."""
    transform = from_origin(origin_x, origin_y, _RES, _RES)
    data = np.full((_SIZE, _SIZE), value, dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_SIZE,
        width=_SIZE,
        count=1,
        dtype="uint16",
        crs=_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def test_vrt_has_three_rgb_bands(tmp_path):
    red = read_grid(_write_tif(tmp_path / "r.tif", 300000, 4500000, value=10))
    green = read_grid(_write_tif(tmp_path / "g.tif", 300000, 4500000, value=20))
    blue = read_grid(_write_tif(tmp_path / "b.tif", 300000, 4500000, value=30))

    xml = build_rgb_vrt_xml([[red], [green], [blue]])
    vrt_path = tmp_path / "rgb.vrt"
    vrt_path.write_text(xml)

    with rasterio.open(vrt_path) as src:
        assert src.count == 3
        assert [ci.name for ci in src.colorinterp] == ["red", "green", "blue"]
        assert (src.width, src.height) == (_SIZE, _SIZE)
        # Each band reads back its source's value (a window read works).
        assert int(src.read(1)[0, 0]) == 10
        assert int(src.read(2)[0, 0]) == 20
        assert int(src.read(3)[0, 0]) == 30


def test_vrt_mosaics_sources_into_the_union_grid(tmp_path):
    # Two tiles side by side (4 px * 10 m = 40 m apart) → an 8-px-wide union.
    a = read_grid(_write_tif(tmp_path / "a.tif", 300000, 4500000, value=1))
    b_x = 300000 + _SIZE * _RES
    b = read_grid(_write_tif(tmp_path / "b.tif", b_x, 4500000, value=2))

    xml = build_rgb_vrt_xml([[a, b], [a, b], [a, b]])
    vrt_path = tmp_path / "mosaic.vrt"
    vrt_path.write_text(xml)

    with rasterio.open(vrt_path) as src:
        assert src.width == 2 * _SIZE
        assert src.height == _SIZE
        # The union origin and resolution come through unchanged.
        assert src.transform.c == pytest.approx(300000)
        assert src.transform.f == pytest.approx(4500000)
        assert src.transform.a == pytest.approx(_RES)
        band = src.read(1)
        assert int(band[0, 0]) == 1  # left tile
        assert int(band[0, _SIZE]) == 2  # right tile, placed by its DstRect


def test_vrt_carries_nodata(tmp_path):
    red = read_grid(_write_tif(tmp_path / "r.tif", 300000, 4500000, nodata=0))
    green = read_grid(_write_tif(tmp_path / "g.tif", 300000, 4500000, nodata=0))
    blue = read_grid(_write_tif(tmp_path / "b.tif", 300000, 4500000, nodata=0))

    xml = build_rgb_vrt_xml([[red], [green], [blue]])
    assert "<NoDataValue>0</NoDataValue>" in xml
    vrt_path = tmp_path / "rgb.vrt"
    vrt_path.write_text(xml)
    with rasterio.open(vrt_path) as src:
        assert src.nodata == 0


def test_build_rgb_vrt_requires_three_bands(tmp_path):
    red = read_grid(_write_tif(tmp_path / "r.tif", 300000, 4500000))
    with pytest.raises(ValueError, match="3 bands"):
        build_rgb_vrt_xml([[red], [red]])
