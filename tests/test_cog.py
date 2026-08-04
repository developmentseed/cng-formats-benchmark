"""Tests for the COG adapter conversion + enumeration (requires the `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.formats.cog import CogAdapter  # noqa: E402


@pytest.fixture
def source_raster(tmp_path):
    """A small valid raster on disk to use as a conversion baseline."""
    path = tmp_path / "source.tif"
    path.write_bytes(generate_cog_bytes(size=256, blocksize=256))
    return path


def test_convert_produces_a_valid_tiled_cog(source_raster, tmp_path):
    target = tmp_path / "out.tif"
    CogAdapter().convert(str(source_raster), str(target), {"block_size": 128})

    import rasterio
    from rio_cogeo.cogeo import cog_validate

    is_valid, errors, _ = cog_validate(str(target))
    assert is_valid, errors
    with rasterio.open(target) as src:
        assert src.block_shapes[0] == (128, 128)  # grouping lever applied


def test_enumerate_objects_returns_single_file_size(source_raster, tmp_path):
    target = tmp_path / "out.tif"
    adapter = CogAdapter()
    adapter.convert(str(source_raster), str(target), {})
    assert adapter.enumerate_objects(str(target)) == [target.stat().st_size]


def test_convert_defaults_to_deflate_codec(source_raster, tmp_path):
    target = tmp_path / "out.tif"
    CogAdapter().convert(str(source_raster), str(target), {})

    import rasterio

    with rasterio.open(target) as src:
        assert src.profile.get("compress", "").lower() == "deflate"


def test_convert_codec_param_selects_the_compression(source_raster, tmp_path):
    # The matched-codec arm (#72): COG must be able to run with zstd, like the
    # GeoZarr arm's `codec` param, so the size gap is attributable to the format
    # rather than a codec mismatch.
    target = tmp_path / "out.tif"
    CogAdapter().convert(str(source_raster), str(target), {"codec": "zstd"})

    import rasterio

    with rasterio.open(target) as src:
        assert src.profile.get("compress", "").lower() == "zstd"


def test_describe_layout_reports_the_configured_codec(source_raster, tmp_path):
    target = tmp_path / "out.tif"
    adapter = CogAdapter()
    adapter.convert(str(source_raster), str(target), {"codec": "zstd"})
    layouts = adapter.describe_layout(str(target))
    assert layouts[0].codec == "zstd"


def test_convert_nodata_param_is_written_to_produced_cog(tmp_path):
    # MAJA S2 sources don't declare nodata in the file header; `params['nodata']`
    # lets the benchmark config inject the known fill value (-10000).
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    src_path = tmp_path / "src.tif"
    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 0.01, 0.01),
    ) as dst:
        dst.write(np.zeros((1, 64, 64), dtype="int16"))

    target = tmp_path / "out.tif"
    CogAdapter().convert(str(src_path), str(target), {"nodata": -10000})
    with rasterio.open(target) as src:
        assert src.nodata == -10000
