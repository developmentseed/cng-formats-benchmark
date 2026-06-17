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
