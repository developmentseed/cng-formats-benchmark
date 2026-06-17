"""Tests for synthetic fixture generation (requires the `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402


def test_generated_fixture_is_a_valid_tiled_cog(tmp_path):
    data = generate_cog_bytes(size=256, blocksize=128)
    assert len(data) > 0

    cog_path = tmp_path / "fixture.tif"
    cog_path.write_bytes(data)

    import rasterio
    from rio_cogeo.cogeo import cog_validate

    is_valid, errors, _ = cog_validate(str(cog_path))
    assert is_valid, errors

    with rasterio.open(cog_path) as src:
        assert src.count == 3
        assert src.width == 256 and src.height == 256
        assert src.block_shapes[0] == (128, 128)  # internally tiled
        assert src.overviews(1)  # has overviews
