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


# --- Bundled multi-band writes (#102) --------------------------------------


def _write_band(
    path, *, value=1234, width=64, height=64, origin=(0.0, 1.0), nodata=None
):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(origin[0], origin[1], 0.01, 0.01),
        **({} if nodata is None else {"nodata": nodata}),
    ) as dst:
        dst.write(np.full((height, width), value, dtype="int16"), 1)


def test_convert_batch_writes_one_multiband_file(tmp_path):
    from cng_benchmark.datasets.base import SourceObject

    for name, value in [("wse", 10), ("sig0", 20), ("area", 30)]:
        _write_band(str(tmp_path / f"{name}.tif"), value=value)
    sources = [
        SourceObject(name=n, uri=str(tmp_path / f"{n}.tif"))
        for n in ("wse", "sig0", "area")
    ]

    target = tmp_path / "bundle.tif"
    adapter = CogAdapter()
    adapter.convert_batch(sources, str(target), {"block_size": 32})

    import rasterio

    with rasterio.open(target) as src:
        assert src.count == 3
        assert list(src.descriptions) == ["wse", "sig0", "area"]
        assert (src.read(1) == 10).all()
        assert (src.read(2) == 20).all()
        assert (src.read(3) == 30).all()

    # One physical object regardless of band count -- the dramatic side of
    # the fix for COG: N single-band files become 1.
    assert adapter.enumerate_objects(str(target)) == [target.stat().st_size]

    for name, band in [("wse", "1"), ("sig0", "2"), ("area", "3")]:
        assert adapter.component_locator(str(target), name) == band
    assert adapter.component_locator(str(target), "nonexistent") is None


def test_convert_batch_describe_layout_reports_one_layout_with_band_names(tmp_path):
    from cng_benchmark.datasets.base import SourceObject

    for name in ("wse", "sig0"):
        _write_band(str(tmp_path / f"{name}.tif"))
    sources = [
        SourceObject(name=n, uri=str(tmp_path / f"{n}.tif")) for n in ("wse", "sig0")
    ]

    target = tmp_path / "bundle.tif"
    adapter = CogAdapter()
    adapter.convert_batch(sources, str(target), {})

    (layout,) = adapter.describe_layout(str(target), name="ignored-for-batched")
    assert layout.band_names == ["wse", "sig0"]
    assert layout.size_bytes == target.stat().st_size

    # A non-batched target keeps today's single-layout, empty band_names shape.
    solo_target = tmp_path / "solo.tif"
    CogAdapter().convert(str(tmp_path / "wse.tif"), str(solo_target), {})
    (solo_layout,) = CogAdapter().describe_layout(str(solo_target), name="wse")
    assert solo_layout.band_names == []


def test_convert_batch_raises_on_a_mismatched_grid(tmp_path):
    from cng_benchmark.datasets.base import SourceObject

    _write_band(str(tmp_path / "wse.tif"))
    _write_band(str(tmp_path / "other.tif"), width=32, height=32)  # different grid
    sources = [
        SourceObject(name="wse", uri=str(tmp_path / "wse.tif")),
        SourceObject(name="other", uri=str(tmp_path / "other.tif")),
    ]

    target = tmp_path / "bundle.tif"
    with pytest.raises(ValueError, match="share one grid"):
        CogAdapter().convert_batch(sources, str(target), {})


def test_convert_batch_raises_on_mismatched_nodata(tmp_path):
    from cng_benchmark.datasets.base import SourceObject

    _write_band(str(tmp_path / "a.tif"), nodata=-1)
    _write_band(str(tmp_path / "b.tif"), nodata=-9999)
    sources = [
        SourceObject(name="a", uri=str(tmp_path / "a.tif")),
        SourceObject(name="b", uri=str(tmp_path / "b.tif")),
    ]

    target = tmp_path / "bundle.tif"
    with pytest.raises(ValueError, match="share one NODATA"):
        CogAdapter().convert_batch(sources, str(target), {})


def test_convert_batch_honours_component_nodata_override(tmp_path):
    # A component's own SourceObject.nodata (e.g. from a dataset reader that
    # knows a MAJA-style fill value the file itself doesn't declare) is what
    # every component must agree on -- and is what actually reaches the file.
    from cng_benchmark.datasets.base import SourceObject

    _write_band(str(tmp_path / "a.tif"))
    _write_band(str(tmp_path / "b.tif"))
    sources = [
        SourceObject(name="a", uri=str(tmp_path / "a.tif"), nodata=-10000),
        SourceObject(name="b", uri=str(tmp_path / "b.tif"), nodata=-10000),
    ]

    target = tmp_path / "bundle.tif"
    CogAdapter().convert_batch(sources, str(target), {})

    import rasterio

    with rasterio.open(target) as src:
        assert src.nodata == -10000
