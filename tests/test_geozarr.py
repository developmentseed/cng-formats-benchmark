"""Tests for the GeoZarr v3 (2D, per-component) adapter.

The store-writing core (chunk/shard lever, enumerate, layout) is exercised on
synthetic in-memory arrays with only zarr + xarray + numpy, so it runs in CI. The
``convert`` source-read path needs rioxarray and is guarded with ``importorskip``.
"""

import pytest

pytest.importorskip("zarr")
pytest.importorskip("xarray")
np = pytest.importorskip("numpy")

from cng_benchmark.formats.geozarr import (  # noqa: E402
    DATA_VAR,
    GeoZarrParams,
    _fit_shard,
    _spatial_pair,
    _write_sharded,
    describe_store_layout,
    enumerate_store_objects,
)


def _store(tmp_path, name="g.zarr", **kw):
    store = str(tmp_path / name)
    data = (np.arange(2048 * 2048, dtype="uint16") % 1000).reshape(2048, 2048)
    opts = dict(chunk=(512, 512), shard=(1024, 1024), codec="zstd")
    opts.update(kw)
    _write_sharded(store, data, **opts)
    return store


def test_spatial_pair_normalises_shapes():
    # scalar -> square; swept list of shapes -> first; 3D -> trailing two;
    # 2D -> as is; fallback for empty/None.
    assert _spatial_pair(1024, (9, 9)) == (1024, 1024)
    assert _spatial_pair([[1, 2048, 2048], [1, 1024, 1024]], (9, 9)) == (2048, 2048)
    assert _spatial_pair([1, 2048, 1024], (9, 9)) == (2048, 1024)
    assert _spatial_pair([256, 512], (9, 9)) == (256, 512)
    assert _spatial_pair(None, (9, 9)) == (9, 9)
    assert _spatial_pair([], (9, 9)) == (9, 9)


def test_fit_shard_aligns_to_chunk_multiple_and_clamps():
    # A shard must be a whole multiple of the chunk and may not exceed the array.
    assert _fit_shard((1500, 1500), (512, 512), (2048, 2048)) == (1024, 1024)
    assert _fit_shard((4096, 4096), (512, 512), (2048, 2048)) == (2048, 2048)


def test_enumerate_returns_shard_data_excluding_metadata(tmp_path):
    store = _store(tmp_path)
    sizes = enumerate_store_objects(store)
    # 2048/1024 = 2 shards per side -> 4 shard objects, no zarr.json among them.
    assert len(sizes) == 4
    assert all(s > 0 for s in sizes)
    import os

    names = {f for _r, _d, fs in os.walk(store) for f in fs}
    assert "zarr.json" in names  # present in the store, excluded from enumeration


def test_describe_layout_reports_chunk_shard_codec(tmp_path):
    store = _store(tmp_path)
    ly = describe_store_layout(store, "FRE_B4")
    assert ly.kind == "geozarr"
    assert ly.name == "FRE_B4"
    assert ly.chunk_shape == [512, 512]
    assert ly.shard_shape == [1024, 1024]
    assert ly.chunks_per_shard == 4  # (1024/512) ** 2
    assert ly.codec == "zstd"
    assert ly.multiscale_levels == 0
    assert ly.shard_count == 4
    assert ly.size_bytes == sum(enumerate_store_objects(store))


def test_codec_none_is_uncompressed(tmp_path):
    store = _store(tmp_path, name="raw.zarr", codec="none")
    ly = describe_store_layout(store, "x")
    assert ly.codec == "none"


def test_multiscale_levels_build_a_pyramid(tmp_path):
    store = _store(tmp_path, name="ms.zarr", multiscale_levels=2)
    ly = describe_store_layout(store, "x")
    assert ly.multiscale_levels == 2
    # The base array plus two coarsened levels each add shard objects.
    assert ly.shard_count > 4


def test_unknown_codec_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown geozarr codec"):
        _store(tmp_path, name="bad.zarr", codec="lz4-nope")


def test_params_default_and_tolerate_extra_keys():
    opts = GeoZarrParams.model_validate({"codec": "zstd", "scope": "product-set"})
    assert opts.codec == "zstd"
    assert opts.multiscale_levels == 0


def test_convert_reads_a_raster_and_writes_a_store(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import rasterio
    from rasterio.transform import from_origin

    from cng_benchmark.formats.geozarr import GeoZarrAdapter

    source = str(tmp_path / "src.tif")
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=1,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(300000, 4900020, 10, 10),
    ) as dst:
        band = (np.arange(1024 * 1024, dtype="uint16") % 1000).reshape(1024, 1024)
        dst.write(band, 1)

    target = str(tmp_path / "out.zarr")
    GeoZarrAdapter().convert(
        source, target, {"chunk_shape": [256, 256], "shard_shape": [512, 512]}
    )
    ly = describe_store_layout(target, "B4")
    assert ly.chunk_shape == [256, 256]
    assert ly.shard_shape == [512, 512]
    assert ly.shard_count == 4  # 1024/512 = 2 per side

    # The store round-trips through the zarr-native read collector.
    from cng_benchmark.metrics.read import measure_zarr_read

    metrics = {m.name: m.value for m in measure_zarr_read(target, role="sink")}
    assert metrics["read_window_count"] >= 1
    assert metrics["read_decoded_throughput"] > 0


def test_finest_array_is_readable_by_name(tmp_path):
    # The single-level store exposes the data variable at the root.
    import zarr

    store = _store(tmp_path)
    group = zarr.open_group(store, mode="r")
    assert DATA_VAR in group
