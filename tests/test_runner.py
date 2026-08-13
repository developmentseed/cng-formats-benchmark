"""Tests for the orchestration runner (independent of the CLI)."""

import numpy as np
import pytest

from cng_benchmark import __version__
from cng_benchmark.config import load_benchmark_config
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.runner import (
    _measure_display_object,
    _safe_display_metrics,
    _safe_read_metrics,
    _tool_versions,
    run_benchmark,
    run_conversion_benchmark,
    tier_policy_from_config,
)

BENCHMARK_EXAMPLE = "configs/benchmarks/example_cog.yaml"
SYNTHETIC = "configs/benchmarks/synthetic_cog.yaml"


def test_run_benchmark_populates_run_context():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    sizes = [10, 20, 30, 40]
    run = run_benchmark(cfg, sizes)

    assert run.dataset_id == cfg.dataset
    assert run.format_id == cfg.formats[0]
    assert run.tool_versions["cng_benchmark"] == __version__
    assert "grouping_lever" in run.params

    expected = profile_object_sizes(sizes, tier_policy_from_config(cfg.tiers))
    assert run.object_profile == expected
    assert {m.name for m in run.metrics} == {"object_count", "total_bytes"}


def test_run_benchmark_selects_named_format():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    run = run_benchmark(cfg, [1, 2, 3], format_id="geozarr")
    assert run.format_id == "geozarr"
    assert "Zarr v3" in run.params["grouping_lever"]


def test_run_benchmark_unknown_format_raises():
    cfg = load_benchmark_config(BENCHMARK_EXAMPLE)
    with pytest.raises(KeyError):
        run_benchmark(cfg, [1, 2, 3], format_id="nonesuch")


def test_run_conversion_benchmark_local_publishes_and_collects(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=256, blocksize=256))
    output = tmp_path / "out"

    # Local end-to-end without services: write/object_size/read (no display).
    cfg = load_benchmark_config(SYNTHETIC).model_copy(
        update={"metrics": ["write", "object_size", "read"]}
    )
    run = run_conversion_benchmark(cfg, str(source), str(output))

    assert run.format_id == "cog"
    assert run.object_profile.count == 1
    names = {m.name for m in run.metrics}
    assert {"write_elapsed", "object_count", "read_window_count"} <= names
    # The produced object is always published under the output location.
    assert (output / "cog" / "cog.tif").exists()


def test_run_conversion_benchmark_display_skips_without_endpoint(tmp_path):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    from cng_benchmark.fixtures import generate_cog_bytes

    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=128, blocksize=128))
    cfg = load_benchmark_config(SYNTHETIC).model_copy(update={"metrics": ["display"]})
    # A missing endpoint is caught and surfaced as a skipped metric — the run
    # completes rather than aborting, even in the single-object path.
    run = run_conversion_benchmark(cfg, str(source), str(tmp_path / "out"))
    names = {m.name for m in run.metrics}
    assert "display_skipped" in names
    skipped = next(m for m in run.metrics if m.name == "display_skipped")
    assert "TiTiler endpoint" in skipped.detail["error"]


def test_measure_display_object_cog_locator_selects_band(tmp_path, monkeypatch):
    # A bundled COG's component addresses a specific band (#102) via TiTiler's
    # `bidx` query param -- regression test for the query-construction gap
    # left when the locator plumbing landed (verified live against docker-
    # compose titiler that a missing `bidx` silently always reads band 1).
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    import cng_benchmark.runner as _runner
    from cng_benchmark.fixtures import generate_cog_bytes
    from cng_benchmark.formats.cog import CogAdapter

    source = tmp_path / "bundle.tif"
    source.write_bytes(generate_cog_bytes(size=64, blocksize=64))

    captured = {}

    def _fake_measure_display(endpoint, uri, tiles, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(_runner, "measure_display", _fake_measure_display)

    cfg = load_benchmark_config(SYNTHETIC)
    _measure_display_object(
        cfg,
        CogAdapter(),
        str(source),
        "s3://bucket/bundle.tif",
        str(tmp_path),
        "http://titiler.example",
        locator="2",
        name="compB",
    )

    assert captured["extra_query"] == {"bidx": "2"}


def test_measure_display_object_cog_no_locator_omits_bidx(tmp_path, monkeypatch):
    pytest.importorskip("rasterio")
    pytest.importorskip("rio_cogeo")
    import cng_benchmark.runner as _runner
    from cng_benchmark.fixtures import generate_cog_bytes
    from cng_benchmark.formats.cog import CogAdapter

    source = tmp_path / "single.tif"
    source.write_bytes(generate_cog_bytes(size=64, blocksize=64))

    captured = {}

    def _fake_measure_display(endpoint, uri, tiles, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(_runner, "measure_display", _fake_measure_display)

    cfg = load_benchmark_config(SYNTHETIC)
    _measure_display_object(
        cfg,
        CogAdapter(),
        str(source),
        "s3://bucket/single.tif",
        str(tmp_path),
        "http://titiler.example",
    )

    assert captured["extra_query"] is None


def test_measure_display_object_zarr_query_group_is_fixed_not_per_tile(
    tmp_path, monkeypatch
):
    # #121: the stock /zarr router's query must name one *fixed* group (its
    # native/canonical level -- titiler.xarray needs some group to locate a
    # nested variable at all, and this writer nests even the native level
    # under its own integer group whenever the store carries any pyramid),
    # the same for every tile regardless of which resolution is being timed
    # -- not the #116/#117 per-tile override that swapped in a *different*,
    # resolution-appropriate group per scenario, crediting the router with
    # multiscale awareness it doesn't have.
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import rasterio
    from rasterio.transform import from_origin

    import cng_benchmark.runner as _runner
    from cng_benchmark.formats.geozarr import DATA_VAR, GeoZarrAdapter

    source = tmp_path / "source.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=1,
        dtype="int16",
        crs="EPSG:32631",
        transform=from_origin(300000, 4900020, 10, 10),
    ) as dst:
        dst.write(np.full((256, 256), 1234, dtype="int16"), 1)

    target = tmp_path / "store.zarr"
    GeoZarrAdapter().convert(
        str(source),
        str(target),
        {"chunk_shape": [64, 64], "shard_shape": [128, 128], "multiscale_levels": 1},
    )

    captured_calls = []

    def _fake_measure_display(endpoint, uri, tiles, **kwargs):
        captured_calls.append(kwargs)
        return []

    monkeypatch.setattr(_runner, "measure_display", _fake_measure_display)

    cfg = load_benchmark_config("configs/benchmarks/synthetic_geozarr.yaml")
    _measure_display_object(
        cfg,
        GeoZarrAdapter(),
        str(target),
        "s3://bucket/store.zarr",
        str(tmp_path),
        "http://titiler.example",
    )
    assert len(captured_calls) == 1
    # Native level 0 -- fixed, never a per-tile override.
    assert captured_calls[0]["extra_query"] == {"variable": DATA_VAR, "group": "0"}
    assert captured_calls[0]["group_query_key"] is None

    # Same for a run with resolution-coverage scenarios configured -- #121's
    # bug specifically compounded there via the per-tile override, which
    # would have swapped `group` to a coarser level for each res_*m tile.
    captured_calls.clear()
    cfg = cfg.model_copy(
        update={"params": {**cfg.params, "display_target_resolutions": [10, 20]}}
    )
    _measure_display_object(
        cfg,
        GeoZarrAdapter(),
        str(target),
        "s3://bucket/store.zarr",
        str(tmp_path),
        "http://titiler.example",
    )
    assert len(captured_calls) == 1
    assert captured_calls[0]["extra_query"] == {"variable": DATA_VAR, "group": "0"}
    assert captured_calls[0]["group_query_key"] is None


def test_measure_display_object_geozarr_addresses_the_multiscales_owning_group(
    tmp_path, monkeypatch
):
    # #119: GeoZarrReader only resolves the pyramid from the requested zoom
    # for the *specific* group whose own zarr_conventions declare
    # `multiscales` -- any other group is read as-is, no zoom-awareness at
    # all. For a #114 cross-tier unified pyramid (S2: 10 m reflectance + 20
    # m masks), that group is always the store root, never a component's
    # own component_locator (its native *level*, e.g. "0"/"1") -- addressing
    # a component there would silently pin every request to its native
    # level regardless of the tile's actual zoom. A #102 single-resolution
    # bundle (S1: VV+VH) is unaffected -- there, component_locator ("grid0")
    # already IS the group that owns the doc.
    pytest.importorskip("rasterio")
    pytest.importorskip("rioxarray")
    import rasterio
    from rasterio.transform import from_origin

    import cng_benchmark.runner as _runner
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

    captured_calls = []

    def _fake_measure_display(endpoint, uri, tiles, **kwargs):
        captured_calls.append(kwargs)
        return []

    monkeypatch.setattr(_runner, "measure_display", _fake_measure_display)

    cfg = load_benchmark_config("configs/benchmarks/synthetic_geozarr_reader.yaml")
    cfg = cfg.model_copy(
        update={"params": {**cfg.params, "display_titiler_path": "geozarr"}}
    )

    # A #114 unified pyramid: b2 (10 m, native level "0") + clm (20 m,
    # native level "1") -- the shape #119 was filed against.
    _write_source(str(tmp_path / "b2.tif"), 1, 10)
    _write_source(str(tmp_path / "clm.tif"), 2, 20)
    sources = [
        SourceObject(name="b2", uri=str(tmp_path / "b2.tif")),
        SourceObject(name="clm", uri=str(tmp_path / "clm.tif")),
    ]
    unified_target = tmp_path / "unified.zarr"
    adapter = GeoZarrAdapter()
    adapter.convert_batch(
        sources,
        str(unified_target),
        {
            "chunk_shape": [32, 32],
            "shard_shape": [64, 64],
            "multiscale_levels": [20, 60],
        },
    )
    for name in ("b2", "clm"):
        captured_calls.clear()
        locator = adapter.component_locator(str(unified_target), name)
        _measure_display_object(
            cfg,
            adapter,
            str(unified_target),
            "s3://bucket/unified.zarr",
            str(tmp_path),
            "http://titiler.example",
            locator=locator,
            name=name,
        )
        # Every tile, whatever component or resolution, addresses the
        # store root -- not "locator" (a native level like "0" or "1").
        assert all(
            c["extra_query"] == {"variables": f"/:{name}"} for c in captured_calls
        ), captured_calls

    # A #102 single-resolution bundle: unaffected -- component_locator
    # ("grid0") already IS the group that owns the multiscales doc.
    _write_source(str(tmp_path / "vv.tif"), 1, 10)
    _write_source(str(tmp_path / "vh.tif"), 2, 10)
    bundle_sources = [
        SourceObject(name="vv", uri=str(tmp_path / "vv.tif")),
        SourceObject(name="vh", uri=str(tmp_path / "vh.tif")),
    ]
    bundle_target = tmp_path / "bundle.zarr"
    adapter.convert_batch(
        bundle_sources,
        str(bundle_target),
        {"chunk_shape": [32, 32], "shard_shape": [64, 64], "multiscale_levels": 2},
    )
    captured_calls.clear()
    locator = adapter.component_locator(str(bundle_target), "vv")
    _measure_display_object(
        cfg,
        adapter,
        str(bundle_target),
        "s3://bucket/bundle.zarr",
        str(tmp_path),
        "http://titiler.example",
        locator=locator,
        name="vv",
    )
    assert all(
        c["extra_query"] == {"variables": "/grid0:vv"} for c in captured_calls
    ), captured_calls


def test_tool_versions_records_titiler_healthz_failure(monkeypatch):
    # #119: a run's tool_versions must show *why* titiler_eopf (etc.) is
    # missing rather than silently omitting it -- indistinguishable from a
    # healthy endpoint that just reported nothing.
    import cng_benchmark.runner as _runner

    def _fake_fetch_titiler_versions(endpoint, **kwargs):
        return {"tiler_healthz_error": "TiTiler unreachable at ...: refused"}

    monkeypatch.setattr(_runner, "fetch_titiler_versions", _fake_fetch_titiler_versions)
    versions = _tool_versions("http://titiler.example")
    assert versions["cng_benchmark"] == __version__
    assert "tiler_healthz_error" in versions


def test_safe_read_metrics_returns_skipped_on_failure(monkeypatch):
    import cng_benchmark.runner as _runner

    def _raise(*a, **k):
        raise RuntimeError("timed out")

    monkeypatch.setattr(_runner, "_measure_object_read", _raise)
    result = _safe_read_metrics(adapter=None, object_uri="s3://b/k")
    assert len(result) == 1
    assert result[0].name == "read_skipped"
    assert "timed out" in result[0].detail["error"]


def test_safe_display_metrics_returns_skipped_on_failure(monkeypatch):
    import cng_benchmark.runner as _runner

    def _raise(*a, **k):
        raise RuntimeError("HTTP 504")

    monkeypatch.setattr(_runner, "_measure_display_object", _raise)
    metrics, artifacts = _safe_display_metrics(
        config=None,
        adapter=None,
        local_target="",
        object_uri="",
        artifact_dir="",
        titiler_endpoint=None,
    )
    assert len(metrics) == 1
    assert metrics[0].name == "display_skipped"
    assert "HTTP 504" in metrics[0].detail["error"]
    assert artifacts == []
