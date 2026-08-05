"""Tests for the write/read/display metric collectors.

The write and read collectors need rasterio (the `cog` extra) and run against a
local file. The display collector talks HTTP, so it is tested with a fake
``urlopen`` — no TiTiler required.
"""

import random

import pytest

from cng_benchmark.metrics import display
from cng_benchmark.metrics.display import TileSpec

_TILES = [
    TileSpec("1chunk", 0, 0, 0, 1),
    TileSpec("2chunk", 1, 0, 0, 2),
]


def test_display_measures_tiles_with_fake_titiler(monkeypatch):
    calls = []

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"\x89PNG\r\n"  # plausible tile body

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return _FakeResp()

    monkeypatch.setattr(display.urllib.request, "urlopen", fake_urlopen)

    metrics = display.measure_display(
        "http://titiler:8000/", "s3://bench/results/cog/cog.tif", _TILES, samples=3
    )
    names = {m.name for m in metrics}
    # Flat per-scenario metrics, one mean+p50 per chunk bucket, plus a summary.
    assert {
        "display_1chunk_latency_mean",
        "display_1chunk_latency_p50",
        "display_2chunk_latency_mean",
        "display_2chunk_latency_p50",
        "display_scenarios",
    } <= names
    assert next(m.value for m in metrics if m.name == "display_scenarios") == 2
    detail = next(m.detail for m in metrics if m.name == "display_1chunk_latency_mean")
    assert detail["chunks"] == 1
    assert "cold_s" in detail
    # Every metric records which router served it (#88) — no figure ambiguous
    # about which reader produced it.
    assert detail["path_prefix"] == "cog"
    p50_detail = next(
        m.detail for m in metrics if m.name == "display_1chunk_latency_p50"
    )
    assert p50_detail["path_prefix"] == "cog"
    scenarios_detail = next(m.detail for m in metrics if m.name == "display_scenarios")
    assert scenarios_detail["path_prefix"] == "cog"
    # /cog/info once, then samples fetches per tile (1 + 2*3 = 7 calls).
    assert calls[0].startswith("http://titiler:8000/cog/info?url=")
    assert len(calls) == 1 + len(_TILES) * 3
    assert "/cog/tiles/WebMercatorQuad/0/0/0.png?url=" in calls[1]
    assert "s3%3A%2F%2Fbench" in calls[1]


def test_display_handles_no_reachable_scenarios(monkeypatch):
    monkeypatch.setattr(
        display.urllib.request, "urlopen", lambda url, timeout=None: _Resp()
    )
    metrics = display.measure_display("http://titiler:8000", "s3://b/k.tif", [])
    names = {m.name for m in metrics}
    assert names == {"display_scenarios"}
    assert next(m.value for m in metrics if m.name == "display_scenarios") == 0


class _Resp:
    def __init__(self, body: bytes = b""):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_display_raises_clear_error_on_http_failure(monkeypatch):
    import urllib.error

    def boom(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(display.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        display.measure_display("http://titiler:8000", "s3://b/k.tif", _TILES)


def test_display_rejects_zero_samples():
    with pytest.raises(ValueError, match="samples"):
        display.measure_display(
            "http://titiler:8000", "s3://b/k.tif", _TILES, samples=0
        )


def test_display_records_a_non_default_path_prefix(monkeypatch):
    # The geozarr arm (#88): a GeoZarrReader-backed router, addressed with
    # `variables` (not `variable`) per its own query contract.
    monkeypatch.setattr(
        display.urllib.request, "urlopen", lambda url, timeout=None: _Resp(b"tile")
    )
    metrics = display.measure_display(
        "http://titiler:8000",
        "s3://b/k.zarr",
        _TILES[:1],
        path_prefix="geozarr",
        extra_query={"variables": "/:data"},
    )
    detail = next(m.detail for m in metrics if m.name == "display_1chunk_latency_mean")
    assert detail["path_prefix"] == "geozarr"


def test_fetch_titiler_versions_parses_healthz(monkeypatch):
    body = (
        b'{"versions": {"titiler_core": "2.2.1", "titiler_eopf": "0.10.0", '
        b'"rasterio": "1.5.0"}}'
    )
    monkeypatch.setattr(
        display.urllib.request, "urlopen", lambda url, timeout=None: _Resp(body)
    )
    versions = display.fetch_titiler_versions("http://titiler:8000")
    # titiler_* keys pass through as-is; the generic GDAL-stack ones get a
    # `tiler_` prefix so they don't collide with the harness's own versions.
    assert versions == {
        "titiler_core": "2.2.1",
        "titiler_eopf": "0.10.0",
        "tiler_rasterio": "1.5.0",
    }


def test_fetch_titiler_versions_is_best_effort_on_failure(monkeypatch):
    import urllib.error

    def boom(url, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(display.urllib.request, "urlopen", boom)
    assert display.fetch_titiler_versions("http://titiler:8000") == {}


# --- write + read need rasterio -------------------------------------------------

pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.formats.cog import CogAdapter  # noqa: E402
from cng_benchmark.metrics.read import measure_read  # noqa: E402
from cng_benchmark.metrics.write import measure_write  # noqa: E402


def test_write_metric_converts_and_times(tmp_path):
    source = tmp_path / "source.tif"
    source.write_bytes(generate_cog_bytes(size=256, blocksize=256))
    target = tmp_path / "out.tif"

    metrics = measure_write(CogAdapter(), str(source), str(target), {"block_size": 128})

    assert target.exists()
    by_name = {m.name: m for m in metrics}
    assert by_name["write_elapsed"].value >= 0
    assert by_name["write_throughput"].value > 0
    assert by_name["write_throughput"].detail["bytes_out"] == target.stat().st_size


def test_read_metric_reads_windows_locally(tmp_path):
    cog = tmp_path / "cog.tif"
    cog.write_bytes(generate_cog_bytes(size=512, blocksize=256))

    metrics = measure_read(str(cog), windows=4, window_size=256)

    by_name = {m.name: m for m in metrics}
    assert by_name["read_window_count"].value >= 1
    assert by_name["read_latency_mean"].value >= 0
    assert by_name["read_latency_spread"].value >= 0
    assert len(by_name["read_latency_spread"].detail["latencies"]) == 4
    assert by_name["read_decoded_throughput"].value > 0
    assert by_name["read_decoded_throughput"].unit == "decoded-bytes/s"
    assert by_name["read_decoded_throughput"].detail["decoded_bytes"] > 0


def test_read_metric_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="window"):
        measure_read(str(tmp_path / "whatever.tif"), windows=0)


def test_random_origins_are_seeded_reproducible_and_bounded():
    from cng_benchmark.metrics.read import _random_origins

    a = _random_origins(1000, 1000, 100, 6, random.Random(0))
    b = _random_origins(1000, 1000, 100, 6, random.Random(0))
    assert a == b  # same seed -> same sample
    c = _random_origins(1000, 1000, 100, 6, random.Random(1))
    assert a != c  # different seed -> different sample
    assert all(0 <= x <= 900 and 0 <= y <= 900 for x, y in a)


def test_random_origins_alternate_tile_aligned_and_unaligned():
    from cng_benchmark.metrics.read import _random_origins

    block = (256, 256)  # (block_h, block_w)
    origins = _random_origins(2048, 2048, 300, 8, random.Random(0), block)
    aligned = origins[0::2]
    unaligned = origins[1::2]
    assert all(x % 256 == 0 and y % 256 == 0 for x, y in aligned)
    assert any(x % 256 != 0 or y % 256 != 0 for x, y in unaligned)
