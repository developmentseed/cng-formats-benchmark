"""Tests for the write/read/display metric collectors.

The write and read collectors need rasterio (the `cog` extra) and run against a
local file. The display collector talks HTTP, so it is tested with a fake
``urlopen`` — no TiTiler required.
"""

import pytest

from cng_benchmark.metrics import display


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
        "http://titiler:8000/", "s3://bench/results/cog/cog.tif", samples=3
    )
    names = {m.name for m in metrics}
    assert {"display_tile_count", "display_latency_mean"} <= names
    assert "display_latency_p50" in names
    assert next(m.value for m in metrics if m.name == "display_tile_count") == 3
    # First call validates via /cog/info, the rest fetch tiles; url is encoded.
    assert calls[0].startswith("http://titiler:8000/cog/info?url=")
    assert "/cog/tiles/WebMercatorQuad/0/0/0.png?url=" in calls[1]
    assert "s3%3A%2F%2Fbench" in calls[1]


def test_display_raises_clear_error_on_http_failure(monkeypatch):
    import urllib.error

    def boom(url, timeout=None):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(display.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        display.measure_display("http://titiler:8000", "s3://b/k.tif")


def test_display_rejects_zero_samples():
    with pytest.raises(ValueError, match="samples"):
        display.measure_display("http://titiler:8000", "s3://b/k.tif", samples=0)


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
    assert by_name["read_decoded_throughput"].value > 0
    assert by_name["read_decoded_throughput"].unit == "decoded-bytes/s"
    assert by_name["read_decoded_throughput"].detail["decoded_bytes"] > 0


def test_read_metric_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="window"):
        measure_read(str(tmp_path / "whatever.tif"), windows=0)
