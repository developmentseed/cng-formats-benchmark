"""Display metric — tile latency against an in-stack TiTiler service.

Measures how quickly a tile server can render map tiles from the produced object
— the "can you actually look at it on a map" question. It calls a running
TiTiler instance (a deployment service dependency, not a Python dep of the
harness) over HTTP using only the standard library, so the collector stays
import-light; TiTiler reads the object itself via GDAL from the configured store.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from statistics import median
from urllib.parse import quote

from cng_benchmark.models import MetricResult


def _fetch(url: str, timeout: float) -> bytes:
    """GET ``url`` and return the body, raising a clear error on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"TiTiler returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TiTiler unreachable at {url}: {exc.reason}") from exc


def measure_display(
    endpoint: str,
    cog_uri: str,
    *,
    samples: int = 8,
    tile_matrix_set: str = "WebMercatorQuad",
    tile: tuple[int, int, int] = (0, 0, 0),
    fmt: str = "png",
    timeout: float = 30.0,
) -> list[MetricResult]:
    """Time repeated tile fetches from TiTiler and return display metrics.

    ``endpoint`` is the TiTiler base URL; ``cog_uri`` is the GDAL-readable URL
    TiTiler serves from (e.g. ``s3://…``). The default tile ``z/x/y = 0/0/0``
    covers the whole world, so it renders for any global raster.
    """
    base = endpoint.rstrip("/")
    encoded = quote(cog_uri, safe="")

    # Validate the object is servable before timing tiles (clearer failures).
    _fetch(f"{base}/cog/info?url={encoded}", timeout)

    z, x, y = tile
    tile_url = f"{base}/cog/tiles/{tile_matrix_set}/{z}/{x}/{y}.{fmt}?url={encoded}"

    latencies: list[float] = []
    bytes_total = 0
    for _ in range(samples):
        start = time.perf_counter()
        body = _fetch(tile_url, timeout)
        latencies.append(time.perf_counter() - start)
        bytes_total += len(body)

    total = sum(latencies)
    return [
        MetricResult(name="display_tile_count", value=len(latencies)),
        MetricResult(
            name="display_latency_mean", value=total / len(latencies), unit="s"
        ),
        MetricResult(
            name="display_latency_p50", value=float(median(latencies)), unit="s"
        ),
        MetricResult(
            name="display_latency_max",
            value=max(latencies),
            unit="s",
            detail={"bytes_total": bytes_total, "tile": f"{z}/{x}/{y}"},
        ),
    ]
