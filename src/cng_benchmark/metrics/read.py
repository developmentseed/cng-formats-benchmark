"""Read metric — range-request-aware read latency and throughput.

Opens the produced object with rasterio and reads a grid of windows, timing each
read. When the object lives on S3 (``s3://`` mapped to GDAL ``/vsis3``), those
window reads become HTTP range requests against the store, so this measures the
realistic cloud-native access pattern — partial reads of an internally tiled
COG — rather than a bulk download. Requires the ``cog`` extra (rasterio).
"""

from __future__ import annotations

import math
import time
from statistics import median

from cng_benchmark.models import MetricResult


def _require_geo():
    try:
        import rasterio
        from rasterio.windows import Window
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the read metric requires the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return rasterio, Window


def _vsi_path(uri: str) -> str:
    """Map a storage URI to a GDAL-openable path (``s3://`` → ``/vsis3/``)."""
    if uri.startswith("s3://"):
        return "/vsis3/" + uri[len("s3://") :]
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def _grid_origins(
    width: int, height: int, win: int, count: int
) -> list[tuple[int, int]]:
    """Return up to ``count`` distinct ``(col, row)`` window origins on a grid."""
    per_side = max(1, int(math.ceil(math.sqrt(count))))
    xs = sorted({min(i * win, max(0, width - win)) for i in range(per_side)})
    ys = sorted({min(j * win, max(0, height - win)) for j in range(per_side)})
    origins = [(x, y) for y in ys for x in xs]
    return origins[:count]


def measure_read(
    uri: str,
    *,
    windows: int = 8,
    window_size: int = 256,
) -> list[MetricResult]:
    """Read a grid of windows from the object at ``uri`` and return read metrics."""
    rasterio, Window = _require_geo()
    path = _vsi_path(uri)

    latencies: list[float] = []
    bytes_read = 0
    with rasterio.open(path) as src:
        win = min(window_size, src.width, src.height)
        for col, row in _grid_origins(src.width, src.height, win, windows):
            start = time.perf_counter()
            data = src.read(1, window=Window(col, row, win, win))
            latencies.append(time.perf_counter() - start)
            bytes_read += int(data.nbytes)

    total = sum(latencies)
    throughput = bytes_read / total if total > 0 else float("inf")
    return [
        MetricResult(name="read_window_count", value=len(latencies)),
        MetricResult(name="read_latency_mean", value=total / len(latencies), unit="s"),
        MetricResult(name="read_latency_p50", value=float(median(latencies)), unit="s"),
        MetricResult(
            name="read_throughput",
            value=throughput,
            unit="bytes/s",
            detail={"bytes_read": bytes_read, "window_px": win},
        ),
    ]
