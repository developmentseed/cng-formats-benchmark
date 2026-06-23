"""Read metric — range-request-aware read latency and throughput.

Opens the produced object with rasterio and reads a grid of windows, timing each
read. When the object lives on S3 (``s3://`` mapped to GDAL ``/vsis3``), those
window reads become HTTP range requests against the store, so this measures the
realistic cloud-native access pattern — partial reads of an internally tiled
COG — rather than a bulk download. Requires the ``cog`` extra (rasterio).

Latency reflects the full range-request round-trip. Throughput is reported as
*decoded* bytes per second (``read_decoded_throughput``), not bytes over the
wire — a fair relative number across formats (all decode), explicitly named so
it is not mistaken for wire transfer. True wire bytes would need GDAL
``/vsis3`` transfer stats (a later refinement).
"""

from __future__ import annotations

import math
import time
from statistics import median

from cng_benchmark.models import MetricResult
from cng_benchmark.storage import to_gdal_path


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
    if windows < 1 or window_size < 1:
        raise ValueError("windows and window_size must be >= 1")
    rasterio, Window = _require_geo()
    path = to_gdal_path(uri)

    latencies: list[float] = []
    bytes_read = 0
    with rasterio.open(path) as src:
        win = min(window_size, src.width, src.height)
        for col, row in _grid_origins(src.width, src.height, win, windows):
            start = time.perf_counter()
            data = src.read(1, window=Window(col, row, win, win))
            latencies.append(time.perf_counter() - start)
            bytes_read += int(data.nbytes)

    return _read_metrics(latencies, bytes_read, win)


def _read_metrics(
    latencies: list[float], bytes_read: int, window_px: int
) -> list[MetricResult]:
    """Assemble the shared ``read_*`` metrics from per-window latencies.

    Shared by the rasterio (COG) and zarr-native (GeoZarr) collectors so both
    formats report the same names/units; throughput is *decoded* in-memory bytes
    per second (a fair relative cross-format number, not wire transfer).
    """
    total = sum(latencies)
    throughput = bytes_read / total if total > 0 else float("inf")
    return [
        MetricResult(name="read_window_count", value=len(latencies)),
        MetricResult(name="read_latency_mean", value=total / len(latencies), unit="s"),
        MetricResult(name="read_latency_p50", value=float(median(latencies)), unit="s"),
        MetricResult(
            name="read_decoded_throughput",
            value=throughput,
            unit="decoded-bytes/s",
            detail={"decoded_bytes": bytes_read, "window_px": window_px},
        ),
    ]


def measure_zarr_read(
    uri: str,
    *,
    role: str = "sink",
    windows: int = 8,
    window_size: int = 256,
) -> list[MetricResult]:
    """Read a grid of windows from the GeoZarr store at ``uri`` and return metrics.

    The zarr-native counterpart to :func:`measure_read`: GDAL's Zarr driver cannot
    read the ``sharding_indexed`` codec, so the finest array is opened with
    zarr-python over fsspec. Each window read pulls only the chunks it overlaps —
    HTTP range requests against the shard objects when ``uri`` is S3 — so this is
    the realistic partial-access pattern for a sharded cube. Emits the same
    ``read_*`` metrics as the COG path.
    """
    if windows < 1 or window_size < 1:
        raise ValueError("windows and window_size must be >= 1")
    arr = _open_zarr_array(uri, role)
    height, width = arr.shape[-2], arr.shape[-1]
    win = min(window_size, width, height)

    latencies: list[float] = []
    bytes_read = 0
    for col, row in _grid_origins(width, height, win, windows):
        start = time.perf_counter()
        data = arr[row : row + win, col : col + win]
        latencies.append(time.perf_counter() - start)
        bytes_read += int(data.nbytes)
    return _read_metrics(latencies, bytes_read, win)


def _open_zarr_array(uri: str, role: str):
    """Open the finest 2D array of a GeoZarr store (root array or multiscale 0)."""
    import zarr

    from cng_benchmark.formats.geozarr import DATA_VAR
    from cng_benchmark.storage import fsspec_storage_options, is_s3

    storage_options = fsspec_storage_options(role) if is_s3(uri) else None
    group = zarr.open_group(uri, mode="r", storage_options=storage_options)
    if DATA_VAR in group:
        return group[DATA_VAR]
    level_keys = sorted((k for k in group.group_keys()), key=lambda k: int(k))
    return group[level_keys[0]][DATA_VAR]


def _require_vector():
    try:
        import geopandas  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the vector read metric requires the 'geoparquet' extra; install with "
            "`uv sync --extra geoparquet` (or `pip install cng-benchmark[geoparquet]`)"
        ) from exc
    import geopandas

    return geopandas


def _bbox_grid(
    bounds: tuple[float, float, float, float], count: int
) -> list[tuple[float, float, float, float]]:
    """Tile ``bounds`` into up to ``count`` cell bboxes for partial spatial reads."""
    minx, miny, maxx, maxy = bounds
    per_side = max(1, int(math.ceil(math.sqrt(count))))
    dx = (maxx - minx) / per_side or 1.0
    dy = (maxy - miny) / per_side or 1.0
    cells = [
        (minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy)
        for j in range(per_side)
        for i in range(per_side)
    ]
    return cells[:count]


def measure_vector_read(
    uri: str,
    *,
    role: str = "sink",
    queries: int = 8,
) -> list[MetricResult]:
    """Run a grid of bbox spatial queries against the GeoParquet at ``uri``.

    The vector counterpart to :func:`measure_read`: each query passes a bbox
    predicate to ``geopandas.read_parquet``, which pushes it down to the row
    groups whose covering bbox overlaps — only those row groups are fetched (HTTP
    range requests against the file when ``uri`` is S3), so this measures the
    realistic partial-access pattern for a GeoParquet, not a full table scan. The
    file's total extent (read once, untimed) seeds the query grid. Emits the same
    ``read_latency_*`` / ``read_decoded_throughput`` family as the raster path,
    counting returned features rather than pixels.
    """
    if queries < 1:
        raise ValueError("queries must be >= 1")
    gpd = _require_vector()
    storage_options = _vector_storage_options(uri, role)

    bounds = _vector_total_bounds(gpd, uri, storage_options)

    latencies: list[float] = []
    features = 0
    decoded_bytes = 0
    for bbox in _bbox_grid(bounds, queries):
        start = time.perf_counter()
        sub = gpd.read_parquet(uri, bbox=bbox, storage_options=storage_options)
        latencies.append(time.perf_counter() - start)
        features += len(sub)
        decoded_bytes += int(sub.memory_usage(deep=True).sum())

    return _vector_read_metrics(latencies, decoded_bytes, features)


def _vector_storage_options(uri: str, role: str) -> dict | None:
    """fsspec options for an S3 GeoParquet, or ``None`` for a local file."""
    from cng_benchmark.storage import fsspec_storage_options, is_s3

    return fsspec_storage_options(role) if is_s3(uri) else None


def _vector_total_bounds(
    gpd, uri: str, storage_options: dict | None
) -> tuple[float, float, float, float]:
    """Return the GeoParquet's total ``(minx, miny, maxx, maxy)`` extent.

    Reads the geometry once (untimed setup) to derive the query grid; the timed
    reads are the per-bbox partial accesses that follow.
    """
    geom = gpd.read_parquet(uri, storage_options=storage_options)
    minx, miny, maxx, maxy = (float(v) for v in geom.total_bounds)
    return (minx, miny, maxx, maxy)


def _vector_read_metrics(
    latencies: list[float], decoded_bytes: int, features: int
) -> list[MetricResult]:
    """Assemble the ``read_*`` metrics from per-query latencies (vector path).

    Mirrors :func:`_read_metrics` so the vector and raster arms report the same
    latency/throughput names; throughput is *decoded* in-memory bytes per second
    and the partial-access unit is a bbox query, not a raster window.
    """
    total = sum(latencies)
    throughput = decoded_bytes / total if total > 0 else float("inf")
    return [
        MetricResult(name="read_query_count", value=len(latencies)),
        MetricResult(name="read_latency_mean", value=total / len(latencies), unit="s"),
        MetricResult(name="read_latency_p50", value=float(median(latencies)), unit="s"),
        MetricResult(
            name="read_decoded_throughput",
            value=throughput,
            unit="decoded-bytes/s",
            detail={"decoded_bytes": decoded_bytes, "features": features},
        ),
    ]
