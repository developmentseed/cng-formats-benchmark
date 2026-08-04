"""Read metric — range-request-aware subsetting-read latency and throughput.

Opens the produced object and reads a handful of random windows — a bbox /
sub-zone query, not a full/sequential scan — timing each. This is the CNES
subsetting-read pattern (issue #74): windows are seeded-random rather than
laid out on a fixed grid, so repeated runs sample different parts of the
object while staying reproducible. When the object lives on S3 (``s3://``
mapped to GDAL ``/vsis3``), those window reads become HTTP range requests
against the store, so this measures the realistic cloud-native access
pattern — partial reads — rather than a bulk download. Requires the ``cog``
extra (rasterio).

Latency reflects the full range-request round-trip. Throughput is reported as
*decoded* bytes per second (``read_decoded_throughput``), not bytes over the
wire — a fair relative number across formats (all decode), explicitly named so
it is not mistaken for wire transfer. True wire bytes would need GDAL
``/vsis3`` transfer stats (a later refinement).
"""

from __future__ import annotations

import math
import random
import time
from statistics import median, pstdev

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


def _random_origins(
    width: int,
    height: int,
    win: int,
    count: int,
    rng: random.Random,
    block: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Return ``count`` random ``(col, row)`` window origins within bounds.

    Positions are drawn uniformly at random (seeded via ``rng``) rather than
    laid out on a fixed grid, so the sample lands on different parts of the
    raster from run to run while staying reproducible for a given seed. When
    ``block`` — a COG's internal tile shape as ``(block_h, block_w)`` — is
    given, alternates between tile-aligned origins (snapped to the block grid)
    and tile-unaligned ones, since a real bbox query stresses both depending on
    where it falls relative to the tiling.
    """
    max_x = max(0, width - win)
    max_y = max(0, height - win)
    origins = []
    for i in range(count):
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)
        if block is not None:
            block_h, block_w = block
            if i % 2 == 0:
                if block_w > 0:
                    x = min((x // block_w) * block_w, max_x)
                if block_h > 0:
                    y = min((y // block_h) * block_h, max_y)
            else:
                if block_w > 1 and x % block_w == 0:
                    x = min(x + block_w // 2, max_x)
                if block_h > 1 and y % block_h == 0:
                    y = min(y + block_h // 2, max_y)
        origins.append((x, y))
    return origins


def measure_read(
    uri: str,
    *,
    windows: int = 8,
    window_size: int = 256,
    seed: int = 0,
) -> list[MetricResult]:
    """Read ``windows`` random sub-zones from the object at ``uri``.

    Alternates tile-aligned and tile-unaligned origins (the COG's internal
    block grid) so both access patterns are exercised. ``seed`` makes the
    sample reproducible across runs.
    """
    if windows < 1 or window_size < 1:
        raise ValueError("windows and window_size must be >= 1")
    rasterio, Window = _require_geo()
    path = to_gdal_path(uri)
    rng = random.Random(seed)

    latencies: list[float] = []
    bytes_read = 0
    with rasterio.open(path) as src:
        win = min(window_size, src.width, src.height)
        block = src.block_shapes[0] if src.block_shapes else None
        origins = _random_origins(src.width, src.height, win, windows, rng, block)
        for col, row in origins:
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
            name="read_latency_spread",
            value=float(pstdev(latencies)),
            unit="s",
            detail={"latencies": latencies},
        ),
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
    seed: int = 0,
) -> list[MetricResult]:
    """Read ``windows`` random chunk-range slices from the GeoZarr store at ``uri``.

    The zarr-native counterpart to :func:`measure_read`: GDAL's Zarr driver cannot
    read the ``sharding_indexed`` codec, so the finest array is opened with
    zarr-python over fsspec. Each random slice pulls only the chunks it overlaps —
    HTTP range requests against the shard objects when ``uri`` is S3 — so this is
    the realistic partial-access pattern for a sharded cube. ``seed`` makes the
    sample reproducible across runs. Emits the same ``read_*`` metrics as the COG
    path.
    """
    if windows < 1 or window_size < 1:
        raise ValueError("windows and window_size must be >= 1")
    rng = random.Random(seed)
    arr = _open_zarr_array(uri, role)
    height, width = arr.shape[-2], arr.shape[-1]
    win = min(window_size, width, height)

    latencies: list[float] = []
    bytes_read = 0
    for col, row in _random_origins(width, height, win, windows, rng):
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


def _random_bboxes(
    bounds: tuple[float, float, float, float], count: int, rng: random.Random
) -> list[tuple[float, float, float, float]]:
    """Return ``count`` random bbox windows within ``bounds``.

    Each window keeps the footprint a grid cell would (``extent / sqrt(count)``
    per side, so query selectivity stays comparable across the sample) but is
    placed at a random position (seeded via ``rng``) rather than tiled, so
    repeated queries land on different, potentially row-group-crossing, parts of
    the extent.
    """
    minx, miny, maxx, maxy = bounds
    per_side = max(1, int(math.ceil(math.sqrt(count))))
    dx = (maxx - minx) / per_side or 1.0
    dy = (maxy - miny) / per_side or 1.0
    span_x = max(0.0, (maxx - minx) - dx)
    span_y = max(0.0, (maxy - miny) - dy)
    boxes = []
    for _ in range(count):
        ox = minx + rng.uniform(0, span_x)
        oy = miny + rng.uniform(0, span_y)
        boxes.append((ox, oy, ox + dx, oy + dy))
    return boxes


def measure_vector_read(
    uri: str,
    *,
    role: str = "sink",
    queries: int = 8,
    seed: int = 0,
) -> list[MetricResult]:
    """Run ``queries`` random bbox spatial queries against the GeoParquet at ``uri``.

    The vector counterpart to :func:`measure_read`: each query passes a bbox
    predicate to ``geopandas.read_parquet``, which pushes it down to the row
    groups whose covering bbox overlaps — only those row groups are fetched (HTTP
    range requests against the file when ``uri`` is S3), so this measures the
    realistic partial-access pattern for a GeoParquet, not a full table scan. The
    file's total extent (read once, untimed) seeds the random query positions;
    ``seed`` makes the sample reproducible across runs. Emits the same
    ``read_latency_*`` / ``read_decoded_throughput`` family as the raster path,
    counting returned features rather than pixels.
    """
    if queries < 1:
        raise ValueError("queries must be >= 1")
    gpd = _require_vector()
    storage_options = _vector_storage_options(uri, role)
    rng = random.Random(seed)

    bounds = _vector_total_bounds(gpd, uri, storage_options)

    latencies: list[float] = []
    features = 0
    decoded_bytes = 0
    for bbox in _random_bboxes(bounds, queries, rng):
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


def _require_pointcloud():
    try:
        import laspy  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the COPC read metric requires the 'copc' extra; install with "
            "`uv sync --extra copc` (or `pip install cng-benchmark[copc]`)"
        ) from exc
    import laspy

    return laspy


def measure_copc_read(
    uri: str,
    *,
    role: str = "sink",
    queries: int = 8,
    seed: int = 0,
) -> list[MetricResult]:
    """Run ``queries`` random octree-node spatial queries against the COPC at ``uri``.

    The point-cloud counterpart to :func:`measure_read`: each query passes a 3D
    bbox to laspy's ``CopcReader.spatial_query``, which fetches only the octree
    nodes that overlap (HTTP range requests against the file when ``uri`` is S3, via
    an fsspec stream), so this measures the realistic partial-access pattern for a
    COPC — an octree-node fetch — not a full-cloud scan. The file's extent (read
    from the header, untimed) seeds the random query positions; ``seed`` makes the
    sample reproducible across runs. Emits the same ``read_latency_*`` /
    ``read_decoded_throughput`` family as the other arms, counting returned points
    rather than pixels.
    """
    if queries < 1:
        raise ValueError("queries must be >= 1")
    laspy = _require_pointcloud()
    rng = random.Random(seed)
    handle = _copc_handle(uri, role)
    try:
        reader = laspy.CopcReader.open(handle)
        header = reader.header
        mins = [float(v) for v in header.mins]
        maxs = [float(v) for v in header.maxs]

        latencies: list[float] = []
        points = 0
        decoded_bytes = 0
        for box in _random_boxes(mins, maxs, queries, rng):
            start = time.perf_counter()
            sub = reader.spatial_query(box)
            latencies.append(time.perf_counter() - start)
            n = len(sub.x)
            points += n
            decoded_bytes += n * 3 * 8  # x, y, z as float64
    finally:
        close = getattr(handle, "close", None)
        if callable(close):
            close()
    return _pointcloud_read_metrics(latencies, decoded_bytes, points)


def _copc_handle(uri: str, role: str):
    """Open ``uri`` as a laspy COPC source: a local path or an fsspec S3 stream."""
    from cng_benchmark.storage import fsspec_storage_options, is_s3

    if is_s3(uri):
        import fsspec

        return fsspec.open(uri, mode="rb", **fsspec_storage_options(role)).open()
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return uri


def _random_boxes(
    mins: list[float], maxs: list[float], count: int, rng: random.Random
):
    """Return ``count`` random 3D ``laspy.copc.Bounds`` (full Z, random X–Y).

    Each box keeps the footprint a grid cell would (``extent / sqrt(count)`` per
    side) but is placed at a random X–Y position (seeded via ``rng``) rather than
    tiled, so repeated queries land on different, potentially octree-node-crossing,
    parts of the extent.
    """
    from laspy.copc import Bounds

    minx, miny, minz = mins
    maxx, maxy, maxz = maxs
    per_side = max(1, int(math.ceil(math.sqrt(count))))
    dx = (maxx - minx) / per_side or 1.0
    dy = (maxy - miny) / per_side or 1.0
    span_x = max(0.0, (maxx - minx) - dx)
    span_y = max(0.0, (maxy - miny) - dy)
    boxes = []
    for _ in range(count):
        ox = minx + rng.uniform(0, span_x)
        oy = miny + rng.uniform(0, span_y)
        boxes.append(Bounds([ox, oy, minz], [ox + dx, oy + dy, maxz]))
    return boxes


def _pointcloud_read_metrics(
    latencies: list[float], decoded_bytes: int, points: int
) -> list[MetricResult]:
    """Assemble the ``read_*`` metrics from per-query latencies (point-cloud path).

    Mirrors :func:`_read_metrics` / :func:`_vector_read_metrics` so every arm
    reports the same latency/throughput names; the partial-access unit is an
    octree-node spatial query and throughput counts decoded point bytes.
    """
    total = sum(latencies)
    throughput = decoded_bytes / total if total > 0 else float("inf")
    return [
        MetricResult(name="read_query_count", value=len(latencies)),
        MetricResult(name="read_latency_mean", value=total / len(latencies), unit="s"),
        MetricResult(name="read_latency_p50", value=float(median(latencies)), unit="s"),
        MetricResult(
            name="read_latency_spread",
            value=float(pstdev(latencies)),
            unit="s",
            detail={"latencies": latencies},
        ),
        MetricResult(
            name="read_decoded_throughput",
            value=throughput,
            unit="decoded-bytes/s",
            detail={"decoded_bytes": decoded_bytes, "points": points},
        ),
    ]


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
            name="read_latency_spread",
            value=float(pstdev(latencies)),
            unit="s",
            detail={"latencies": latencies},
        ),
        MetricResult(
            name="read_decoded_throughput",
            value=throughput,
            unit="decoded-bytes/s",
            detail={"decoded_bytes": decoded_bytes, "features": features},
        ),
    ]
