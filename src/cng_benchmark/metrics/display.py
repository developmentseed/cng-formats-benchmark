"""Display metric — chunk-aware tile latency against an in-stack TiTiler service.

Measures how quickly a tile server can render map tiles from the produced object
— the "can you actually look at it on a map" question — and, crucially, *how that
cost scales with chunk-crossing*. Rendering a 256-px web tile costs roughly as
many internal block (chunk) reads as the tile footprint straddles, so we time a
set of tiles deliberately chosen to touch 1, 2, 4 and 9+ blocks (selection lives
in :mod:`cng_benchmark.metrics.display_tiles`, which needs the geo stack).

This collector itself stays import-light: it only talks to a running TiTiler
instance (a deployment service dependency, not a Python dep of the harness) over
HTTP using the standard library. TiTiler reads the object via GDAL from the
configured store. The geo-dependent tile *selection* is done upstream and handed
in as :class:`TileSpec` values, so this module imports no rasterio/morecantile.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from statistics import median
from typing import NamedTuple
from urllib.parse import quote

from cng_benchmark.models import MetricResult


class TileSpec(NamedTuple):
    """A tile chosen to exercise a particular chunk-crossing scenario.

    ``label`` names the scenario (e.g. ``"1chunk"``); ``z/x/y`` is the
    WebMercator tile; ``chunks`` is the number of internal blocks the tile is
    estimated to touch; ``approx`` flags a substitute when the exact bucket was
    unreachable. Defined here (no geo deps) so :func:`measure_display` stays
    import-light; built by :mod:`cng_benchmark.metrics.display_tiles`.
    """

    label: str
    z: int
    x: int
    y: int
    chunks: int
    approx: bool = False


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
    tiles: list[TileSpec],
    *,
    samples: int = 8,
    tile_matrix_set: str = "WebMercatorQuad",
    fmt: str = "png",
    timeout: float = 30.0,
) -> list[MetricResult]:
    """Time TiTiler tile fetches per chunk-crossing scenario.

    ``endpoint`` is the TiTiler base URL; ``cog_uri`` is the GDAL-readable URL
    TiTiler serves from (e.g. ``s3://…``). ``tiles`` are the scenarios to time,
    one per chunk bucket (see :func:`display_tiles.select_chunk_tiles`); each is
    fetched ``samples`` times and reported as its own flat ``display_{label}_*``
    metrics. Returns an empty-scenario summary if ``tiles`` is empty (e.g. no
    bucket was reachable for this object).
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    base = endpoint.rstrip("/")
    encoded = quote(cog_uri, safe="")

    # Validate the object is servable before timing tiles (clearer failures).
    _fetch(f"{base}/cog/info?url={encoded}", timeout)

    metrics: list[MetricResult] = []
    for spec in tiles:
        tile_url = (
            f"{base}/cog/tiles/{tile_matrix_set}/"
            f"{spec.z}/{spec.x}/{spec.y}.{fmt}?url={encoded}"
        )
        latencies: list[float] = []
        bytes_total = 0
        for _ in range(samples):
            start = time.perf_counter()
            body = _fetch(tile_url, timeout)
            latencies.append(time.perf_counter() - start)
            bytes_total += len(body)

        total = sum(latencies)
        # First sample is the cold read; TiTiler warms its cache afterwards.
        metrics += [
            MetricResult(
                name=f"display_{spec.label}_latency_mean",
                value=total / len(latencies),
                unit="s",
                detail={
                    "tile": f"{spec.z}/{spec.x}/{spec.y}",
                    "chunks": spec.chunks,
                    "approx": spec.approx,
                    "bytes_total": bytes_total,
                    "cold_s": latencies[0],
                },
            ),
            MetricResult(
                name=f"display_{spec.label}_latency_p50",
                value=float(median(latencies)),
                unit="s",
            ),
        ]

    metrics.append(
        MetricResult(
            name="display_scenarios",
            value=len(tiles),
            detail={
                "tile_matrix_set": tile_matrix_set,
                "samples": samples,
                "scenarios": [
                    {"label": s.label, "tile": f"{s.z}/{s.x}/{s.y}", "chunks": s.chunks}
                    for s in tiles
                ],
            },
        )
    )
    return metrics
