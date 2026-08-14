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

import json
import logging
import time
import urllib.error
import urllib.request
from typing import NamedTuple
from urllib.parse import quote

from cng_benchmark.models import MetricResult

logger = logging.getLogger(__name__)


class TileSpec(NamedTuple):
    """A tile chosen to exercise a particular chunk-crossing or resolution
    scenario.

    ``label`` names the scenario (e.g. ``"1chunk"``, ``"res_60m"``); ``z/x/y``
    is the WebMercator tile; ``chunks`` is the number of internal blocks the
    tile is estimated to touch; ``approx`` flags a substitute when the exact
    bucket was unreachable. ``group`` is the GeoZarr group that actually
    serves this tile's resolution (``None`` for a chunk-crossing-bucket tile,
    which doesn't target a specific level, and always for COG, which has no
    server-side group concept — rio-tiler resolves its own overview by zoom).
    Informational only (``MetricResult.detail["group"]``) — no router's query
    is built from a per-tile ``.group`` anymore (#121: doing so for the stock
    xarray router credited it with multiscale awareness it doesn't have; a
    real client's query is fixed, the same for every tile regardless of the
    resolution being timed). Defined here (no geo deps) so
    :func:`measure_display` stays import-light; built by
    :mod:`cng_benchmark.metrics.display_tiles`.
    """

    label: str
    z: int
    x: int
    y: int
    chunks: int
    approx: bool = False
    group: str | None = None


def _fetch(url: str, timeout: float) -> bytes:
    """GET ``url`` and return the body, raising a clear error on failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"TiTiler returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"TiTiler unreachable at {url}: {exc.reason}") from exc


def fetch_titiler_versions(endpoint: str, *, timeout: float = 5.0) -> dict[str, str]:
    """Best-effort ``/healthz`` versions, so a run records which reader build
    produced its display numbers (``titiler_core``/``titiler_xarray``/
    ``titiler_eopf`` as-is; the generic GDAL-stack keys prefixed ``tiler_`` so
    they don't collide with the harness's own tool versions). On any failure
    (unreachable endpoint, unexpected payload) this must never fail a run —
    but a silent ``{}`` left no trace of *why* a run's ``tool_versions`` was
    missing every ``titiler_*``/``tiler_*`` key, indistinguishable from a
    healthy endpoint that just reported nothing (#119: this gap meant a
    crashing titiler left no record that ``/healthz`` itself never
    succeeded). Returns ``{"tiler_healthz_error": ...}`` instead, and logs
    the failure.
    """
    try:
        body = _fetch(f"{endpoint.rstrip('/')}/healthz", timeout)
        versions = json.loads(body)["versions"]
    except Exception as exc:  # noqa: BLE001 - best-effort, never fails a run
        logger.warning("titiler /healthz unreachable at %s: %s", endpoint, exc)
        return {"tiler_healthz_error": str(exc)}
    return {
        (k if k.startswith("titiler_") else f"tiler_{k}"): str(v)
        for k, v in versions.items()
    }


def measure_display(
    endpoint: str,
    cog_uri: str,
    tiles: list[TileSpec],
    *,
    tile_matrix_set: str = "WebMercatorQuad",
    tilesize: int = 256,
    fmt: str = "png",
    timeout: float = 30.0,
    path_prefix: str = "cog",
    extra_query: dict[str, str] | None = None,
    group_query_key: str | None = None,
) -> list[MetricResult]:
    """Time one TiTiler tile fetch per chunk-crossing or resolution scenario.

    ``endpoint`` is the TiTiler base URL; ``cog_uri`` is the URL TiTiler serves
    from (e.g. ``s3://…``). ``tiles`` are the scenarios to time — chunk-bucket
    tiles (see :func:`display_tiles.select_chunk_tiles`) and/or one-per-target-
    resolution tiles (:func:`display_tiles.select_zarr_resolution_tiles`) —
    each fetched *once* and reported as its own flat ``display_{label}_latency``
    metric (#122: repeating the same fetch to average out noise let a warm
    cache build up on one URL, which a real pan/zoom session never does — one
    deterministic fetch per tile is the honest number, not an average).
    Returns an empty-scenario summary if ``tiles`` is empty (e.g. no bucket
    was reachable for this object).

    ``tilesize`` (256, ``WebMercatorQuad``'s own tile size) is sent
    explicitly on every tile fetch rather than left for the router to
    default — a router that only coalesces a missing ``tilesize`` for *some*
    readers, not others, silently reads full native resolution at every zoom
    for the readers it doesn't coalesce for, never touching the multiscale
    pyramid the scenario is supposed to be measuring (#130: found via
    ``GeoZarrReader.tile()``, upstream-fixed in
    ``EOPF-Explorer/titiler-eopf#140``, but a benchmark's own numbers
    shouldn't depend on remembering that a *specific* reader's default
    happened to be safe).

    ``path_prefix`` selects the tiler router — ``"cog"`` for a COG against
    TiTiler's COG endpoints, ``"zarr"``/``"geozarr"`` for a GeoZarr store's
    stock-xarray or ``GeoZarrReader`` router — and ``extra_query`` carries any
    extra query parameters that router needs (e.g. ``{"variable": "data"}`` to
    pick the array), the same for every tile.

    ``group_query_key`` (#116) is the escape hatch from that "same for every
    tile" default: when set, a tile that carries its own ``.group`` (a
    resolution-scenario tile addressing the GeoZarr level that actually
    serves it, not the chunk-bucket tiles alongside it, which carry none)
    gets that one query key overridden to ``spec.group`` for *its own*
    fetch only. Without this, the stock xarray router — which has no
    multiscale awareness of its own — would read the same (typically
    native) level for every tile regardless of the zoom being tested,
    making it structurally incapable of showing the store's pyramid have
    any effect. ``None`` (the default) preserves today's one-query-for-
    every-tile behaviour exactly.

    Every metric's ``detail`` records ``path_prefix``, so a report is never
    ambiguous about which reader produced a given number.
    """
    base = endpoint.rstrip("/")
    encoded = quote(cog_uri, safe="")
    prefix = f"/{path_prefix.strip('/')}" if path_prefix.strip("/") else ""
    base_query = dict(extra_query or {})
    extra = "".join(f"&{quote(k)}={quote(str(v))}" for k, v in base_query.items())

    # Validate the object is servable before timing tiles (clearer failures).
    _fetch(f"{base}{prefix}/info?url={encoded}{extra}", timeout)

    metrics: list[MetricResult] = []
    for spec in tiles:
        tile_query = base_query
        if group_query_key and spec.group is not None:
            tile_query = {**base_query, group_query_key: spec.group}
        tile_extra = "".join(
            f"&{quote(k)}={quote(str(v))}" for k, v in tile_query.items()
        )
        tile_url = (
            f"{base}{prefix}/tiles/{tile_matrix_set}/"
            f"{spec.z}/{spec.x}/{spec.y}.{fmt}"
            f"?url={encoded}&tilesize={tilesize}{tile_extra}"
        )
        start = time.perf_counter()
        body = _fetch(tile_url, timeout)
        latency = time.perf_counter() - start

        metrics.append(
            MetricResult(
                name=f"display_{spec.label}_latency",
                value=latency,
                unit="s",
                detail={
                    "tile": f"{spec.z}/{spec.x}/{spec.y}",
                    "chunks": spec.chunks,
                    "approx": spec.approx,
                    "bytes": len(body),
                    "path_prefix": path_prefix,
                    "group": spec.group,
                },
            )
        )

    metrics.append(
        MetricResult(
            name="display_scenarios",
            value=len(tiles),
            detail={
                "tile_matrix_set": tile_matrix_set,
                "tilesize": tilesize,
                "path_prefix": path_prefix,
                "scenarios": [
                    {
                        "label": s.label,
                        "tile": f"{s.z}/{s.x}/{s.y}",
                        "chunks": s.chunks,
                        "group": s.group,
                    }
                    for s in tiles
                ],
            },
        )
    )
    return metrics
