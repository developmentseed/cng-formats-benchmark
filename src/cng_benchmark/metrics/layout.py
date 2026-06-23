"""Tiling-layout collector — the internal structure of a produced object.

Reads a produced raster's *internal* layout (block/tile size, overview
decimations, tiled vs striped) and records it as an
:class:`~cng_benchmark.models.ObjectLayout`. This is the structural companion to
the object-size profile: size says whether an object clears a storage tier, the
layout says whether a reader can fetch part of it with a range request — the
core partial-access question, and the same structure the chunk-aware display
metric reads to bucket its tiles. It is captured per produced object, needs no
tile server, and is cheap (a header read).

Requires rasterio (the ``cog`` extra); the caller treats an unreadable output
(e.g. a non-raster format) as "no layout" rather than a failure.
"""

from __future__ import annotations

import math

from cng_benchmark.models import CogLayout


def _require_geo():
    try:
        import rasterio
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the tiling-layout collector requires the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return rasterio


def describe_cog_layout(name: str, path: str, size_bytes: int) -> CogLayout:
    """Return the :class:`CogLayout` of the raster at ``path``.

    ``is_tiled`` is true when the block does not span the full raster width —
    i.e. the file is internally tiled (a COG), not striped. ``internal_tiles`` is
    the block-grid cell count at full resolution. Raises ``RuntimeError`` when the
    geo stack is missing and propagates rasterio errors for an unreadable raster,
    so the caller can decide whether a missing layout is acceptable.
    """
    rasterio = _require_geo()
    with rasterio.open(path) as src:
        block_h, block_w = src.block_shapes[0]
        width, height = src.width, src.height
        tiled = bool(src.profile.get("tiled", False)) or block_w < width
        decimations = [int(d) for d in src.overviews(1)]
        internal_tiles = math.ceil(width / block_w) * math.ceil(height / block_h)
    return CogLayout(
        name=name,
        size_bytes=size_bytes,
        is_tiled=tiled,
        block_height=int(block_h),
        block_width=int(block_w),
        overview_decimations=decimations,
        internal_tiles=int(internal_tiles),
    )
