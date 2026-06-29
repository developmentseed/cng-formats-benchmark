"""Cloud-Optimized GeoTIFF adapter.

The grouping lever for COG is its internal tiling (block size) and overview
layout, which together determine how many byte ranges a reader must fetch. The
adapter converts a baseline raster to a COG with rio-cogeo (the ``cog`` extra)
and reports the produced object's size; a COG is a single addressable object, so
``enumerate_objects`` returns one size.
"""

from __future__ import annotations

import os
from typing import Any

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import CogLayout
from cng_benchmark.registry import FORMATS

#: Default internal tile (block) size when the config carries no lever value.
DEFAULT_BLOCK_SIZE = 512


def _require_geo():
    """Import the geo stack, raising a clear error if the ``cog`` extra is absent."""
    try:
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "COG conversion requires the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return cog_translate, cog_profiles


@FORMATS.register("cog")
class CogAdapter(FormatAdapter):
    name = "cog"
    object_kind = ObjectKind.RASTER_FILE

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (a GDAL-readable raster) to a COG at ``target``.

        The grouping lever is the internal block size, taken from
        ``params['block_size']`` (default :data:`DEFAULT_BLOCK_SIZE`); a list
        value uses its first element so a swept lever degrades to a single run.
        Compression defaults to deflate and can be overridden with
        ``params['compress']``.
        """
        cog_translate, cog_profiles = _require_geo()

        block = params.get("block_size", DEFAULT_BLOCK_SIZE)
        if isinstance(block, list | tuple):
            block = block[0]
        block = int(block)

        profile = cog_profiles.get(str(params.get("compress", "deflate")))
        profile.update(blockxsize=block, blockysize=block)
        kwargs: dict = {}
        if "nodata" in params:
            kwargs["nodata"] = params["nodata"]
        cog_translate(source, target, profile, quiet=True, **kwargs)

    def describe_grouping_lever(self) -> str:
        return "COG internal tiling (block size) and overview layout"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the size (bytes) of the produced COG — a single object."""
        return [os.path.getsize(target)]

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[CogLayout]:
        """Return the produced COG's internal tiling layout (one object)."""
        from cng_benchmark.metrics.layout import describe_cog_layout

        return [describe_cog_layout(name or self.name, target, os.path.getsize(target))]
