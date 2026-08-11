"""Cloud-Optimized GeoTIFF adapter.

The grouping lever for COG is its internal tiling (block size) and overview
layout, which together determine how many byte ranges a reader must fetch. The
adapter converts a baseline raster to a COG with rio-cogeo (the ``cog`` extra)
and reports the produced object's size; a COG is a single addressable object, so
``enumerate_objects`` returns one size.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.formats.grid import GridKey, group_by_grid
from cng_benchmark.models import CogLayout
from cng_benchmark.registry import FORMATS

if TYPE_CHECKING:
    from cng_benchmark.datasets.base import SourceObject

#: Default internal tile (block) size when the config carries no lever value.
DEFAULT_BLOCK_SIZE = 512

#: Default codec when the config carries no `codec` value — GDAL's traditional
#: COG default, and the deflate side of the matched-codec comparison (#72).
DEFAULT_CODEC = "deflate"


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


def _profile_kwargs(cog_profiles, params: dict[str, Any]):
    """Build the rio-cogeo profile from the shared ``block_size``/``codec`` levers."""
    block = params.get("block_size", DEFAULT_BLOCK_SIZE)
    if isinstance(block, list | tuple):
        block = block[0]
    profile = cog_profiles.get(str(params.get("codec", DEFAULT_CODEC)))
    profile.update(blockxsize=int(block), blockysize=int(block))
    return profile


@FORMATS.register("cog")
class CogAdapter(FormatAdapter):
    name = "cog"
    object_kind = ObjectKind.RASTER_FILE
    supports_batch = True

    def __init__(self) -> None:
        # Batch-time bookkeeping: which 1-based band a component landed on,
        # keyed by target file path — describe_layout/component_locator read
        # it back for a target this instance just wrote via convert_batch
        # (#102, mirrors the FlatGeobuf null_geometry bookkeeping pattern
        # from #98 and GeoZarrAdapter's own grid bookkeeping).
        self._batch_bands_by_target: dict[str, dict[str, int]] = {}

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (a GDAL-readable raster) to a COG at ``target``.

        The grouping lever is the internal block size, taken from
        ``params['block_size']`` (default :data:`DEFAULT_BLOCK_SIZE`); a list
        value uses its first element so a swept lever degrades to a single run.
        The codec is taken from ``params['codec']`` (default
        :data:`DEFAULT_CODEC`; e.g. ``deflate`` | ``zstd``) — the COG side of the
        matched-codec comparison against GeoZarr's own ``codec`` param (#72).
        """
        cog_translate, cog_profiles = _require_geo()
        profile = _profile_kwargs(cog_profiles, params)
        kwargs: dict = {}
        if "nodata" in params:
            kwargs["nodata"] = params["nodata"]
        cog_translate(source, target, profile, quiet=True, **kwargs)

    def convert_batch(
        self, sources: list[SourceObject], target: str, params: dict[str, Any]
    ) -> None:
        """Convert every one of ``sources`` into one multi-band COG at ``target``.

        Requires every source to share one grid (shape, transform, CRS) — unlike
        GeoZarr's store, a single GeoTIFF cannot hold more than one grid, so a
        mismatched product raises naming the odd component(s) out rather than
        attempting multiple output files (a secondary, smaller ask than GeoZarr
        bundling, per the issue). Also requires a homogeneous NODATA across the
        group: the stacking VRT declares one per band, but GDAL's translate only
        carries a single dataset-level value through ``cog_translate`` (verified
        empirically for #102 — differing per-component NODATA would otherwise
        silently apply the wrong value to every band but the first), so this
        raises rather than measuring that silently.
        """
        from cng_benchmark import vrt

        cog_translate, cog_profiles = _require_geo()

        grids: dict[str, vrt.GridMeta] = {}
        for src in sources:
            grid = vrt.read_grid(src.uri)
            # An explicit override always wins (component's own nodata, else
            # the run's shared params.nodata) — mirrors convert()'s single-
            # source `kwargs["nodata"] = params["nodata"]`, which forces the
            # value onto the destination regardless of what the source
            # declares, not just when the source is silent about it.
            override = src.nodata if src.nodata is not None else params.get("nodata")
            if override is not None:
                grid = dataclasses.replace(grid, nodata=float(override))
            grids[src.name] = grid

        keys = [
            GridKey(
                name=name,
                shape=(g.height, g.width),
                geotransform=g.transform,
                crs=g.crs_wkt,
            )
            for name, g in grids.items()
        ]
        groups = group_by_grid(keys)
        if len(groups) > 1:
            detail = "; ".join("{" + ", ".join(g) + "}" for g in groups)
            raise ValueError(
                "cog bundling requires every component to share one grid "
                f"(shape/transform/CRS); got {len(groups)} distinct grids: {detail}"
            )

        nodatas = {grids[name].nodata for name in grids}
        if len(nodatas) > 1:
            detail = ", ".join(f"{name}={grids[name].nodata}" for name in grids)
            raise ValueError(
                "cog bundling requires every component to share one NODATA "
                f"value (GDAL only carries one dataset-level value through "
                f"cog_translate); got {detail}"
            )

        ordered_names = [src.name for src in sources]
        xml = vrt.build_stack_vrt_xml([grids[n] for n in ordered_names], ordered_names)
        vrt_path = os.path.join(os.path.dirname(target) or ".", "_stack.vrt")
        with open(vrt_path, "w") as fh:
            fh.write(xml)

        profile = _profile_kwargs(cog_profiles, params)
        kwargs: dict = {}
        resolved_nodata = next(iter(nodatas), None)
        if resolved_nodata is not None:
            kwargs["nodata"] = resolved_nodata
        cog_translate(vrt_path, target, profile, quiet=True, **kwargs)

        self._batch_bands_by_target[target] = {
            name: i for i, name in enumerate(ordered_names, start=1)
        }

    def component_locator(self, target: str, name: str) -> str | None:
        """The 1-based band index ``name`` landed on, for a batched ``target``."""
        band = self._batch_bands_by_target.get(target, {}).get(name)
        return None if band is None else str(band)

    def describe_grouping_lever(self) -> str:
        return "COG internal tiling (block size) and overview layout"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the size (bytes) of the produced COG — a single object.

        Unchanged for a batched target: bundling several components into one
        multi-band file *is* the object-count reduction — N single-band files
        become one, regardless of how many bands it holds.
        """
        return [os.path.getsize(target)]

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[CogLayout]:
        """Return the produced COG's internal tiling layout (one object).

        Bytes aren't separable per band in an interleaved multi-band GeoTIFF,
        so a target :meth:`convert_batch` wrote still gets one ``CogLayout`` —
        not one per component — with ``band_names`` recording which band is
        which.
        """
        from cng_benchmark.metrics.layout import describe_cog_layout

        bands = self._batch_bands_by_target.get(target)
        band_names = None
        if bands is not None:
            band_names = [n for n, _i in sorted(bands.items(), key=lambda kv: kv[1])]
        return [
            describe_cog_layout(
                name or self.name,
                target,
                os.path.getsize(target),
                band_names=band_names,
            )
        ]
