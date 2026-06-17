"""GeoZarr v3 adapter (stub).

The grouping lever for Zarr v3 is its chunk and shard shape: sharding packs many
chunks into a single object, directly controlling the mean object size.
"""

from __future__ import annotations

from typing import Any

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.registry import FORMATS


@FORMATS.register("geozarr")
class GeoZarrAdapter(FormatAdapter):
    name = "geozarr"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        raise NotImplementedError("GeoZarr conversion lands with the stack")

    def describe_grouping_lever(self) -> str:
        return "Zarr v3 chunk and shard shape"

    def enumerate_objects(self, target: str) -> list[int]:
        raise NotImplementedError("GeoZarr object enumeration lands with the stack")
