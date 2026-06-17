"""Cloud-Optimized GeoTIFF adapter (stub).

The grouping lever for COG is its internal tiling (block size) and overview
layout, which together determine how many byte ranges a reader must fetch.
"""

from __future__ import annotations

from typing import Any

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.registry import FORMATS


@FORMATS.register("cog")
class CogAdapter(FormatAdapter):
    name = "cog"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        raise NotImplementedError("COG conversion lands with the deployable stack")

    def describe_grouping_lever(self) -> str:
        return "COG internal tiling (block size) and overview layout"

    def enumerate_objects(self, target: str) -> list[int]:
        raise NotImplementedError("COG object enumeration lands with the stack")
