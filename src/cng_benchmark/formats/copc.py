"""Cloud-Optimized Point Cloud adapter (stub).

The grouping lever for COPC is its octree node structure: the octree depth and
per-node point budget set how points are grouped into addressable chunks.
"""

from __future__ import annotations

from typing import Any

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.registry import FORMATS


@FORMATS.register("copc")
class CopcAdapter(FormatAdapter):
    name = "copc"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        raise NotImplementedError("COPC conversion lands with the stack")

    def describe_grouping_lever(self) -> str:
        return "COPC octree depth and per-node point budget"

    def enumerate_objects(self, target: str) -> list[int]:
        raise NotImplementedError("COPC object enumeration lands with the stack")
