"""GeoParquet adapter (stub).

The grouping lever for GeoParquet is its row-group size and any file-level
partitioning, which together determine how features are grouped into objects.
"""

from __future__ import annotations

from typing import Any

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.registry import FORMATS


@FORMATS.register("geoparquet")
class GeoParquetAdapter(FormatAdapter):
    name = "geoparquet"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        raise NotImplementedError("GeoParquet conversion lands with the stack")

    def describe_grouping_lever(self) -> str:
        return "GeoParquet row-group size and file partitioning"

    def enumerate_objects(self, target: str) -> list[int]:
        raise NotImplementedError("GeoParquet object enumeration lands with the stack")
