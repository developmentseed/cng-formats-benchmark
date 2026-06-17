"""Format adapter contract.

A :class:`FormatAdapter` converts a dataset from its baseline format into a
target cloud-native format and exposes the *grouping lever* — the format-specific
knob that controls how bytes are grouped into addressable objects (COG internal
tiling, Zarr v3 sharding, COPC octree, GeoParquet row groups). Sweeping that
lever and profiling the resulting object sizes is the core of the benchmark.

This module defines only the contract. Concrete adapters are thin, registered
subclasses; the actual conversion and object enumeration require GDAL/Zarr/PDAL
and real IO and are implemented with the deployable stack (M2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FormatAdapter(ABC):
    """Convert a baseline dataset to a target format and describe its objects."""

    #: Stable short name the adapter is registered under (e.g. ``"cog"``).
    name: str

    @abstractmethod
    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` to the target format, writing to ``target``.

        ``params`` carries the grouping-lever settings for this run.
        """

    @abstractmethod
    def describe_grouping_lever(self) -> str:
        """Return a human-readable description of the object-grouping lever."""

    @abstractmethod
    def enumerate_objects(self, target: str) -> list[int]:
        """Return the sizes (bytes) of the objects produced at ``target``."""
