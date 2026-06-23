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
from enum import StrEnum
from typing import ClassVar

from cng_benchmark.models import ObjectLayout


class ObjectKind(StrEnum):
    """The kind of object an adapter materialises at the conversion target.

    A COG is a single ``RASTER_FILE`` (one openable raster); a GeoZarr array is a
    ``ZARR_STORE`` (a directory tree of shard objects + metadata); a GeoParquet
    file is a single ``VECTOR_FILE`` (one openable table whose row groups are the
    addressable byte ranges). The runner branches on this for the few things that
    genuinely differ per kind — output naming, upload (single file vs tree), the
    read collector (rasterio window vs zarr-native chunk vs vector bbox query) and
    the display surface — while the conversion contract itself is shared. It
    describes the *materialised object*; a future time-stacked cube is a separate
    concern layered on top, not a change to this.
    """

    RASTER_FILE = "raster_file"
    ZARR_STORE = "zarr_store"
    VECTOR_FILE = "vector_file"


class FormatAdapter(ABC):
    """Convert a baseline dataset to a target format and describe its objects."""

    #: Stable short name the adapter is registered under (e.g. ``"cog"``).
    name: str

    #: What the adapter writes at ``target`` (see :class:`ObjectKind`).
    object_kind: ClassVar[ObjectKind] = ObjectKind.RASTER_FILE

    @abstractmethod
    def convert(self, source: str, target: str, params: dict[str, object]) -> None:
        """Convert ``source`` to the target format, writing to ``target``.

        ``params`` carries the grouping-lever settings for this run.
        """

    @abstractmethod
    def describe_grouping_lever(self) -> str:
        """Return a human-readable description of the object-grouping lever."""

    @abstractmethod
    def enumerate_objects(self, target: str) -> list[int]:
        """Return the sizes (bytes) of the objects produced at ``target``."""

    def target_basename(self) -> str:
        """Return the local filename/dirname the runner converts into.

        A raster-file adapter writes one file (``cog.tif``); a store adapter writes
        a directory (``geozarr.zarr``). Defaults to ``<name>.tif`` for the existing
        raster adapters.
        """
        return f"{self.name}.tif"

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[ObjectLayout]:
        """Describe the partial-access layout of the object(s) at ``target``.

        Returns one typed :class:`~cng_benchmark.models.ObjectLayout` subclass per
        produced object (e.g. a ``CogLayout`` per COG, a ``GeoZarrLayout`` per
        array). ``name`` is the object label the runner uses (the component name,
        falling back to the adapter name). The structural sibling of
        :meth:`describe_grouping_lever`; the default is empty for adapters that
        have not implemented it yet.
        """
        return []
