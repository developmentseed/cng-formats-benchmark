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
from typing import TYPE_CHECKING, ClassVar

from cng_benchmark.models import ObjectLayout

if TYPE_CHECKING:
    from cng_benchmark.datasets.base import SourceObject


class ObjectKind(StrEnum):
    """The kind of object an adapter materialises at the conversion target.

    A COG is a single ``RASTER_FILE`` (one openable raster); a GeoZarr array is a
    ``ZARR_STORE`` (a directory tree of shard objects + metadata); a GeoParquet or
    FlatGeobuf file is a single ``VECTOR_FILE`` (one openable table whose row
    groups — or whose R-tree-indexed features — are the addressable byte ranges);
    a COPC file is a single ``POINT_CLOUD_FILE`` (one
    openable LAZ whose octree nodes are the addressable byte ranges). The runner
    branches on this for the few things that genuinely differ per kind — output
    naming, upload (single file vs tree), the read collector (rasterio window vs
    zarr-native chunk vs vector bbox query vs octree-node fetch) and the display
    surface — while the conversion contract itself is shared. It describes the
    *materialised object*; a future time-stacked cube is a separate concern layered
    on top, not a change to this.
    """

    RASTER_FILE = "raster_file"
    ZARR_STORE = "zarr_store"
    VECTOR_FILE = "vector_file"
    POINT_CLOUD_FILE = "point_cloud_file"


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

    #: Whether this adapter can bundle several dataset components into one
    #: produced object via :meth:`convert_batch` (#102). ``False`` by default —
    #: the runner's per-component loop is the only path unless both this is
    #: ``True`` *and* the benchmark config opts in (``params.bundle_components``);
    #: bundling is never automatic just because a format could support it.
    supports_batch: ClassVar[bool] = False

    def convert_batch(
        self, sources: list[SourceObject], target: str, params: dict[str, object]
    ) -> None:
        """Convert every one of ``sources`` into one bundled object at ``target``.

        Only called when :attr:`supports_batch` is ``True`` and a benchmark
        config's ``params.bundle_components`` names this group. ``sources``
        carries each component's own ``nodata``/``scale_factor``/
        ``standard_name`` (the same per-component metadata the non-batched path
        merges into ``params`` in the runner) rather than a bare path, since a
        batched write still has to honour it per component.

        After this returns, :meth:`component_locator` must be able to say
        where each source's data landed inside ``target``.
        """
        raise NotImplementedError(f"{self.name} does not support batched conversion")

    def component_locator(self, target: str, name: str) -> str | None:
        """Where ``name``'s data lives within a batched ``target``.

        A format-specific address a read/display collector can act on — a
        Zarr group path for GeoZarr, a 1-based band index for COG — or
        ``None`` when ``target`` was not produced by :meth:`convert_batch`
        (every non-batched adapter, and any target this instance did not just
        write). The default adapter never batches, so it is always ``None``.
        """
        return None

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
