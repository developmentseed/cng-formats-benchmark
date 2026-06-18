"""Zip-per-scene delivery base.

A delivery layout where each scene is a single ``.zip`` under the dataset
``source`` (the S2/S1 ground-segment shape). This base enumerates the scenes,
reads each zip's central directory **once** (a cheap range read, never the whole
archive), and hands the member listing to the subclass to choose which members
become the product's components. Member selection (which name is which band or
mask) is layout-specific and lives in the subclass; this base is layout-agnostic.

Selected members are read **on the fly** through GDAL's ``/vsizip//vsis3`` chain
— no pre-extraction — so the conversion's write metric pays the real archive
read cost, exactly as a migration would.
"""

from __future__ import annotations

import zipfile
from abc import abstractmethod
from pathlib import PurePosixPath

from cng_benchmark import storage
from cng_benchmark.datasets.base import Dataset, Product, SourceObject


def _scene_id(zip_uri: str) -> str:
    """Derive a stable scene id from a zip URI (its filename without ``.zip``)."""
    name = PurePosixPath(zip_uri.split("://", 1)[-1]).name
    return name[: -len(".zip")] if name.endswith(".zip") else name


def _member_vsi_uri(zip_uri: str, member: str) -> str:
    """Compose a GDAL ``/vsizip`` path addressing ``member`` inside ``zip_uri``.

    The zip itself is mapped to its GDAL path first (``/vsis3/…`` for S3, a local
    path otherwise), then wrapped in ``/vsizip/`` so the member is read in place.
    The result is passed through by :func:`storage.to_gdal_path` unchanged.
    """
    return f"/vsizip/{storage.to_gdal_path(zip_uri)}/{member}"


class ZipDeliveryDataset(Dataset):
    """One ``.zip`` per scene under ``source``; members chosen by the subclass."""

    def products(
        self, *, prefix: str | None = None, limit: int | None = None
    ) -> list[Product]:
        # Bound enumeration server-side: ``prefix`` is a path prefix under the
        # dataset root and ``limit`` caps the listing, so a huge tile root is
        # never paginated in full to find a handful of scenes (see #14).
        zips = storage.list_uris(
            self.source_uri,
            role="source",
            prefix=prefix,
            suffix=".zip",
            limit=limit,
        )

        products: list[Product] = []
        for zip_uri in zips:
            members = self._zip_member_names(zip_uri)
            components = self._select_members(members, zip_uri)
            if components:
                products.append(Product(id=_scene_id(zip_uri), components=components))
        return products

    def _zip_member_names(self, zip_uri: str) -> list[str]:
        """Return the member names in ``zip_uri`` (central directory read only)."""
        fileobj = storage.open_seekable(zip_uri, role="source")
        try:
            with zipfile.ZipFile(fileobj) as zf:
                return zf.namelist()
        finally:
            fileobj.close()

    @abstractmethod
    def _select_members(self, members: list[str], zip_uri: str) -> list[SourceObject]:
        """Pick the components from a zip's ``members`` (subclass-specific).

        Returns a :class:`SourceObject` per chosen member with a stable ``name``
        and an on-the-fly ``/vsizip`` ``uri`` (see :func:`_member_vsi_uri`).
        """
