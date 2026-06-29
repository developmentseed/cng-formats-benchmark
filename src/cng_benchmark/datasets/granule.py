"""Granule delivery base — one file per granule under ``source``, no zip.

The other layout family the harness sees: a flat directory of self-contained
granule *files* (the SWOT netCDF-raster shape), as opposed to the zip-per-scene
delivery (:mod:`cng_benchmark.datasets.zip_delivery`). Each granule is a single
file under the dataset ``source``; this base enumerates them and hands each
granule URI to the subclass to choose which **components** it yields. What a
component *is* — a whole raster, or a named variable inside a multi-variable
container — is layout-specific and lives in the subclass; this base is
layout-agnostic.

A granule that holds many CF variables (a netCDF/HDF5 file) is addressed
**per variable** through GDAL's subdataset syntax (:func:`_subdataset_vsi_uri`),
so a selected variable is read in place — no pre-extraction — and the
conversion's write metric pays the real granule read cost, exactly as the
zip-delivery readers read their members on the fly.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import PurePosixPath
from typing import ClassVar

from cng_benchmark import storage
from cng_benchmark.datasets.base import Dataset, Product, SourceObject


def _granule_id(granule_uri: str, suffix: str) -> str:
    """Derive a stable granule id from a granule URI (filename without ``suffix``)."""
    name = PurePosixPath(granule_uri.split("://", 1)[-1]).name
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def _subdataset_vsi_uri(granule_uri: str, variable: str) -> str:
    """Compose a GDAL subdataset path addressing ``variable`` inside ``granule_uri``.

    The granule is mapped to its GDAL path first (``/vsis3/…`` for S3, a local
    path otherwise) and wrapped in GDAL's CF subdataset syntax
    (``NETCDF:"<path>":<variable>``), so a single CF variable is read in place
    via rioxarray/GDAL. The result is passed through by
    :func:`storage.to_gdal_path` unchanged (it is neither an ``s3://`` nor a
    ``file://`` URI).
    """
    return f'NETCDF:"{storage.to_gdal_path(granule_uri)}":{variable}'


class GranuleDataset(Dataset):
    """One granule file per product under ``source``; components chosen by subclass."""

    #: File suffix identifying a granule under ``source`` (subclass overrides).
    granule_suffix: ClassVar[str] = ".nc"

    def products(
        self,
        *,
        prefix: str | None = None,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> list[Product]:
        # Bound enumeration server-side, like the zip-delivery base: ``prefix`` is
        # a path prefix under the dataset root and ``limit`` caps the listing, so a
        # huge granule root is never paginated in full to find a handful (see #14).
        # ``pattern`` regex-filters the candidates (e.g. a UTM-zone subset for a
        # SWOT-raster mosaic) where a single path-prefix is not enough.
        granules = storage.list_uris(
            self.source_uri,
            role="source",
            prefix=prefix,
            suffix=self.granule_suffix,
            pattern=pattern,
            limit=limit,
        )

        products: list[Product] = []
        for granule_uri in granules:
            components = self._select_components(granule_uri)
            if components:
                products.append(
                    Product(
                        id=_granule_id(granule_uri, self.granule_suffix),
                        components=components,
                    )
                )
        return products

    @abstractmethod
    def _select_components(self, granule_uri: str) -> list[SourceObject]:
        """Pick the components from a ``granule_uri`` (subclass-specific).

        Returns a :class:`SourceObject` per chosen component with a stable
        ``name`` and an on-the-fly ``uri`` — a whole-file GDAL path, or a CF
        subdataset path (see :func:`_subdataset_vsi_uri`).
        """
