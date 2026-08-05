"""CO3D CARS reader — the tiled-LAZ point-cloud delivery.

The fourth scoped mission. The CARS pipeline (the CO3D stereo reconstruction)
delivers its point cloud as **tiled LAZ**: one LAS/LAZ file per ground tile,
under a delivery directory. So the *product* is a set of tiles, not a single
granule — the opposite shape to the SWOT readers, where one granule file is one
product. Each tile becomes one **component**, keeping the tile identity in the
component name, so the run's object-size distribution is *per tile*.

Tiles are grouped into products by the directory that holds them, relative to
``source``: one delivery = one product. A root staging a single delivery (tiles
flat under ``source``, or all under one ``point_cloud/`` directory) therefore
yields one product whose components are its tiles, and a root staging several
deliveries yields one product each — so ``prefix``/``pattern``/``limit`` keep the
same meaning here as in the granule readers (they bound the *product* set).

Each tile is converted to a COPC file by the COPC adapter, whose point loader
reads a ``.las``/``.laz`` path with laspy — the same adapter and point-cloud path
as the SWOT PIXC arm (:mod:`cng_benchmark.datasets.swot_pixc`), reused unchanged.

A single tile is small — it may sit below Tier 2 — so which grouping lever
applies here is the open question this arm answers: the COPC octree node budget
*inside* one tile, or *merging* tiles into one COPC. This reader implements the
first (one tile = one object); merging is a separate aggregation lever.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import field_validator

from cng_benchmark import storage
from cng_benchmark.datasets.base import Dataset, DatasetOptions, Product, SourceObject
from cng_benchmark.registry import DATASETS

#: Tile extensions the COPC adapter's point loader can read (see
#: :func:`cng_benchmark.formats.copc._load_points`).
POINT_SUFFIXES = (".laz", ".las")

#: The CARS delivery extension — compressed LAZ — when ``options`` carries none.
DEFAULT_TILE_SUFFIX = ".laz"


def _relative_key(source_uri: str, tile_uri: str) -> str:
    """Return the tile's key *relative to* the dataset root."""
    base = source_uri.split("://", 1)[-1].rstrip("/")
    key = tile_uri.split("://", 1)[-1]
    prefix = base + "/"
    return key[len(prefix) :] if base and key.startswith(prefix) else key


def _tile_name(tile_uri: str, suffix: str) -> str:
    """Derive a stable tile id from a tile URI (filename without ``suffix``).

    The name is flat (no path separator): it labels the component's results and
    lays out the produced object, and tiles of one delivery share a directory, so
    the filename alone is unique within a product.
    """
    name = PurePosixPath(tile_uri.split("://", 1)[-1]).name
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


class Co3dCarsOptions(DatasetOptions):
    """Tile-selection picks for a CARS delivery.

    ``tile_suffix`` is the delivered tile extension — ``.laz`` for the compressed
    CARS output, ``.las`` for an uncompressed staging. It is matched literally
    against the object keys (so an upper-case staging sets ``.LAZ``) and is
    restricted to what the COPC point loader reads, :data:`POINT_SUFFIXES`, so a
    typo fails at construction rather than at conversion time.
    """

    tile_suffix: str = DEFAULT_TILE_SUFFIX

    @field_validator("tile_suffix")
    @classmethod
    def _readable_suffix(cls, value: str) -> str:
        if value.lower() not in POINT_SUFFIXES:
            raise ValueError(
                f"tile_suffix {value!r} is not a point-cloud tile extension; "
                f"expected one of {', '.join(POINT_SUFFIXES)}"
            )
        return value


@DATASETS.register("co3d-cars")
class Co3dCarsDataset(Dataset):
    """Enumerate a CARS delivery's LAZ tiles — one product, one component per tile."""

    Options = Co3dCarsOptions

    def products(
        self,
        *,
        prefix: str | None = None,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> list[Product]:
        # A delivery is *defined by* its tiles, so the listing bound (``limit``)
        # is applied to the grouped products, not to the tile listing — capping
        # the listing would truncate a delivery mid-way and profile a partial
        # tile set. Narrow a large root server-side with ``prefix`` instead.
        opts: Co3dCarsOptions = self.options
        tiles = storage.list_uris(
            self.source_uri,
            role="source",
            prefix=prefix,
            suffix=opts.tile_suffix,
            pattern=pattern,
        )

        deliveries: dict[str, list[SourceObject]] = {}
        for tile_uri in tiles:
            relative = _relative_key(self.source_uri, tile_uri)
            directory = str(PurePosixPath(relative).parent)
            deliveries.setdefault(directory, []).append(
                SourceObject(name=_tile_name(tile_uri, opts.tile_suffix), uri=tile_uri)
            )

        products = [
            Product(id=self._delivery_id(directory), components=components)
            for directory, components in deliveries.items()
        ]
        return products[:limit] if limit is not None else products

    def _delivery_id(self, directory: str) -> str:
        """Name the product holding the tiles in ``directory`` (relative to root).

        A nested delivery is named after its directory (path separators flattened
        so the id is one path segment); tiles sitting directly under ``source``
        take the root's own name — the staged delivery's name — falling back to
        the dataset id for a root that has none.
        """
        if directory in ("", "."):
            root = PurePosixPath(self.source_uri.split("://", 1)[-1].rstrip("/")).name
            return root or self.id
        return directory.strip("/").replace("/", "_")
