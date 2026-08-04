"""Dataset contract — the layout-aware enumeration seam.

A :class:`Dataset` is the harness's handle on the source data named by a
:class:`~cng_benchmark.config.DatasetConfig`. It exposes the config-derived
identity (id, baseline and target formats) generically and, crucially,
enumerates the **products** and the **components** within them that the runner
converts and profiles.

The single-source assumption (one source raster → one output object) is lifted
here: a real delivery is a *product* of many *components* — a MAJA L2A scene is
~15-20 rasters (FRE/SRE reflectance bands plus CLM/EDG/SAT/MG2 masks) — and a
benchmark may cover a *set* of products. The shape is therefore
``SourceObject {name, uri}`` → ``Product {id, components}`` →
``Dataset.products() -> list[Product]``.

Layout families share intermediate bases (a zip-per-scene delivery, one
netCDF file per granule, …); each leaf **owns a typed pydantic ``Options``
model** (its layout-specific picks, e.g. reflectance/bands/masks) parsed from
``config.options`` in the constructor. Adding a layout is a new subclass + its
``Options`` + one registry line — no change to the core config or runner.
Member-name patterns (``FRE_B3.tif``, ``MASKS/…_CLM_R1.tif``) live in the
subclass, never in shared config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from cng_benchmark.config import DatasetConfig


@dataclass(frozen=True)
class SourceObject:
    """One addressable baseline input within a product.

    ``uri`` is anything the storage/GDAL layer can open: a local path, an
    ``s3://`` URI, or an already-composed VSI path (e.g.
    ``/vsizip//vsis3/bucket/scene.zip/FRE_B3.tif`` for a zip member read on the
    fly). ``name`` is a short, stable identifier used to lay out the produced
    object and to label per-component results.

    ``nodata`` is the fill value for this component's pixel data and
    ``scale_factor`` the CF multiplier that turns a stored count into its
    physical unit (``physical = stored * scale_factor``). Both are set by
    readers that know the product convention — MAJA FRE/SRE reflectance is
    stored as ``int16`` DN with -10000 as fill and a 1/10000 quantification —
    because the delivered rasters declare neither in their headers. ``None``
    means "whatever the source declares".

    ``scale_offset`` asks the target format to apply ``scale_factor`` in the
    object's own encoding, rather than leaving a client to unscale — a request
    only some formats can honour, which is what makes it worth measuring.
    """

    name: str
    uri: str
    nodata: float | None = None
    scale_factor: float | None = None
    scale_offset: bool = False


@dataclass(frozen=True)
class Product:
    """A delivery unit — one scene/granule — and its baseline components."""

    id: str
    components: list[SourceObject]


@dataclass(frozen=True)
class RgbComposite:
    """Three component names to stack as a (Red, Green, Blue) viewer VRT.

    ``name`` is the composite's label (``"natural"``, ``"color-infrared"``,
    ``"swir"``, ``"dualpol"``); ``bands`` is a 3-tuple of component names in
    R/G/B order; ``rescale`` is a ``(lo, hi)`` stretch hint for TiTiler (the
    reflectance/gamma0 values are not 8-bit, so the viewer needs a hint to
    display them).  Owned by the dataset reader — layout knowledge lives here,
    not in the runner.
    """

    name: str
    bands: tuple[str, str, str]
    rescale: tuple[float, float] | None = None


@dataclass(frozen=True)
class SingleBandComposite:
    """One component name to mosaic as a single-band (Gray) viewer VRT.

    Used for scalar datasets whose products have a single meaningful variable
    (e.g. SWOT WSE, DEM elevation) where an RGB composite is not meaningful.
    ``band`` matches a component name; ``rescale`` is an optional ``(lo, hi)``
    stretch hint for TiTiler.  Owned by the dataset reader.
    """

    name: str
    band: str
    rescale: tuple[float, float] | None = None


class DatasetOptions(BaseModel):
    """Base for a reader's typed options block.

    ``extra="forbid"`` makes an unknown key in ``config.options`` a validation
    error rather than a silently ignored typo — the descriptor is the user
    interface, so its mistakes should be loud.
    """

    model_config = ConfigDict(extra="forbid")


class Dataset(ABC):
    """A source dataset constructed from its config.

    Subclasses set :attr:`Options` to their typed options model; the base
    constructor validates ``config.options`` against it, so a malformed
    descriptor fails fast at construction. The default :class:`DatasetOptions`
    accepts no fields, which is what ``single-object`` and other option-free
    readers want.
    """

    #: Per-layout typed options model; leaves override it.
    Options: ClassVar[type[DatasetOptions]] = DatasetOptions

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.options = type(self).Options.model_validate(config.options or {})

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def baseline_format(self) -> str:
        return self.config.baseline_format

    @property
    def target_formats(self) -> list[str]:
        return self.config.target_formats

    @property
    def source_uri(self) -> str:
        return self.config.source

    @abstractmethod
    def products(
        self,
        *,
        prefix: str | None = None,
        pattern: str | None = None,
        limit: int | None = None,
    ) -> list[Product]:
        """Enumerate the dataset's products, each with its components.

        ``prefix``, ``pattern`` and ``limit`` bound a product-set enumeration
        (the run-shape ``params.products`` knob): only products whose key starts
        with ``prefix`` and matches the ``pattern`` regex are yielded, and at
        most ``limit`` of them. ``pattern`` lets one run select a set that is not
        a single path-prefix (e.g. one date across adjacent tiles). Single-product
        readers ignore all three.
        """

    def rgb_composites(self) -> list[RgbComposite]:
        """Viewer composites this dataset can assemble from its components.

        Each names three already-enumerated component names to stack R/G/B in a
        VRT for manual TiTiler inspection. Base: none — override in dataset
        subclasses that have meaningful band combinations (Sentinel-2, Sentinel-1).
        """
        return []

    def viewer_bands(self) -> list[SingleBandComposite]:
        """Single-band viewer VRTs this dataset can assemble from its components.

        Each names one component to mosaic as a Gray VRT for datasets whose
        products are scalar/single-variable (e.g. SWOT WSE, DEMs). Base: none —
        override in single-variable dataset subclasses.
        """
        return []
