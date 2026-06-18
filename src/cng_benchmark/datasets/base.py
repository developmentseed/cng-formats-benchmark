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
    """

    name: str
    uri: str


@dataclass(frozen=True)
class Product:
    """A delivery unit — one scene/granule — and its baseline components."""

    id: str
    components: list[SourceObject]


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
        self, *, prefix: str | None = None, limit: int | None = None
    ) -> list[Product]:
        """Enumerate the dataset's products, each with its components.

        ``prefix`` and ``limit`` bound a product-set enumeration (the run-shape
        ``params.products`` knob): only products under ``prefix`` are yielded,
        and at most ``limit`` of them. Single-product readers ignore both.
        """
