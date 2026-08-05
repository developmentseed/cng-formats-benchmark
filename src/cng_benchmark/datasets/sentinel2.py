"""Sentinel-2 L2A (THEIA/MAJA) reader.

A MAJA L2A product is a zip-per-scene delivery whose components are the
reflectance bands (FRE flat-reflectance and/or SRE surface-reflectance, at 10/20/60 m)
plus the small per-pixel masks (CLM cloud, EDG edge, SAT saturation, MG2
geophysical) under ``MASKS/``. Which of those a run profiles is the
layout-specific pick, carried in ``options`` and validated by
:class:`Sentinel2MajaOptions`.

The MAJA member-name patterns live here, not in shared config: reflectance is
``…_<FRE|SRE>_<band>.tif`` at the scene root, masks are
``MASKS/…_<CLM|EDG|SAT|MG2>_R<n>.tif``. The whole point of profiling a product
is the object-size *distribution* it produces — a handful of large 10 m bands
and a fan-out of small masks — which only appears once the masks are included.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from cng_benchmark.datasets.base import DatasetOptions, RgbComposite, SourceObject
from cng_benchmark.datasets.granule import GranuleDataset
from cng_benchmark.datasets.zip_delivery import ZipDeliveryDataset, _member_vsi_uri
from cng_benchmark.registry import DATASETS

#: A reflectance member: ``<product>_<FRE|SRE>_<band>.tif`` at the scene root.
_REFLECTANCE_RE = re.compile(r"(?:^|/)[^/]*_(FRE|SRE)_(B\w+)\.tif$", re.IGNORECASE)
#: MAJA Int16 reflectance fill value (pixels outside the swath / masked as no-data).
#: The delivered GeoTIFFs declare neither this nor the quantification below in
#: their headers — it lives in MAJA side-metadata — so the reader carries both
#: to the writers, which is the only way they reach the produced object (#70).
_MAJA_NODATA = -10000.0
#: MAJA quantification: reflectance is stored as DN = reflectance x 10000.
_MAJA_SCALE_FACTOR = 1.0 / 10000.0
#: The CF quantity MAJA FRE (flat, slope-corrected) and SRE reflectance hold.
_MAJA_STANDARD_NAME = "surface_bidirectional_reflectance"
#: The MASKS members (CLM cloud, EDG edge, SAT saturation, MG2 geophysical) are
#: per-pixel condition flags, not a measured quantity.
_MAJA_MASK_STANDARD_NAME = "quality_flag"
#: A mask member: ``MASKS/<product>_<CLM|EDG|SAT|MG2>_R<n>.tif``.
_MASK_RE = re.compile(
    r"(?:^|/)MASKS/[^/]*_(CLM|EDG|SAT|MG2)_(R\d+)\.tif$", re.IGNORECASE
)

#: The 10 m reflectance bands — the representative band a default read/display
#: sample should land on (the masks are tiny and unrepresentative).
_TEN_M_BANDS = frozenset({"B2", "B3", "B4", "B8"})
_MASK_KINDS = frozenset({"CLM", "EDG", "SAT", "MG2"})

#: Viewer composites as ``(name, (red, green, blue) bands, (lo, hi) rescale)``.
#: SWIR's B11 is 20 m and not in the default ``bands`` pick, so the ``swir``
#: composite is realised only when B11 is added to the dataset options.
_COMPOSITES: tuple[tuple[str, tuple[str, str, str], tuple[float, float]], ...] = (
    ("natural", ("B4", "B3", "B2"), (0.0, 3000.0)),
    ("color-infrared", ("B8", "B4", "B3"), (0.0, 4000.0)),
    ("swir", ("B11", "B8", "B4"), (0.0, 4000.0)),
)


def _component_sort_key(name: str) -> tuple[bool, bool, str]:
    """Order components reflectance-first, 10 m bands first (see #13).

    The sample selection in the runner takes the first ``samples`` components in
    order, so a product's components must lead with a representative 10 m
    reflectance band rather than alphabetically (``CLM_R1`` < ``FRE_B2``).
    """
    kind, _, rest = name.partition("_")
    is_mask = kind.upper() in _MASK_KINDS
    is_ten_m = not is_mask and rest.upper() in _TEN_M_BANDS
    return (is_mask, not is_ten_m, name)


class Sentinel2MajaOptions(DatasetOptions):
    """Component picks for a MAJA L2A product.

    ``reflectance`` selects the reflectance kind(s) (FRE/SRE), ``bands`` the
    spectral bands to include, and ``masks`` which mask families to fan in. An
    empty list means "none of that family"; omit ``masks`` to profile
    reflectance only.
    """

    reflectance: list[str] = ["FRE"]
    bands: list[str] = ["B2", "B3", "B4", "B8"]
    masks: list[str] = []


@DATASETS.register("sentinel2-maja")
class Sentinel2MajaDataset(ZipDeliveryDataset):
    """Enumerate the selected FRE/SRE bands + CLM/EDG/SAT/MG2 masks per scene."""

    Options = Sentinel2MajaOptions

    def _select_members(self, members: list[str], zip_uri: str) -> list[SourceObject]:
        opts: Sentinel2MajaOptions = self.options
        want_refl = {r.upper() for r in opts.reflectance}
        want_bands = {b.upper() for b in opts.bands}
        want_masks = {m.upper() for m in opts.masks}

        selected: list[SourceObject] = []
        for member in members:
            refl = _REFLECTANCE_RE.search(member)
            if refl:
                kind, band = refl.group(1).upper(), refl.group(2).upper()
                if kind in want_refl and band in want_bands:
                    selected.append(
                        SourceObject(
                            name=f"{kind}_{band}",
                            uri=_member_vsi_uri(zip_uri, member),
                            nodata=_MAJA_NODATA,
                            scale_factor=_MAJA_SCALE_FACTOR,
                            standard_name=_MAJA_STANDARD_NAME,
                        )
                    )
                continue
            mask = _MASK_RE.search(member)
            if mask:
                kind, res = mask.group(1).upper(), mask.group(2).upper()
                if kind in want_masks:
                    selected.append(
                        SourceObject(
                            name=f"{kind}_{res}",
                            uri=_member_vsi_uri(zip_uri, member),
                            standard_name=_MAJA_MASK_STANDARD_NAME,
                        )
                    )
        selected.sort(key=lambda c: _component_sort_key(c.name))
        return selected

    def rgb_composites(self) -> list[RgbComposite]:
        """The natural/false-colour stacks realisable from the selected bands.

        The reflectance prefix is the first configured kind (``FRE``/``SRE``), and
        a composite is emitted only when all three of its bands are in
        ``options.bands`` — so ``swir`` appears solely when B11 was added to the
        pick. The component names match what :meth:`_select_members` lays out
        (``<kind>_<band>``), so the runner can address the produced COGs directly.
        """
        opts: Sentinel2MajaOptions = self.options
        if not opts.reflectance:
            return []
        kind = opts.reflectance[0].upper()
        have = {b.upper() for b in opts.bands}
        composites: list[RgbComposite] = []
        for name, bands, rescale in _COMPOSITES:
            if all(b in have for b in bands):
                composites.append(
                    RgbComposite(
                        name=name,
                        bands=tuple(f"{kind}_{b}" for b in bands),  # type: ignore[arg-type]
                        rescale=rescale,
                    )
                )
        return composites


# --- Sentinel-2 L2B snow (Let-it-Snow / LIS) --------------------------------
#
# The Datalake `sentinel2-l2b-snow-sprid` bucket stages Let-it-Snow (LIS)
# output as **loose per-date GeoTIFFs** under a tile prefix (`T31TCH/…`), not a
# zip-per-scene archive — the small-file anti-pattern the study's ch. 7.3
# tier-remediation priority names. Each date is one self-contained raster, so a
# component is the whole granule file, addressed directly (no `/vsizip` member
# decomposition, no CF subdataset split) — the granule base's prefix-listing
# with a single-object read per file, mirroring how :mod:`swot` addresses a
# netCDF granule but without the subdataset step.

#: An 8-digit ``YYYYMMDD`` acquisition date, wherever it falls in the filename.
_LIS_DATE_RE = re.compile(r"(\d{8})")


@DATASETS.register("sentinel2-l2b-snow-lis")
class Sentinel2LisDataset(GranuleDataset):
    """Enumerate the per-date snow-mask GeoTIFFs under a LIS tile prefix.

    One granule file = one product = one component: LIS delivers a single
    raster per date, so there is no band/mask pick to carry in ``options``
    (the default, empty :class:`~cng_benchmark.datasets.base.DatasetOptions`
    applies). The component is named for its acquisition date (extracted from
    the filename) rather than a fixed variable name, so the per-date fan-out
    that is the whole point of profiling this arm (#84) stays visible in the
    per-component results even when many dates are pooled into one dataset
    run. The product id (the filename stem) already carries the date too;
    naming the component the same way keeps every reported label
    self-describing without cross-referencing the product.
    """

    granule_suffix = ".tif"

    def _select_components(self, granule_uri: str) -> list[SourceObject]:
        stem = PurePosixPath(granule_uri.split("://", 1)[-1]).stem
        match = _LIS_DATE_RE.search(stem)
        name = match.group(1) if match else stem
        return [SourceObject(name=name, uri=granule_uri)]
