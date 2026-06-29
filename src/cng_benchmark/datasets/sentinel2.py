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

from cng_benchmark.datasets.base import DatasetOptions, RgbComposite, SourceObject
from cng_benchmark.datasets.zip_delivery import ZipDeliveryDataset, _member_vsi_uri
from cng_benchmark.registry import DATASETS

#: A reflectance member: ``<product>_<FRE|SRE>_<band>.tif`` at the scene root.
_REFLECTANCE_RE = re.compile(r"(?:^|/)[^/]*_(FRE|SRE)_(B\w+)\.tif$", re.IGNORECASE)
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
