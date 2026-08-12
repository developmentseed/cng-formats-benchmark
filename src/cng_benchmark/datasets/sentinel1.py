"""Sentinel-1 RTC reader — the S1Tiling (OTB) gamma0 product.

The CNES S1 RTC tiles are produced by the S1Tiling chain, which is built on the
Orfeo ToolBox (OTB); the delivery is a zip-per-scene — the same ground-segment
shape as the MAJA L2A delivery — but far simpler: each zip holds just the two
polarisation band rasters (VV and VH gamma0, ~10 m MGRS tiles) plus quicklooks.
Which polarisations a run profiles is the layout-specific pick, carried in
``options`` and validated by :class:`Sentinel1OtbRtcOptions`. The class and its
reader id ``sentinel1-otb-rtc`` are named for the processor, mirroring the MAJA
reader (``sentinel2-maja``) in ``datasets/sentinel2.py``.

The member-name pattern lives here, not in shared config: an RTC band is
``…_<VV|VH>_GAM_…\\.tif`` (GAM = RTC gamma0) at the scene root; the ``_QKL_ALL.jpg``
quicklooks and ``.aux.xml`` sidecars are ignored. Members are read **on the fly**
through GDAL's ``/vsizip//vsis3`` chain, so the conversion's write metric pays the
real archive read cost, exactly as the MAJA reader does.

No border/validity mask member: the reference EOPF ``s1-tiling`` ingestion
(``ingest_s1tiling_acquisition``, #108) bundles VV, VH *and* a border mask
into one store, but a captured real S1Tiling delivery listing
(``tests/test_datasets.py``'s ``S1_MEMBERS``) has only the two ``_GAM_`` band
rasters plus a quicklook jpg and an ``.aux.xml`` sidecar — no third raster
this reader could plausibly be a border mask. This CNES delivery genuinely
has no equivalent to bundle as a third component, rather than the reader
silently missing one.
"""

from __future__ import annotations

import re

from cng_benchmark.datasets.base import DatasetOptions, RgbComposite, SourceObject
from cng_benchmark.datasets.zip_delivery import ZipDeliveryDataset, _member_vsi_uri
from cng_benchmark.registry import DATASETS

#: An RTC gamma0 band member: ``<scene>_<VV|VH>_GAM_<...>.tif`` at the scene root.
_BAND_RE = re.compile(r"(?:^|/)[^/]*_(VV|VH)_GAM_[^/]*\.tif$", re.IGNORECASE)
#: The CF quantity a terrain-corrected gamma0 band holds. The delivered GeoTIFFs
#: name it nowhere, so the reader carries it to the writers.
_RTC_STANDARD_NAME = "surface_backwards_scattering_coefficient_of_radar_wave"


class Sentinel1OtbRtcOptions(DatasetOptions):
    """Polarisation picks for an S1Tiling (OTB) RTC product.

    ``polarizations`` selects which gamma0 bands to profile; both (VV, VH) by
    default. An empty list profiles none.
    """

    polarizations: list[str] = ["VV", "VH"]


@DATASETS.register("sentinel1-otb-rtc")
class Sentinel1OtbRtcDataset(ZipDeliveryDataset):
    """Enumerate the selected VV/VH gamma0 bands per S1Tiling (OTB) RTC scene."""

    Options = Sentinel1OtbRtcOptions

    def _select_members(self, members: list[str], zip_uri: str) -> list[SourceObject]:
        opts: Sentinel1OtbRtcOptions = self.options
        want = [p.upper() for p in opts.polarizations]
        want_set = set(want)

        selected: list[SourceObject] = []
        for member in members:
            band = _BAND_RE.search(member)
            if not band:
                continue
            pol = band.group(1).upper()
            if pol in want_set:
                selected.append(
                    SourceObject(
                        name=pol,
                        uri=_member_vsi_uri(zip_uri, member),
                        standard_name=_RTC_STANDARD_NAME,
                    )
                )
        # Order by the configured polarisation list, so the representative-band
        # sample (the runner takes the first component) is deterministic.
        selected.sort(key=lambda c: want.index(c.name) if c.name in want else len(want))
        return selected

    def rgb_composites(self) -> list[RgbComposite]:
        """The dual-pol quicklook stack, when both polarisations are profiled.

        A common Sentinel-1 RGB is ``R=VV, G=VH, B=VV/VH`` — but the ratio band
        needs a GDAL Python pixel function, which the slim runner/TiTiler image
        does not enable, so we use the standard ``VV/VH/VV`` quicklook instead.
        gamma0 is float, so a ``(0, 0.4)`` rescale hint travels with it.
        """
        want = {p.upper() for p in self.options.polarizations}
        if {"VV", "VH"} <= want:
            return [
                RgbComposite(
                    name="dualpol", bands=("VV", "VH", "VV"), rescale=(0.0, 0.4)
                )
            ]
        return []
