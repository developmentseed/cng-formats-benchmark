"""SWOT LakeSP Prior reader — the zipped-shapefile vector granule.

The cross-mission extensibility proof: a *vector* arm wired purely through
config. The CNES testbed stages SWOT **L2 HR LakeSP Prior** products as one
zipped ESRI Shapefile per pass, flat under ``source``
(``SWOT_L2_HR_LakeSP_Prior_*_EU_*.zip``, 7 KB–37 MB each) — lake vector features
(polygons/points) with attributes, on a nominal/calibration orbit over Europe.
This is the same zip-per-scene delivery shape as the S1/S2 readers
(:mod:`cng_benchmark.datasets.zip_delivery`), so only member selection is new:
each ``.shp`` member becomes one component (one pass = one layer), read **on the
fly** through GDAL's ``/vsizip//vsis3`` chain — the OGR shapefile driver finds
the ``.shx``/``.dbf``/``.prj`` sidecars inside the same archive — so the
conversion's write metric pays the real archive read cost, exactly as the raster
readers do.

Each pass is converted to a GeoParquet file by the GeoParquet adapter. Passes are
tiny, so per-pass they sit below Tier 2; the GeoParquet grouping lever is what
decides whether a useful grouping clears the floor — the sub-Tier-2 fan-out
story, like the S2 snow masks.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from cng_benchmark.datasets.base import SourceObject
from cng_benchmark.datasets.zip_delivery import ZipDeliveryDataset, _member_vsi_uri
from cng_benchmark.registry import DATASETS

#: A shapefile member: a ``.shp`` at any depth in the archive. The ``.shp.xml``
#: metadata sidecar ends in ``.xml`` and is therefore not matched.
_SHAPEFILE_RE = re.compile(r"\.shp$", re.IGNORECASE)


@DATASETS.register("swot-lakesp-prior")
class SwotLakeSpPriorDataset(ZipDeliveryDataset):
    """Enumerate the shapefile layer(s) per SWOT LakeSP Prior pass — one zip each."""

    def _select_members(self, members: list[str], zip_uri: str) -> list[SourceObject]:
        selected: list[SourceObject] = []
        for member in members:
            if not _SHAPEFILE_RE.search(member):
                continue
            selected.append(
                SourceObject(
                    name=PurePosixPath(member).stem,
                    uri=_member_vsi_uri(zip_uri, member),
                )
            )
        # Sort by layer name so the representative sample (the runner takes the
        # first component) is deterministic when a pass holds several shapefiles.
        selected.sort(key=lambda c: c.name)
        return selected
