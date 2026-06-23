"""SWOT PIXC reader — the netCDF point-cloud granule.

The heaviest SWOT arm: the CNES testbed stages SWOT **L2 HR PIXC** products as
one netCDF *pixel cloud* per granule, flat under ``source``
(``SWOT_L2_HR_PIXC_*.nc``, ~0.4–0.95 GB each) — height/geolocation per water
pixel, held in the netCDF ``pixel_cloud`` group. This is the same granule layout
as the SWOT Raster100m reader (:mod:`cng_benchmark.datasets.granule`), but a
component is a *point-cloud group* rather than a CF raster variable: each selected
group becomes one component, converted to a COPC file by the COPC adapter, whose
point loader reads the group's lon/lat/height with xarray in place (so the
conversion's write metric pays the real granule read cost).

Which groups a run profiles is the layout-specific pick, carried in ``options``
and validated by :class:`SwotPixcOptions`; the default is the primary
``pixel_cloud`` group. The COPC adapter and this point-cloud path are then reused
by the CO3D CARS arm (tiled LAZ → COPC).
"""

from __future__ import annotations

from cng_benchmark.datasets.base import DatasetOptions, SourceObject
from cng_benchmark.datasets.granule import GranuleDataset
from cng_benchmark.formats.copc import PIXC_SCHEME
from cng_benchmark.registry import DATASETS

#: The primary SWOT PIXC group — the pixel cloud — profiled when ``options.groups``
#: is unset. Overridable to select any netCDF group(s) read as a point cloud.
DEFAULT_GROUPS = ["pixel_cloud"]


def _pixc_group_uri(granule_uri: str, group: str) -> str:
    """Compose a point-cloud source URI addressing ``group`` inside ``granule_uri``.

    The COPC adapter's point loader dispatches on the ``PIXC:`` scheme
    (:data:`~cng_benchmark.formats.copc.PIXC_SCHEME`), reading the netCDF group's
    lon/lat/height in place. The original granule URI (``s3://`` or local) is kept
    intact so the loader opens it with xarray/fsspec; ``storage.to_gdal_path``
    passes the composed string through unchanged.
    """
    return f"{PIXC_SCHEME}{granule_uri}::{group}"


class SwotPixcOptions(DatasetOptions):
    """Group picks for a SWOT PIXC granule.

    ``groups`` selects which netCDF groups to read as point clouds, one component
    each. ``None`` (the default) profiles the primary :data:`DEFAULT_GROUPS`; an
    explicit list selects exactly those, in order, so the representative read
    sample (the runner takes the first component) is deterministic.
    """

    groups: list[str] | None = None


@DATASETS.register("swot-pixc")
class SwotPixcDataset(GranuleDataset):
    """Enumerate the selected pixel-cloud group(s) per SWOT PIXC netCDF granule."""

    Options = SwotPixcOptions
    granule_suffix = ".nc"

    def _select_components(self, granule_uri: str) -> list[SourceObject]:
        opts: SwotPixcOptions = self.options
        groups = opts.groups if opts.groups else DEFAULT_GROUPS
        return [
            SourceObject(name=group, uri=_pixc_group_uri(granule_uri, group))
            for group in groups
        ]
