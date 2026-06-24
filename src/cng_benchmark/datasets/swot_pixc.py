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


def _pixc_group_uri(
    granule_uri: str,
    group: str,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Compose a point-cloud source URI addressing ``group`` inside ``granule_uri``.

    The COPC adapter's point loader dispatches on the ``PIXC:`` scheme
    (:data:`~cng_benchmark.formats.copc.PIXC_SCHEME`), reading the netCDF group's
    geometry and the carried point variables in place. ``include`` (an allow-list)
    and ``exclude`` (a deny-list) travel as a ``?include=…&exclude=…`` query that
    the loader applies; with neither, every point-dimensioned variable is carried.
    The original granule URI (``s3://`` or local) is kept intact so the loader opens
    it with xarray/fsspec; ``storage.to_gdal_path`` passes the string through.
    """
    uri = f"{PIXC_SCHEME}{granule_uri}::{group}"
    parts: list[str] = []
    if include is not None:
        parts.append("include=" + ",".join(include))
    if exclude:
        parts.append("exclude=" + ",".join(exclude))
    return uri + ("?" + "&".join(parts) if parts else "")


class SwotPixcOptions(DatasetOptions):
    """Group and carried-variable picks for a SWOT PIXC granule.

    ``groups`` selects which netCDF groups to read as point clouds, one component
    each (default :data:`DEFAULT_GROUPS`, in order, so the runner's representative
    sample is deterministic). ``point_variables`` and ``exclude_variables`` choose
    which per-point variables the COPC carries as LAS extra dimensions:
    ``point_variables`` is an allow-list (``None`` = carry **every** point variable,
    the content-complete default) and ``exclude_variables`` a deny-list. The
    geometry (lon/lat/height → x/y/z) is always carried.
    """

    groups: list[str] | None = None
    point_variables: list[str] | None = None
    exclude_variables: list[str] = []


@DATASETS.register("swot-pixc")
class SwotPixcDataset(GranuleDataset):
    """Enumerate the selected pixel-cloud group(s) per SWOT PIXC netCDF granule."""

    Options = SwotPixcOptions
    granule_suffix = ".nc"

    def _select_components(self, granule_uri: str) -> list[SourceObject]:
        opts: SwotPixcOptions = self.options
        groups = opts.groups if opts.groups else DEFAULT_GROUPS
        return [
            SourceObject(
                name=group,
                uri=_pixc_group_uri(
                    granule_uri,
                    group,
                    include=opts.point_variables,
                    exclude=opts.exclude_variables,
                ),
            )
            for group in groups
        ]
