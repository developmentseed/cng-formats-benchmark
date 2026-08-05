"""SWOT Raster100m reader — the netCDF-raster granule (GranuleDataset).

The lowest-effort SWOT arm: the CNES testbed stages SWOT **L2 HR Raster 100m**
products as one netCDF file per granule, flat under ``source``
(``SWOT_L2_HR_Raster_100m_UTM*_*.nc``, ~50–71 MB each) — a CF/geo 2D raster swath
on a 100 m UTM grid. There is no new display surface and no new format: each
selected CF variable becomes one component, converted to a sharded 2D GeoZarr v3
store by the existing GeoZarr adapter (the same per-component path proven on
S1/S2), and displayed through the bench tiler's GeoZarr router. Only the
reader is new.

A variable is read in place through GDAL's CF subdataset syntax
(:func:`~cng_benchmark.datasets.granule._subdataset_vsi_uri`), so the
conversion's write metric pays the real granule read cost. Which variables a run
profiles is the layout-specific pick, carried in ``options`` and validated by
:class:`SwotRaster100mOptions`; the default is the primary water-surface-elevation
variable (``wse``). ``variables: all`` (or ``"*"``) instead profiles **every** CF
variable the granule carries (enumerated via
:func:`~cng_benchmark.datasets.granule._list_cf_subdatasets`) — the
content-complete pick a like-for-like comparison against the as-delivered
multi-variable ``.nc`` needs (#86, the same class of correction as the
content-complete PIXC point cloud, #38): profiling ``wse`` alone against a
granule that carries fifteen to twenty CF variables measures dropped content,
not compression.
"""

from __future__ import annotations

from typing import Literal

from cng_benchmark import storage
from cng_benchmark.datasets.base import (
    DatasetOptions,
    SingleBandComposite,
    SourceObject,
)
from cng_benchmark.datasets.granule import (
    GranuleDataset,
    _list_cf_subdatasets,
    _subdataset_vsi_uri,
)
from cng_benchmark.registry import DATASETS

#: The primary SWOT HR Raster variable — water surface elevation — profiled when
#: ``options.variables`` is unset. Overridable to select any CF variable(s).
DEFAULT_VARIABLES = ["wse"]

#: ``variables`` sentinel values meaning "profile every CF variable in the
#: granule" rather than a fixed pick.
_ALL_VARIABLES = ("all", "*")


class SwotRaster100mOptions(DatasetOptions):
    """Variable picks for a SWOT Raster100m granule.

    ``variables`` selects which CF variables to profile, one component each.
    ``None`` (the default) profiles the primary :data:`DEFAULT_VARIABLES`; an
    explicit list selects exactly those, in order, so the representative read/
    display sample (the runner takes the first component) is deterministic;
    ``"all"`` (or ``"*"``) profiles every CF variable the granule carries,
    enumerated at run time — the content-complete pick for an honest
    compression comparison against the as-delivered ``.nc``.
    """

    variables: list[str] | Literal["all", "*"] | None = None


@DATASETS.register("swot-raster100m")
class SwotRaster100mDataset(GranuleDataset):
    """Enumerate the selected CF variables per SWOT Raster100m netCDF granule."""

    Options = SwotRaster100mOptions
    granule_suffix = ".nc"

    def _select_components(self, granule_uri: str) -> list[SourceObject]:
        variables = self._resolve_variables(granule_uri)
        return [
            SourceObject(name=var, uri=_subdataset_vsi_uri(granule_uri, var))
            for var in variables
        ]

    def _resolve_variables(self, granule_uri: str) -> list[str]:
        opts: SwotRaster100mOptions = self.options
        if opts.variables in _ALL_VARIABLES:
            return _list_cf_subdatasets(granule_uri)
        return opts.variables if opts.variables else DEFAULT_VARIABLES

    def viewer_bands(self) -> list[SingleBandComposite]:
        """One viewer VRT per configured variable, mosaicked across granules.

        For ``variables: all``/``"*"`` the variable set is not known statically
        (it depends on the granule's own content), so one granule under
        ``source`` is peeked at to resolve it; an empty ``source`` yields no
        bands rather than failing the run.
        """
        opts: SwotRaster100mOptions = self.options
        if opts.variables in _ALL_VARIABLES:
            granules = storage.list_uris(
                self.source_uri, role="source", suffix=self.granule_suffix, limit=1
            )
            variables = _list_cf_subdatasets(granules[0]) if granules else []
        else:
            variables = opts.variables if opts.variables else DEFAULT_VARIABLES
        return [SingleBandComposite(name=var, band=var) for var in variables]
