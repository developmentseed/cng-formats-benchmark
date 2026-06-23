"""Runner-image capability contract — the per-arm GDAL/OGR drivers and libraries.

Each benchmark arm needs the runner image to be able to *read its source* and
*write/read its target*, and several of those capabilities are bundled, unpinned,
inside a wheel: the SWOT Raster100m arm only reads its netCDF granules because the
GDAL **netCDF** driver happens to ship in rasterio's manylinux wheel, and the
GeoParquet arm only reads its zipped shapefiles because pyogrio's *own* bundled
GDAL (separate from rasterio's) ships the **ESRI Shapefile** OGR driver. Nothing
declared that, so a future wheel or base-image change could drop a driver and only
a *run* would notice.

This module is that missing declaration: :data:`REQUIRED` lists, per arm, the
GDAL raster drivers, OGR vector drivers, and Python libraries the image must
provide. ``cng-benchmark check-drivers`` verifies them, and the runner image's
build runs it (plus a CI step), so a missing capability fails the **build**, not a
production run. Adding an arm means adding its capabilities here.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

#: A capability kind: a GDAL raster driver (checked via rasterio's bundled GDAL),
#: an OGR vector driver (checked via pyogrio's *own* bundled GDAL — the source-read
#: stack for the vector arm), or an importable Python library.
GDAL_RASTER = "gdal-raster"
OGR_VECTOR = "ogr-vector"
PYTHON = "python"


@dataclass(frozen=True)
class Capability:
    """One thing the runner image must provide, and which arm needs it."""

    arm: str
    kind: str
    name: str
    why: str


#: The per-arm capability contract the runner image must satisfy. Keep this in
#: step with ``docker/Dockerfile.runner``'s extras when adding an arm.
REQUIRED: tuple[Capability, ...] = (
    Capability("cog", GDAL_RASTER, "GTiff", "read/write the GeoTIFF baseline + COG"),
    Capability("cog", PYTHON, "rio_cogeo", "COG translation (rio-cogeo)"),
    Capability(
        "swot-raster100m",
        GDAL_RASTER,
        "netCDF",
        'read NETCDF:"<granule>":<var> subdataset variables (SWOT-A)',
    ),
    Capability("geozarr", PYTHON, "rioxarray", "read the source raster into xarray"),
    Capability("geozarr", PYTHON, "zarr", "write/read the Zarr v3 sharded store"),
    Capability(
        "geoparquet",
        OGR_VECTOR,
        "ESRI Shapefile",
        "read the zipped shapefile source (SWOT-B / LakeSP)",
    ),
    Capability("geoparquet", PYTHON, "geopandas", "write GeoParquet + bbox read"),
    Capability("geoparquet", PYTHON, "pyarrow", "row-group layout + parquet IO"),
    Capability("copc", PYTHON, "copclib", "write the COPC octree (SWOT-C / CARS)"),
    Capability("copc", PYTHON, "laspy", "octree-node spatial-query read"),
    Capability(
        "swot-pixc", PYTHON, "h5netcdf", "read the PIXC pixel_cloud netCDF group"
    ),
)


def _gdal_raster_drivers() -> set[str]:
    """Return the GDAL raster/vector drivers registered in rasterio's GDAL."""
    import rasterio

    with rasterio.Env() as env:
        return set(env.drivers())


def _ogr_drivers() -> set[str]:
    """Return the OGR vector drivers in pyogrio's (own) bundled GDAL."""
    import pyogrio

    return set(pyogrio.list_drivers())


def check_capability(cap: Capability) -> tuple[bool, str]:
    """Return ``(present, detail)`` for one capability.

    Each kind is probed in the stack that actually uses it at run time — a GDAL
    raster driver through rasterio, an OGR driver through pyogrio (whose bundled
    GDAL is what the vector source read uses), a library through ``find_spec`` — so
    the check reflects the real read/write path, not a coincidental one. A missing
    extra (the probe stack itself absent) reports as not-present rather than
    raising, so the report lists every gap at once.
    """
    try:
        if cap.kind == GDAL_RASTER:
            return cap.name in _gdal_raster_drivers(), "rasterio GDAL"
        if cap.kind == OGR_VECTOR:
            return cap.name in _ogr_drivers(), "pyogrio OGR"
        return importlib.util.find_spec(cap.name) is not None, "python import"
    except Exception as exc:  # noqa: BLE001 - report the gap, never crash the report
        return False, f"unavailable ({type(exc).__name__})"


def check_all(
    capabilities: tuple[Capability, ...] | None = None,
) -> list[tuple[Capability, bool, str]]:
    """Check every capability and return ``(capability, present, detail)`` rows.

    Defaults to the module-level :data:`REQUIRED`, resolved at call time so it can
    be substituted in tests.
    """
    caps = REQUIRED if capabilities is None else capabilities
    return [(cap, *check_capability(cap)) for cap in caps]
