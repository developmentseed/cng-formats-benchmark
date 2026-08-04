"""The overview pyramid's metadata: the Zarr conventions a GeoZarr store declares.

A GeoZarr raster carries an overview pyramid the way a COG carries overviews:
each coarser level is a child group ``0``..``N`` (``0`` = native resolution) and
the parent group's attributes describe the pyramid. This module builds those
attributes, plus the per-level georeferencing each level needs to stand on its
own.

**Which spec.** GeoZarr 0.4 is deprecated, and GeoZarr v1 — which will be built
on the Zarr conventions — has not landed, so there is no GeoZarr version to
claim conformance to. What *is* published is the set of conventions v1 will
assemble, so the store is written against those directly and declares them in
``zarr_conventions``:

* ``multiscales`` — the pyramid ``layout``: which group holds each level, which
  level it was derived from, the relative ``scale`` / ``translation`` that
  derivation induced, and the ``resampling_method``.
* ``spatial`` — ``spatial:transform`` (rasterio/affine coefficient order,
  **not** GDAL's), ``spatial:shape``, ``spatial:bbox``, ``spatial:dimensions``,
  ``spatial:registration``, per level and on the group.
* ``geo-proj`` — the **native** CRS as ``proj:code`` (or ``proj:wkt2``); nothing
  is reprojected to web mercator.

The documents themselves come from :mod:`zarr_cm`, which owns each convention's
uuid, its pinned schema/spec URLs and its validation — so a convention revision
is that package's business to track, not ours to copy. What is ours is the
geometry the conventions describe: how deep the pyramid goes, where each level
sits, and the two coefficient orders that have to be kept apart.

The conventions are the *only* place the georeferencing lives. There is no CF
``spatial_ref`` grid-mapping variable duplicating the CRS and transform, no
``grid_mapping`` attribute pointing at one, and no ``_ARRAY_DIMENSIONS`` — Zarr
v3 names an array's axes in its own ``dimension_names`` field, which is what
``spatial:dimensions`` refers to. A reader that wants the grid reads the
convention.

Needs only ``zarr_cm`` + numpy — no zarr, xarray, pyproj or rasterio — so the
metadata is unit-testable in CI without the geo stack.
"""

from __future__ import annotations

import re
from typing import Any

#: A level is only worth writing while its shorter side stays at or above this,
#: i.e. coarsen down to roughly one tile — the rule GDAL follows for COG
#: overviews, so the two arms end up with comparable pyramids.
MIN_LEVEL_DIMENSION = 256

#: How the levels are built; recorded so a reader knows what it is looking at.
RESAMPLING_METHOD = "average"

#: Grid cell registration: coordinates refer to cell edges (GDAL's PixelIsArea).
REGISTRATION = "pixel"

#: The EPSG code in a WKT1 ``AUTHORITY`` or WKT2 ``ID`` node. The outermost CRS
#: authority is written last, so the *last* match is the CRS's own code.
_EPSG_RE = re.compile(r'(?:ID|AUTHORITY)\s*\[\s*"EPSG"\s*,\s*"?(\d+)"?\s*\]')


def auto_depth(shape: tuple[int, int], min_dimension: int = MIN_LEVEL_DIMENSION) -> int:
    """Return the COG-style pyramid depth for an array of ``shape``.

    Halves while the shorter side stays at or above ``min_dimension``, so the
    coarsest level is roughly one tile. Returns the number of *overview* levels:
    0 means the array is already small enough to serve whole.
    """
    h, w = int(shape[0]), int(shape[1])
    depth = 0
    while min(h, w) // 2 >= min_dimension:
        h, w = h // 2, w // 2
        depth += 1
    return depth


def parse_geotransform(geotransform: str) -> tuple[float, ...] | None:
    """Parse a GDAL geotransform string (``c a b f d e``); ``None`` if malformed."""
    try:
        vals = [float(v) for v in str(geotransform).split()]
    except ValueError:
        return None
    return tuple(vals) if len(vals) == 6 else None


def decimate(gt: tuple[float, ...], decimation: int) -> tuple[float, ...]:
    """Return the geotransform of a level coarsened by ``decimation``.

    The origin is unchanged and every pixel-size term scales — a coarser level
    covers the same ground with bigger cells. Writing the native geotransform
    onto every level would mislocate each overview by its decimation factor.
    """
    c, a, b, f, d, e = gt
    return (c, a * decimation, b * decimation, f, d * decimation, e * decimation)


def spatial_transform(gt: tuple[float, ...]) -> list[float]:
    """Convert a GDAL geotransform to ``spatial:transform`` coefficients.

    The convention uses the rasterio/affine order ``[a, b, c, d, e, f]`` (pixel
    sizes and rotations first, origin third and sixth); GDAL's is
    ``[c, a, b, f, d, e]``. Emitting one in the other's order silently shifts
    every coordinate, so the two orders are converted explicitly here.
    """
    c, a, b, f, d, e = gt
    return [a, b, c, d, e, f]


def bounds(gt: tuple[float, ...], shape: tuple[int, int]) -> list[float]:
    """Return the ``[xmin, ymin, xmax, ymax]`` extent of an array of ``shape``."""
    c, a, b, f, d, e = gt
    h, w = int(shape[0]), int(shape[1])
    xs = (c, c + w * a + h * b)
    ys = (f, f + w * d + h * e)
    return [min(xs), min(ys), max(xs), max(ys)]


def is_projected(crs_wkt: str) -> bool:
    """Whether the WKT describes a projected CRS (vs a geographic one)."""
    return "PROJCRS" in crs_wkt or "PROJCS" in crs_wkt


def epsg_code(crs_wkt: str) -> int | None:
    """Return the CRS's EPSG code from its WKT, or ``None`` if it declares none."""
    matches = _EPSG_RE.findall(crs_wkt or "")
    return int(matches[-1]) if matches else None


def proj_attrs(crs_wkt: str) -> dict[str, Any]:
    """Return the ``geo-proj`` data for a CRS, in its native definition.

    ``proj:code`` when the WKT names an EPSG authority — the compact form a
    reader can resolve — otherwise the WKT2 string itself. Empty for a store
    with no CRS at all.
    """
    import zarr_cm

    if not crs_wkt:
        return {}
    code = epsg_code(crs_wkt)
    if code:
        return dict(zarr_cm.proj.create(code=f"EPSG:{code}"))
    return dict(zarr_cm.proj.create(wkt2=crs_wkt))


def coordinate_attrs(crs_wkt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the CF attributes for the ``(y, x)`` coordinate variables.

    Projected axes are metric ``projection_{x,y}_coordinate``; a geographic CRS
    gets ``longitude``/``latitude`` in degrees. What each variable *is*, not
    where the grid sits — that is ``spatial:transform``'s job.
    """
    y: dict[str, Any] = {}
    x: dict[str, Any] = {}
    if crs_wkt and not is_projected(crs_wkt):
        y |= {
            "units": "degrees_north",
            "long_name": "latitude",
            "standard_name": "latitude",
        }
        x |= {
            "units": "degrees_east",
            "long_name": "longitude",
            "standard_name": "longitude",
        }
    else:
        y |= {
            "units": "m",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
        }
        x |= {
            "units": "m",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
        }
    return y, x


def level_coords(gt: tuple[float, ...], shape: tuple[int, int]):
    """Return the ``(y, x)`` cell-centre coordinate arrays for one level.

    ``None`` when the geotransform is rotated or sheared: 1D coordinate
    variables cannot describe such a grid, and ``spatial:transform`` stays the
    authoritative georeferencing either way.
    """
    import numpy as np

    c, a, b, f, d, e = gt
    if b or d:
        return None
    h, w = int(shape[0]), int(shape[1])
    x = c + (np.arange(w, dtype="float64") + 0.5) * a
    y = f + (np.arange(h, dtype="float64") + 0.5) * e
    return y, x


def spatial_attrs(
    gt: tuple[float, ...],
    shape: tuple[int, int],
    *,
    dimensions: tuple[str, str] = ("y", "x"),
) -> dict[str, Any]:
    """Return the ``spatial`` data describing one grid."""
    import zarr_cm

    return dict(
        zarr_cm.spatial.create(
            dimensions=list(dimensions),
            transform=spatial_transform(gt),
            shape=[int(shape[0]), int(shape[1])],
            bbox=bounds(gt, shape),
            registration=REGISTRATION,
        )
    )


def dimensions_attrs(dimensions: tuple[str, str] = ("y", "x")) -> dict[str, Any]:
    """Return the minimum a data array declares: which of its axes are spatial.

    Ordered Y then X — what lets a reader resolve ``spatial:transform`` against
    the array's axes. All a store with no georeferencing at all can say.
    """
    import zarr_cm

    return zarr_cm.create_many(
        {"spatial": zarr_cm.spatial.create(dimensions=list(dimensions))}
    )


def grid_attrs(
    crs_wkt: str,
    gt: tuple[float, ...],
    shape: tuple[int, int],
    *,
    dimensions: tuple[str, str] = ("y", "x"),
) -> dict[str, Any]:
    """Return one grid's full ``spatial`` + ``geo-proj`` description.

    Written on a level group *and* on the array inside it. The convention lets a
    group's description stand in for its arrays, but a reader holding just the
    array — which is what a tile server addresses — only has the array's own
    attributes to go on, so the array repeats them rather than being
    ungeoreferenceable on its own. Each level states its own grid: they cover
    the same ground with cells twice as big.
    """
    import zarr_cm

    conventions: dict[str, Any] = {
        "spatial": spatial_attrs(gt, shape, dimensions=dimensions)
    }
    proj = proj_attrs(crs_wkt)
    if proj:
        conventions["geo-proj"] = proj
    return zarr_cm.create_many(conventions)


def _layout_entry(level: int) -> dict[str, Any]:
    """Return the resampling half of one ``multiscales.layout`` entry.

    ``transform`` holds the **relative** step from the level this one was
    derived from: a 2x2 block average scales coordinates by 2 and keeps the
    origin, so translation is zero. Level 0 derives from nothing and is the
    source of the chain. The absolute position sits *outside* ``transform``, as
    the convention requires — :func:`multiscales_attrs` adds it.
    """
    if level == 0:
        return {
            "asset": "0",
            "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]},
        }
    return {
        "asset": str(level),
        "derived_from": str(level - 1),
        "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
        "resampling_method": RESAMPLING_METHOD,
    }


def multiscales_attrs(
    gt: tuple[float, ...] | None, shapes: list[tuple[int, int]]
) -> dict[str, Any]:
    """Return the ``multiscales`` data for a pyramid of ``shapes``.

    Without a geotransform the levels still form a layout — each derived from
    the one above it — they just carry no absolute position.
    """
    import zarr_cm

    layout = []
    for level, shape in enumerate(shapes):
        entry = _layout_entry(level)
        if gt is not None:
            entry["spatial:transform"] = spatial_transform(decimate(gt, 2**level))
            entry["spatial:shape"] = [int(shape[0]), int(shape[1])]
        layout.append(entry)
    return dict(
        zarr_cm.multiscales.create(
            layout=tuple(layout), resampling_method=RESAMPLING_METHOD
        )
    )


def group_attrs(
    crs_wkt: str,
    gt: tuple[float, ...] | None,
    shapes: list[tuple[int, int]],
    *,
    dimensions: tuple[str, str] = ("y", "x"),
) -> dict[str, Any]:
    """Build the pyramid group's attributes for the levels of ``shapes``.

    The ``multiscales`` layout (one entry per level, each derived from the one
    above it), the group-level ``spatial`` description of the native grid, and
    the native CRS via ``geo-proj`` — composed into one attributes dict with a
    combined ``zarr_conventions``, and validated, by :mod:`zarr_cm`.
    """
    import zarr_cm

    conventions: dict[str, Any] = {
        "multiscales": multiscales_attrs(gt, shapes),
    }
    if gt is not None:
        conventions["spatial"] = spatial_attrs(gt, shapes[0], dimensions=dimensions)
    proj = proj_attrs(crs_wkt)
    if proj:
        conventions["geo-proj"] = proj
    return zarr_cm.create_many(conventions)
