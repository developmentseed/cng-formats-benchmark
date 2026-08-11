"""Grid-equality grouping — shared by every batched-write adapter (#102).

Bundling several dataset components into one store (GeoZarr) or one file (COG)
is only correct when the components involved actually cover the same ground:
same pixel grid, same georeferencing. This module owns the one grouping rule
both adapters need, so "detect by shape/transform equality, don't assume" is
implemented once rather than twice, slightly differently, in each writer.

Each adapter reads a component's grid its own way (GeoZarr via rioxarray, COG
via :func:`cng_benchmark.vrt.read_grid`/rasterio) and normalises it to a
``GridKey`` before calling :func:`group_by_grid` — this module has no IO and no
geo dependency, so it is trivially unit-testable.
"""

from __future__ import annotations

from typing import NamedTuple


class GridKey(NamedTuple):
    """One component's grid, normalised for equality comparison.

    ``shape`` is ``(height, width)`` and ``geotransform`` the 6-coefficient
    affine (either coefficient order is fine as long as a single adapter is
    consistent about which one it uses — this module only compares for
    equality, never interprets the numbers). Floats compare exactly: every
    caller derives them from the same source read, so two components on
    genuinely the same grid produce bit-identical values, and a mismatch
    (different resolution, origin, or CRS) is exactly what should split them
    into separate groups rather than being silently coerced together.
    """

    name: str
    shape: tuple[int, int]
    geotransform: tuple[float, ...]
    crs: str


def group_by_grid(items: list[GridKey]) -> list[list[str]]:
    """Group component names sharing an identical grid.

    Returns groups in first-appearance order (the order the caller's
    ``items`` list itself is in), each a list of component names — the caller
    assigns group ids (``grid0``, ``grid1``, …) from this order. A component
    that shares no grid with any other is still returned as its own
    single-item group.
    """
    order: list[tuple[tuple[int, int], tuple[float, ...], str]] = []
    grouped: dict[tuple[tuple[int, int], tuple[float, ...], str], list[str]] = {}
    for item in items:
        key = (item.shape, item.geotransform, item.crs)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item.name)
    return [grouped[key] for key in order]
