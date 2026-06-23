"""Cloud-Optimized Point Cloud (COPC) adapter — points to a single COPC LAZ.

The grouping lever for COPC is its **octree**: the octree depth and the per-node
point budget (the COPC ``span`` — a node is a ``span``-per-edge voxel grid, so it
holds at most ``span**3`` points) together set how points are grouped into the
range-addressable octree nodes a reader fetches for a spatial query. This adapter
writes one COPC file per component — a single ``POINT_CLOUD_FILE`` object, the
point-cloud analogue of the COG arm — and flows through the same runner paths as
COG, with the read metric an octree-node spatial query
(:func:`cng_benchmark.metrics.read.measure_copc_read`) rather than a raster
window. There is no display surface (a point cloud is not a TiTiler raster tile).

The COPC writer is :mod:`copclib` and the layout reader is its ``FileReader``;
both are pip wheels, so the octree-builder / enumerate / layout logic is
unit-testable in CI on a synthetic point cloud. The source read in
:func:`_load_points` is the only part that needs the granule stack — ``xarray`` +
``h5netcdf`` for a SWOT PIXC ``pixel_cloud`` group, or ``laspy`` for a LAS/LAZ
tile (the CO3D CARS reuse) — imported lazily.

A COPC granule is large (a SWOT PIXC pass is ~0.4–0.95 GB), so as a single object
it already clears the cold tiers; the lever here is about preserving
range-addressable partial access, not reaching a size floor.
"""

from __future__ import annotations

import math
import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import CopcLayout
from cng_benchmark.registry import FORMATS

#: Prefix marking a component URI as a netCDF group read as a point cloud:
#: ``PIXC:<granule_uri>::<group>`` (see :func:`_load_points`). Passed through
#: unchanged by ``storage.to_gdal_path`` (it is neither an ``s3://`` nor a
#: ``file://`` URI), like the SWOT raster reader's ``NETCDF:`` subdataset paths.
PIXC_SCHEME = "PIXC:"

#: Default per-node voxel-grid span when the config carries no lever value. A node
#: holds at most ``span**3`` points, so this is the per-node point budget.
DEFAULT_SPAN = 128

#: COPC point format (6 = the standard LAS 1.4 point with GPS time).
POINT_FORMAT_ID = 6

#: PIXC ``pixel_cloud`` coordinate variables, tried in order (X, Y, Z).
_LON_NAMES = ("longitude", "lon")
_LAT_NAMES = ("latitude", "lat")
_HEIGHT_NAMES = ("height", "elevation", "z")


class CopcParams(BaseModel):
    """COPC octree levers, parsed from ``config.params``.

    ``max_depth`` bounds the octree depth (``None`` derives a depth from the point
    count and ``span``, so a useful octree is built without a hand-tuned value).
    ``span`` is the per-node voxel-grid edge (the per-node point budget ≈
    ``span**3``). Both tolerate a swept *list* of values, taking the first so a
    swept lever degrades to a single run, mirroring COG's ``block_size``.
    ``scale`` is the LAS coordinate quantisation (``None`` derives it from the
    cloud extent so coordinates fit the LAS 32-bit grid without precision loss).
    """

    model_config = ConfigDict(extra="ignore")

    max_depth: Any = None
    span: Any = None
    scale: Any = None


def _first(value: Any, default: Any) -> Any:
    """Return ``value`` (first element if a swept list), or ``default`` for empty."""
    if value is None or value == []:
        return default
    if isinstance(value, list | tuple):
        return value[0]
    return value


def _derive_max_depth(n_points: int, span: int) -> int:
    """Pick an octree depth from the point count and per-node budget (``span**3``).

    Deep enough that the leaves are not overloaded — roughly ``log8(n / span**3)``
    levels — clamped to a sane range so a tiny cloud stays shallow and a huge one
    does not explode.
    """
    budget = max(1, span**3)
    if n_points <= budget:
        return 1
    return min(12, max(1, math.ceil(math.log(n_points / budget, 8)) + 1))


def _load_points(source: str, *, role: str = "source"):
    """Load ``(x, y, z)`` arrays from a point-cloud source.

    Dispatches on the source form: a ``PIXC:<granule>::<group>`` URI reads the
    netCDF group's lon/lat/height with ``xarray`` (the SWOT PIXC pixel cloud); a
    ``.las``/``.laz``/``.copc.laz`` path reads its points with ``laspy`` (the CO3D
    CARS reuse). Non-finite points are dropped.
    """
    import numpy as np

    if source.startswith(PIXC_SCHEME):
        granule_uri, _, group = source[len(PIXC_SCHEME) :].rpartition("::")
        x, y, z = _read_pixc_group(granule_uri, group or "pixel_cloud", role=role)
    elif source.lower().endswith((".las", ".laz")):
        x, y, z = _read_las(source)
    else:
        raise ValueError(
            f"COPC source {source!r} is neither a PIXC netCDF group "
            f"({PIXC_SCHEME}<granule>::<group>) nor a .las/.laz file"
        )

    x = np.asarray(x, dtype="float64").ravel()
    y = np.asarray(y, dtype="float64").ravel()
    z = np.asarray(z, dtype="float64").ravel()
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[finite], y[finite], z[finite]


def _read_pixc_group(granule_uri: str, group: str, *, role: str):
    """Read lon/lat/height from a netCDF ``group`` (local or S3) as point arrays."""
    import xarray as xr

    from cng_benchmark import storage

    if storage.is_s3(granule_uri):
        import fsspec

        handle = fsspec.open(
            granule_uri, mode="rb", **storage.fsspec_storage_options(role)
        ).open()
    elif granule_uri.startswith("file://"):
        handle = granule_uri[len("file://") :]
    else:
        handle = granule_uri

    ds = xr.open_dataset(handle, group=group, engine="h5netcdf")
    try:
        x = ds[_pick_var(ds, _LON_NAMES)].values
        y = ds[_pick_var(ds, _LAT_NAMES)].values
        z = ds[_pick_var(ds, _HEIGHT_NAMES)].values
    finally:
        ds.close()
    return x, y, z


def _pick_var(ds, names: tuple[str, ...]) -> str:
    """Return the first variable in ``ds`` matching ``names`` (case-insensitive)."""
    lower = {str(v).lower(): v for v in ds.variables}
    for name in names:
        if name in lower:
            return lower[name]
    raise KeyError(f"none of {names} found in group variables {list(ds.variables)}")


def _read_las(path: str):
    """Read ``(x, y, z)`` from a LAS/LAZ file with laspy."""
    import laspy

    with laspy.open(path) as f:
        las = f.read()
    return las.x, las.y, las.z


def _build_copc(
    path: str,
    x,
    y,
    z,
    *,
    span: int,
    max_depth: int,
    scale: float | None = None,
) -> None:
    """Write points to a COPC LAZ at ``path``, binning them into an octree.

    The root is the cubic bounds of the cloud. Each node voxel-downsamples its
    points to a ``span``-per-edge grid (one representative per occupied voxel) and
    passes the remainder down to the eight child octants, recursing until the
    points are exhausted or ``max_depth`` is reached (a leaf then holds whatever
    remains). Every point lands in exactly one node, so the cloud round-trips, and
    the hierarchy is connected (a node's parent always holds its own
    representatives). Pure ``copclib`` + ``numpy`` — CI-testable.
    """
    import copclib as copc
    import numpy as np

    xyz = np.stack(
        [np.asarray(x, "float64"), np.asarray(y, "float64"), np.asarray(z, "float64")],
        axis=1,
    )
    mn = xyz.min(axis=0)
    side = float((xyz.max(axis=0) - mn).max()) or 1.0
    if scale is None:
        # Keep quantised coordinates well within the LAS 32-bit grid (< 2**31).
        scale = max(side / 1e8, 1e-9)

    cfg = copc.CopcConfigWriter(POINT_FORMAT_ID, scale=[scale] * 3, offset=list(mn))
    cfg.las_header.min = list(mn)
    cfg.las_header.max = list(mn + side)
    center = mn + side / 2.0
    cfg.copc_info.center_x = center[0]
    cfg.copc_info.center_y = center[1]
    cfg.copc_info.center_z = center[2]
    cfg.copc_info.halfsize = side / 2.0
    cfg.copc_info.spacing = side / span
    writer = copc.FileWriter(path, cfg)

    # Iterative octree build: (key, point indices, depth, node-min corner, side).
    stack = [(copc.VoxelKey(0, 0, 0, 0), np.arange(len(xyz)), 0, mn.copy(), side)]
    try:
        while stack:
            key, idx, depth, nmin, nside = stack.pop()
            if len(idx) == 0:
                continue
            if depth >= max_depth:
                node_idx, rest = idx, np.empty(0, dtype=int)
            else:
                node_idx, rest = _voxel_split(xyz, idx, nmin, nside, span)
            _write_node(writer, key, xyz, node_idx)
            if len(rest):
                half = nside / 2.0
                octant = (xyz[rest] >= (nmin + half)).astype(int)
                for ox in (0, 1):
                    for oy in (0, 1):
                        for oz in (0, 1):
                            mask = (
                                (octant[:, 0] == ox)
                                & (octant[:, 1] == oy)
                                & (octant[:, 2] == oz)
                            )
                            if not mask.any():
                                continue
                            child = copc.VoxelKey(
                                depth + 1,
                                2 * key.x + ox,
                                2 * key.y + oy,
                                2 * key.z + oz,
                            )
                            cmin = nmin + np.array([ox, oy, oz]) * half
                            stack.append((child, rest[mask], depth + 1, cmin, half))
    finally:
        writer.Close()


def _voxel_split(xyz, idx, nmin, nside, span):
    """Split a node's points into voxel representatives and the remainder.

    Keeps one point per occupied cell of the node's ``span``-per-edge voxel grid
    (the node's points) and returns the rest for the child octants.
    """
    import numpy as np

    voxel = nside / span
    cell = np.clip(((xyz[idx] - nmin) / voxel).astype(np.int64), 0, span - 1)
    cell_id = (cell[:, 0] * span + cell[:, 1]) * span + cell[:, 2]
    _, first = np.unique(cell_id, return_index=True)
    keep = np.zeros(len(idx), dtype=bool)
    keep[first] = True
    return idx[keep], idx[~keep]


def _write_node(writer, key, xyz, node_idx) -> None:
    """Write one octree node's points (built from numpy) to the COPC writer."""
    import copclib as copc

    points = copc.Points(POINT_FORMAT_ID)
    points.AddPoints([points.CreatePoint() for _ in range(len(node_idx))])
    points.x = xyz[node_idx, 0]
    points.y = xyz[node_idx, 1]
    points.z = xyz[node_idx, 2]
    writer.AddNode(key, points)


def describe_copc_layout(path: str, name: str) -> CopcLayout:
    """Return the :class:`CopcLayout` of the COPC file at ``path``.

    Reads the octree hierarchy with copclib's ``FileReader``: the node count, the
    octree depth, the total point count, and the largest node's point count (the
    realised per-node budget).
    """
    import copclib as copc

    reader = copc.FileReader(path)
    nodes = reader.GetAllNodes()
    counts = [n.point_count for n in nodes]
    return CopcLayout(
        name=name,
        size_bytes=os.path.getsize(path),
        num_nodes=len(nodes),
        max_depth=reader.GetMaxDepth(),
        point_count=reader.copc_config.las_header.point_count,
        points_per_node=max(counts) if counts else 0,
    )


@FORMATS.register("copc")
class CopcAdapter(FormatAdapter):
    name = "copc"
    object_kind = ObjectKind.POINT_CLOUD_FILE

    def target_basename(self) -> str:
        return "copc.laz"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert a point-cloud ``source`` to a COPC file at ``target``.

        Loads the source points (a PIXC netCDF group, or a LAS/LAZ tile) and bins
        them into a COPC octree whose depth and per-node budget come from
        :class:`CopcParams`.
        """
        try:
            import copclib  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "COPC conversion requires the 'copc' extra; install with "
                "`uv sync --extra copc` (or `pip install cng-benchmark[copc]`)"
            ) from exc

        opts = CopcParams.model_validate(params)
        x, y, z = _load_points(source)
        if len(x) == 0:
            raise ValueError(f"COPC source {source!r} yielded no finite points")

        span = int(_first(opts.span, DEFAULT_SPAN))
        max_depth_value = _first(opts.max_depth, None)
        max_depth = (
            int(max_depth_value)
            if max_depth_value is not None
            else _derive_max_depth(len(x), span)
        )
        scale_value = _first(opts.scale, None)
        _build_copc(
            target,
            x,
            y,
            z,
            span=span,
            max_depth=max_depth,
            scale=float(scale_value) if scale_value is not None else None,
        )

    def describe_grouping_lever(self) -> str:
        return "COPC octree depth and per-node point budget"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the size (bytes) of the produced COPC file — a single object."""
        return [os.path.getsize(target)]

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[CopcLayout]:
        """Return the produced COPC file's octree-node layout (one object)."""
        return [describe_copc_layout(target, name or self.name)]
