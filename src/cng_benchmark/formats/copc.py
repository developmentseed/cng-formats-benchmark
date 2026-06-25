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

A SWOT PIXC ``pixel_cloud`` group is **content-complete**: not just the geometry
(lon/lat/height → x/y/z) but every other per-point variable (``sig0``,
``water_frac``, ``classification``, the quality flags, …) is carried as a LAS
**extra dimension**, preserving dtype — so the produced COPC's size is a
like-for-like basis for comparison with the source netCDF, not a geometry-only
fraction (issue #36). The carried set is configurable from the dataset ``options``
(see :mod:`cng_benchmark.datasets.swot_pixc`); by default every variable on the
point dimension is carried.

The point record is built with :mod:`laspy` (its extra-dimension API is
numpy-native and lays out LAS ExtraBytes correctly), and the COPC octree container
is written with :mod:`copclib`; the two are bridged by a one-point LAZ that hands
copclib the matching ExtraBytes VLR, after which each octree node is filled by a
vectorised :func:`copclib.Points.Unpack` of the laspy point bytes. Both are pip
wheels, so the builder / enumerate / layout logic is unit-testable in CI on a
synthetic cloud. The source read in :func:`_load_points` needs the granule stack —
``xarray`` + ``h5netcdf`` for a PIXC group, or ``laspy`` for a LAS/LAZ tile (the
CO3D CARS reuse) — imported lazily.

A COPC granule is large (a content-complete PIXC pass is many hundred MB), so as a
single object it already clears the cold tiers; the lever here is about preserving
range-addressable partial access, not reaching a size floor.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import CopcLayout
from cng_benchmark.registry import FORMATS

#: Prefix marking a component URI as a netCDF group read as a point cloud:
#: ``PIXC:<granule_uri>::<group>`` with an optional ``?include=…&exclude=…`` query
#: selecting the carried point variables (see :func:`_parse_pixc_uri`). Passed
#: through unchanged by ``storage.to_gdal_path`` (it is neither an ``s3://`` nor a
#: ``file://`` URI), like the SWOT raster reader's ``NETCDF:`` subdataset paths.
PIXC_SCHEME = "PIXC:"

#: Default per-node voxel-grid span when the config carries no lever value. A node
#: holds at most ``span**3`` points, so this is the per-node point budget.
DEFAULT_SPAN = 128

#: COPC point format (6 = the standard LAS 1.4 point with GPS time).
POINT_FORMAT_ID = 6

#: Safety cap on octree depth. The per-node point budget (``span**3``) is the real
#: terminator — a node is subdivided until it holds at most that many points — so
#: no single node ever materialises the whole cloud (the bound that keeps peak
#: memory under control). This cap only guards against non-separable (coincident)
#: points recursing forever; it is deep enough never to fire for real, distinct
#: point clouds.
_SAFETY_MAX_DEPTH = 21

#: Max length of a LAS extra-dimension name (the ExtraBytes name field is 32 bytes).
_MAX_EB_NAME = 32

#: PIXC ``pixel_cloud`` coordinate variables, tried in order (X, Y, Z).
_LON_NAMES = ("longitude", "lon")
_LAT_NAMES = ("latitude", "lat")
_HEIGHT_NAMES = ("height", "elevation", "z")


class CopcParams(BaseModel):
    """COPC octree levers, parsed from ``config.params``.

    ``span`` is the per-node voxel-grid edge and the **primary lever**: a node is
    subdivided until it holds at most ``span**3`` points (the per-node budget), so
    object grouping and peak build memory are both governed by ``span``.
    ``max_depth`` is an optional hard cap on octree depth (``None`` uses a high
    safety cap, :data:`_SAFETY_MAX_DEPTH`, letting the budget terminate the build).
    Both tolerate a swept *list* of values, taking the first so a swept lever
    degrades to a single run, mirroring COG's ``block_size``.
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


def _parse_pixc_uri(source: str) -> tuple[str, str, list[str] | None, list[str]]:
    """Parse a ``PIXC:<granule>::<group>?include=…&exclude=…`` component URI.

    Returns ``(granule_uri, group, include, exclude)`` where ``include`` is an
    explicit allow-list of carried point variables (``None`` = carry all on the
    point dimension) and ``exclude`` a deny-list. The granule URI (``s3://`` or
    local) is kept intact for the xarray/fsspec loader.
    """
    rest = source[len(PIXC_SCHEME) :]
    base, _, query = rest.partition("?")
    granule_uri, _, group = base.rpartition("::")
    include: list[str] | None = None
    exclude: list[str] = []
    for kv in query.split("&") if query else []:
        key, _, val = kv.partition("=")
        vals = [v for v in val.split(",") if v]
        if key == "include":
            include = vals
        elif key == "exclude":
            exclude = vals
    return granule_uri, group or "pixel_cloud", include, exclude


def _load_points(source: str, *, role: str = "source"):
    """Load ``(x, y, z, extras)`` from a point-cloud source.

    Dispatches on the source form: a ``PIXC:`` URI reads the netCDF group's
    lon/lat/height plus its other point variables with ``xarray`` (the SWOT PIXC
    pixel cloud); a ``.las``/``.laz`` path reads its geometry with ``laspy`` (the
    CO3D CARS reuse). ``extras`` maps a variable name to its per-point array.
    Points with a non-finite coordinate are dropped from every array together.
    """
    import numpy as np

    if source.startswith(PIXC_SCHEME):
        granule_uri, group, include, exclude = _parse_pixc_uri(source)
        x, y, z, extras = _read_pixc_group(
            granule_uri, group, role=role, include=include, exclude=exclude
        )
    elif source.lower().endswith((".las", ".laz")):
        x, y, z = _read_las(source)
        extras = {}
    else:
        raise ValueError(
            f"COPC source {source!r} is neither a PIXC netCDF group "
            f"({PIXC_SCHEME}<granule>::<group>) nor a .las/.laz file"
        )

    x = np.asarray(x, dtype="float64").ravel()
    y = np.asarray(y, dtype="float64").ravel()
    z = np.asarray(z, dtype="float64").ravel()
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    # Filter in place, replacing each source array as we go. A content-complete
    # granule carries ~50 point variables; building a second dict alongside the
    # first would double peak memory on a multi-million-point cloud.
    for name in list(extras):
        extras[name] = np.asarray(extras[name]).ravel()[finite]
    return x[finite], y[finite], z[finite], extras


def _read_pixc_group(
    granule_uri: str,
    group: str,
    *,
    role: str,
    include: list[str] | None,
    exclude: list[str],
):
    """Read geometry + the carried point variables from a netCDF ``group``."""
    import numpy as np
    import xarray as xr

    from cng_benchmark import storage

    tmp_download: str | None = None
    if storage.is_s3(granule_uri):
        # Download the granule to a local file with boto3, then open h5netcdf
        # from disk. The content-complete read pulls every point variable; doing
        # that as h5netcdf random-access reads over s3fs trips socket read
        # timeouts (FSTimeoutError) on a large granule over a slow endpoint.
        # boto3's sync multipart transfer (generous timeouts + retries) is
        # robust. The source netCDF is the conversion *input*, not the
        # cloud-native partial-access path — that is benchmarked on the produced
        # COPC (octree-node spatial query) — so reading it whole loses no signal.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_download = tmp.name
        storage.download_s3_object(granule_uri, tmp_download, role=role)
        handle = tmp_download
    elif granule_uri.startswith("file://"):
        handle = granule_uri[len("file://") :]
    else:
        handle = granule_uri

    ds = xr.open_dataset(handle, group=group, engine="h5netcdf")
    try:
        lon = _pick_var(ds, _LON_NAMES)
        lat = _pick_var(ds, _LAT_NAMES)
        height = _pick_var(ds, _HEIGHT_NAMES)
        x = np.asarray(ds[lon].values)
        y = np.asarray(ds[lat].values)
        z = np.asarray(ds[height].values)
        point_dim = ds[lon].dims[0]
        geometry = {lon, lat, height}
        carried = _select_point_vars(ds, point_dim, geometry, include, exclude)
        extras = {name: np.asarray(ds[name].values) for name in carried}
    finally:
        ds.close()
        if tmp_download is not None:
            os.unlink(tmp_download)
    return x, y, z, extras


def _select_point_vars(
    ds, point_dim, geometry: set[str], include: list[str] | None, exclude: list[str]
) -> list[str]:
    """Pick the point-dimensioned variables to carry as LAS extra dimensions.

    Candidates are the variables whose only dimension is the point dimension,
    minus the geometry triplet. ``include`` (if given) restricts to that allow-list
    in its order; ``exclude`` removes names. Default: every point variable.
    """
    candidates = [
        str(v)
        for v in ds.variables
        if tuple(ds[v].dims) == (point_dim,) and str(v) not in geometry
    ]
    wanted = (
        [v for v in include if v in candidates] if include is not None else candidates
    )
    excl = set(exclude)
    return [v for v in wanted if v not in excl]


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


def _las_extra_dtype(dtype):
    """Map a numpy dtype to the nearest LAS-ExtraBytes-supported numpy dtype.

    LAS extra dimensions support 1/2/4/8-byte signed/unsigned integers and 4/8-byte
    floats, so a variable of any of those types keeps its dtype exactly. A bool
    becomes ``uint8`` and a half float widens to ``float32`` (both lossless; LAS has
    neither type); a dtype LAS cannot represent at all (complex, datetime, …) falls
    back to ``float64`` — so a variable's values are preserved without inventing an
    unrepresentable on-disk type.
    """
    import numpy as np

    dt = np.dtype(dtype)
    if dt.kind == "b":
        return np.dtype("u1")
    if dt.kind == "f":
        return dt if dt.itemsize in (4, 8) else np.dtype("f4")
    if dt.kind in ("i", "u"):
        return dt if dt.itemsize in (1, 2, 4, 8) else np.dtype("i8")
    return np.dtype("f8")


def _sanitize_eb_name(name: str, used: set[str]) -> str:
    """Return a unique, length-bounded LAS extra-dimension name for ``name``."""
    base = name[:_MAX_EB_NAME]
    out = base
    i = 1
    while out in used:
        suffix = f"_{i}"
        out = base[: _MAX_EB_NAME - len(suffix)] + suffix
        i += 1
    used.add(out)
    return out


def _build_copc(
    path: str,
    x,
    y,
    z,
    extras: dict | None = None,
    *,
    span: int,
    max_depth: int,
    scale: float | None = None,
) -> None:
    """Write points (geometry + ``extras``) to a COPC LAZ at ``path``.

    The point record — x/y/z plus one LAS extra dimension per ``extras`` variable,
    dtype-mapped by :func:`_las_extra_dtype` — is assembled with laspy; a one-point
    LAZ hands copclib the matching ExtraBytes VLR. The cloud is then binned into a
    COPC octree (root = the cubic bounds; each node voxel-downsamples to a
    ``span``-per-edge grid and passes the remainder to its child octants, to
    ``max_depth``), each node filled by a vectorised ``Points.Unpack`` of the laspy
    point bytes — so every extra value round-trips. Pure ``laspy`` + ``copclib`` +
    ``numpy`` — CI-testable.
    """
    import copclib as copc
    import laspy
    import numpy as np

    extras = extras or {}
    x = np.asarray(x, "float64")
    y = np.asarray(y, "float64")
    z = np.asarray(z, "float64")
    xyz = np.stack([x, y, z], axis=1)
    mn = xyz.min(axis=0)
    side = float((xyz.max(axis=0) - mn).max()) or 1.0
    if scale is None:
        # Keep quantised coordinates well within the LAS 32-bit grid (< 2**31).
        scale = max(side / 1e8, 1e-9)

    header = laspy.LasHeader(point_format=POINT_FORMAT_ID)
    header.global_encoding.wkt = 1  # required for a LAS 1.4 / pf6 file
    header.offsets = mn
    header.scales = [scale, scale, scale]
    # Seed with the standard LAS dimension names (x/y/z, classification, …) so a
    # source variable that collides with a reserved name is carried under a
    # suffixed extra-dimension name rather than clashing with the point record.
    used: set[str] = set(laspy.PointFormat(POINT_FORMAT_ID).standard_dimension_names)
    # Declare the extra dims on the header first (schema only — no value copies),
    # so the LAS record can be allocated once.
    eb_names: dict[str, str] = {}
    for name, arr in extras.items():
        eb_name = _sanitize_eb_name(str(name), used)
        eb_names[name] = eb_name
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name=eb_name, type=_las_extra_dtype(np.asarray(arr).dtype)
            )
        )

    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    # Pack each variable into the record, then drop the source array. ``extras``
    # is consumed so the full source set and the full LAS record never coexist —
    # the peak that OOMs a content-complete, multi-million-point granule.
    for name in list(extras):
        arr = np.asarray(extras.pop(name))
        las[eb_names[name]] = arr.astype(_las_extra_dtype(arr.dtype))
        del arr
    records = las.points.array  # the packed LAS point records (incl. extra bytes)

    # Hand copclib a header carrying the matching ExtraBytes VLR via a 1-point LAZ.
    config, point_header = _copc_config_from_header(header, mn, scale)
    config.las_header.min = list(mn)
    config.las_header.max = list(mn + side)
    center = mn + side / 2.0
    config.copc_info.center_x = center[0]
    config.copc_info.center_y = center[1]
    config.copc_info.center_z = center[2]
    config.copc_info.halfsize = side / 2.0
    config.copc_info.spacing = side / span
    writer = copc.FileWriter(path, config)

    def node_points(idx):
        # View the selected records as raw bytes (no extra .tobytes() copy) and
        # hand them to copclib; the transient is bounded by the per-node budget.
        raw = records[idx].view(np.int8).reshape(-1)
        return copc.Points.Unpack(copc.VectorChar(raw), point_header)

    # A node holds at most ``span**3`` points; above that it is subdivided, so no
    # single node ever materialises a large fraction of the cloud.
    budget = max(1, span**3)
    # Iterative octree build: (key, point indices, depth, node-min corner, side).
    stack = [(copc.VoxelKey(0, 0, 0, 0), np.arange(len(xyz)), 0, mn.copy(), side)]
    try:
        while stack:
            key, idx, depth, nmin, nside = stack.pop()
            if len(idx) == 0:
                continue
            if len(idx) <= budget or depth >= max_depth:
                node_idx, rest = idx, np.empty(0, dtype=int)
            else:
                node_idx, rest = _voxel_split(xyz, idx, nmin, nside, span)
            writer.AddNode(key, node_points(node_idx))
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


def _copc_config_from_header(header, mn, scale: float):
    """Return ``(CopcConfigWriter, LasHeader)`` carrying ``header``'s ExtraBytes.

    copclib cannot build an ExtraBytes VLR from Python, so a one-point LAZ written
    from the laspy ``header`` is read back with copclib to obtain the matching VLR
    (for the COPC config) and the LAS header used to unpack point bytes.
    """
    import copclib as copc
    import laspy

    tiny = laspy.LasData(header)
    tiny.x, tiny.y, tiny.z = [mn[0]], [mn[1]], [mn[2]]
    with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as handle:
        tiny_path = handle.name
    try:
        tiny.write(tiny_path)
        laz_config = copc.LazReader(tiny_path).laz_config
        config = copc.CopcConfigWriter(
            POINT_FORMAT_ID,
            scale=[scale, scale, scale],
            offset=[mn[0], mn[1], mn[2]],
            wkt=laz_config.wkt,
            extra_bytes_vlr=laz_config.extra_bytes_vlr,
        )
        return config, laz_config.las_header
    finally:
        os.unlink(tiny_path)


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


def _copc_extra_dimensions(path: str) -> list[str]:
    """Return the COPC file's LAS extra-dimension names (the carried variables)."""
    import laspy

    reader = laspy.CopcReader.open(path)
    return list(reader.header.point_format.extra_dimension_names)


def describe_copc_layout(path: str, name: str) -> CopcLayout:
    """Return the :class:`CopcLayout` of the COPC file at ``path``.

    Reads the octree hierarchy with copclib's ``FileReader`` (node count, octree
    depth, total points, the largest node's point count) and the carried point
    variables from the LAS ExtraBytes schema with laspy — so the layout is
    self-describing about *what content* the object holds, not only its structure.
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
        extra_dimensions=_copc_extra_dimensions(path),
    )


@FORMATS.register("copc")
class CopcAdapter(FormatAdapter):
    name = "copc"
    object_kind = ObjectKind.POINT_CLOUD_FILE

    def target_basename(self) -> str:
        return "copc.laz"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert a point-cloud ``source`` to a content-complete COPC at ``target``.

        Loads the source points — for a PIXC group, the geometry plus every carried
        point variable — and bins them into a COPC octree whose depth and per-node
        budget come from :class:`CopcParams`, carrying the variables as LAS extra
        dimensions.
        """
        try:
            import copclib  # noqa: F401
            import laspy  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "COPC conversion requires the 'copc' extra; install with "
                "`uv sync --extra copc` (or `pip install cng-benchmark[copc]`)"
            ) from exc

        opts = CopcParams.model_validate(params)
        x, y, z, extras = _load_points(source)
        if len(x) == 0:
            raise ValueError(f"COPC source {source!r} yielded no finite points")

        span = int(_first(opts.span, DEFAULT_SPAN))
        max_depth_value = _first(opts.max_depth, None)
        max_depth = (
            int(max_depth_value)
            if max_depth_value is not None
            else _SAFETY_MAX_DEPTH
        )
        scale_value = _first(opts.scale, None)
        _build_copc(
            target,
            x,
            y,
            z,
            extras,
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
