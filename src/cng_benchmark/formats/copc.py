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

A converted cloud is **content-complete**: not just the geometry
(lon/lat/height → x/y/z) but every other per-point value is carried as a LAS
**extra dimension**, preserving dtype — so the produced COPC's size is a
like-for-like basis for comparison with the source, not a geometry-only fraction
(issue #36). For a SWOT PIXC ``pixel_cloud`` group that is every per-point
variable (``sig0``, ``water_frac``, ``classification``, the quality flags, …),
configurable from the dataset ``options`` (see
:mod:`cng_benchmark.datasets.swot_pixc`); for a CO3D CARS LAZ tile (see
:mod:`cng_benchmark.datasets.co3d`) it is the rest of the tile's point record.

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

from cng_benchmark.formats.base import EmptySourceError, FormatAdapter, ObjectKind
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

#: LAS dimensions holding the raw scaled integer geometry. They are the same
#: information as the x/y/z the loader already returns, so they are never carried
#: as extra dimensions.
_LAS_RAW_GEOMETRY_DIMS = ("X", "Y", "Z")

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
    """Load ``(x, y, z, extras, crs)`` from a point-cloud source.

    Dispatches on the source form: a ``PIXC:`` URI reads the netCDF group's
    lon/lat/height plus its other point variables with ``xarray`` (the SWOT PIXC
    pixel cloud), reprojecting lon/lat to a local UTM zone so x/y/z are all
    metric (:func:`_read_pixc_group`); a ``.las``/``.laz`` path reads its
    geometry plus the rest of its point record with ``laspy`` (the CO3D CARS
    tile), carrying through whatever CRS the tile itself declares, unchanged
    (:func:`_read_las`). ``extras`` maps a variable name to its per-point array;
    ``crs`` is a :class:`pyproj.CRS` when known, else ``None``.
    Points with a non-finite coordinate are dropped from every array together.
    """
    import numpy as np

    if source.startswith(PIXC_SCHEME):
        granule_uri, group, include, exclude = _parse_pixc_uri(source)
        x, y, z, extras, crs = _read_pixc_group(
            granule_uri, group, role=role, include=include, exclude=exclude
        )
    elif source.lower().endswith((".las", ".laz")):
        x, y, z, extras, crs = _read_las(source, role=role)
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
    return x[finite], y[finite], z[finite], extras, crs


def _read_pixc_group(
    granule_uri: str,
    group: str,
    *,
    role: str,
    include: list[str] | None,
    exclude: list[str],
):
    """Read geometry + the carried point variables from a netCDF ``group``.

    lon/lat (degrees, EPSG:4326) are reprojected to a local UTM zone chosen
    from the group's own centroid (:func:`_local_utm_crs`) before returning,
    so x/y come back already in metres, consistent with height (z), which is
    already metres. Un-reprojected lon/lat next to a metric height is what
    sizes the COPC octree cube off the height range alone (issue #134): x/y's
    ~0.001-1 degree footprint is numerically tiny next to a several-hundred-
    metre height range, so the octree cube ends up ~1000x too large in x/y
    and the first several levels split on height only, not space.
    """
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
        lon_deg = np.asarray(ds[lon].values, dtype="float64")
        lat_deg = np.asarray(ds[lat].values, dtype="float64")
        z = np.asarray(ds[height].values)
        point_dim = ds[lon].dims[0]
        geometry = {lon, lat, height}
        carried = _select_point_vars(ds, point_dim, geometry, include, exclude)
        extras = {name: np.asarray(ds[name].values) for name in carried}
    finally:
        ds.close()
        if tmp_download is not None:
            os.unlink(tmp_download)

    crs = _local_utm_crs(lon_deg, lat_deg)
    if crs is None:
        # No finite lon/lat to centre a zone on (an all-fill-value group);
        # leave geometry as degrees, same as before this fix. `_load_points`
        # drops all-non-finite rows anyway, so this cloud converts to nothing
        # (`EmptySourceError`) regardless of projection.
        return lon_deg, lat_deg, z, extras, None
    import pyproj

    x, y = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        lon_deg, lat_deg
    )
    return x, y, z, extras, crs


def _local_utm_crs(lon_deg, lat_deg):
    """Return the UTM zone CRS (a :class:`pyproj.CRS`) centred on the cloud.

    Chosen from the finite points' centroid: WGS84 UTM north for a
    non-negative centroid latitude, south otherwise, zone from the centroid
    longitude. ``None`` when there is no finite point to centre a zone on.
    """
    import numpy as np
    import pyproj

    finite = np.isfinite(lon_deg) & np.isfinite(lat_deg)
    if not finite.any():
        return None
    lon_c = float(np.mean(lon_deg[finite]))
    lat_c = float(np.mean(lat_deg[finite]))
    zone = int((lon_c + 180) / 6) + 1
    epsg = (32600 if lat_c >= 0 else 32700) + zone
    return pyproj.CRS.from_epsg(epsg)


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


def _read_las(path: str, *, role: str = "source"):
    """Read ``(x, y, z, extras, crs)`` from a LAS/LAZ tile with laspy.

    Content-complete like the PIXC group read: every non-geometry dimension the
    tile's point record declares — the standard LAS fields it carries (intensity,
    classification, the CARS colour channels, …) *and* its own extra dimensions —
    travels in ``extras``, so the produced COPC is a like-for-like basis for
    comparison with the source tile rather than a geometry-only fraction (#36).
    ``X``/``Y``/``Z`` are the raw scaled integers of the geometry already carried
    as ``x``/``y``/``z``, so they are skipped; a name that collides with a
    reserved dimension of the *target* point format is suffixed by the writer
    (:func:`_sanitize_eb_name`).

    ``crs`` is the source tile's own declared CRS (``laspy``'s
    ``header.parse_crs()``, ``None`` when the tile carries no CRS VLR at all —
    common for a CARS delivery, whose x/y/z are already in one consistent
    metric CRS but do not name it). Carried through unchanged, never guessed:
    x/y/z are not reprojected here, unlike the PIXC path (:func:`_read_pixc_group`),
    whose lon/lat/height inputs are not already metric.

    laspy needs a local, seekable file — a LAZ chunk table is read by random
    access — so an S3 tile is downloaded first, as the PIXC granule read is.
    """
    import laspy
    import numpy as np

    from cng_benchmark import storage

    uri = storage.from_gdal_path(path)
    tmp_download: str | None = None
    if storage.is_s3(uri):
        with tempfile.NamedTemporaryFile(suffix=".laz", delete=False) as tmp:
            tmp_download = tmp.name
        storage.download_s3_object(uri, tmp_download, role=role)
        handle = tmp_download
    elif uri.startswith("file://"):
        handle = uri[len("file://") :]
    else:
        handle = uri

    try:
        with laspy.open(handle) as reader:
            las = reader.read()
        extras = {
            name: np.asarray(las[name])
            for name in las.point_format.dimension_names
            if name not in _LAS_RAW_GEOMETRY_DIMS
        }
        return las.x, las.y, las.z, extras, las.header.parse_crs()
    finally:
        if tmp_download is not None:
            os.unlink(tmp_download)


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
    crs=None,
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

    ``x``/``y``/``z`` must already be in one consistent metric unit (the caller's
    job: :func:`_load_points` reprojects PIXC's geographic lon/lat to a local UTM
    zone before calling this, and carries a LAS/LAZ source's own CRS through
    unchanged, since its x/y/z are already metric). Building the octree cube on
    mismatched units (geographic x/y next to a metric z) makes ``side`` come out
    driven by whichever axis happens to be numerically largest rather than by
    real spatial extent (issue #134's original defect).

    ``crs`` (a :class:`pyproj.CRS`), when known, is written into the header as a
    WKT VLR (:meth:`laspy.LasHeader.add_crs`), so the produced COPC is
    self-describing; ``None`` leaves the header without one, same as before this
    parameter existed, for a source whose CRS is genuinely unknown (an
    undeclared CARS delivery). ``scale``, left ``None``, is now derived
    **per axis** from that axis's own extent (floored at 1 mm) rather than from
    the largest of the three, so an axis that varies far less than the others
    (height inside a small footprint tile, or the reverse) is not quantised
    coarser than it needs to be.
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
    extent = xyz.max(axis=0) - mn
    side = float(extent.max()) or 1.0
    if scale is None:
        # Per-axis: an axis that varies far less than the cube's overall side
        # (a flat height range inside a wide footprint, or the reverse) keeps
        # its own precision floor (1 mm) rather than inheriting the coarser
        # scale the other axes would force under one shared value. Still kept
        # well within the LAS 32-bit grid (< 2**31 counts) per axis.
        scale_axes = [max(float(e) / 2**31, 1e-3) for e in extent]
    else:
        scale_axes = [float(scale)] * 3

    header = laspy.LasHeader(point_format=POINT_FORMAT_ID)
    header.global_encoding.wkt = 1  # required for a LAS 1.4 / pf6 file
    header.offsets = mn
    header.scales = scale_axes
    if crs is not None:
        header.add_crs(crs)
    # The target format's own standard fields (x/y/z, classification,
    # intensity, …). A source variable whose name matches one of these is the
    # *same field*, carried over from a source point format that also has it
    # (e.g. classification/intensity/return_number are standard on both format
    # 2, a common CARS delivery format, and format 6 here) — it belongs in that
    # native slot, not as a renamed ExtraBytes duplicate that leaves the real
    # field empty and doubles the bytes for no reason. Only a name with no home
    # in the target format's own fields (RGB, a source-specific quality flag, …)
    # is a genuine extra dimension.
    native_names = set(laspy.PointFormat(POINT_FORMAT_ID).standard_dimension_names) - {
        "X",
        "Y",
        "Z",
    }
    native_overlap = {
        name: extras.pop(name) for name in list(extras) if name in native_names
    }
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
    # Assign the native-field overlap straight into the target format's own
    # dimensions (laspy casts to each field's LAS-mandated storage type).
    for name in list(native_overlap):
        las[name] = np.asarray(native_overlap.pop(name))
    # Pack each remaining (genuine-extra) variable into the record, then drop
    # the source array. ``extras`` is consumed so the full source set and the
    # full LAS record never coexist — the peak that OOMs a content-complete,
    # multi-million-point granule.
    for name in list(extras):
        arr = np.asarray(extras.pop(name))
        las[eb_names[name]] = arr.astype(_las_extra_dtype(arr.dtype))
        del arr
    records = las.points.array  # the packed LAS point records (incl. extra bytes)

    # Hand copclib a header carrying the matching ExtraBytes VLR via a 1-point LAZ.
    config, point_header = _copc_config_from_header(header, mn, scale_axes)
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

    # Iterative octree build: (key, point indices, depth, node-min corner, side).
    # Every node is voxel-sampled, even one already under the span**3 budget:
    # `_voxel_split` bounds a node to at most one point per occupied cell of its
    # own span-per-edge grid, so it is already <= span**3 by construction, budget
    # or not (issue #134's second defect: a `len(idx) <= budget` short-circuit
    # here wrote such a node whole, skipping the spatial thinning that makes a
    # coarse level an actual overview instead of a full-density subset).
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


def _copc_config_from_header(header, mn, scale_axes: list[float]):
    """Return ``(CopcConfigWriter, LasHeader)`` carrying ``header``'s ExtraBytes.

    copclib cannot build an ExtraBytes VLR from Python, so a one-point LAZ written
    from the laspy ``header`` is read back with copclib to obtain the matching VLR
    (for the COPC config), the CRS WKT if ``header`` carries one
    (:meth:`laspy.LasHeader.add_crs`), and the LAS header used to unpack point
    bytes. ``scale_axes`` is the per-axis ``[sx, sy, sz]`` quantisation.
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
        # copclib's own ``LazReader.laz_config.wkt`` mis-parses the WKT VLR
        # laspy just wrote (drops its first 6 bytes, "PROJCR" off a
        # "PROJCRS[..." string — confirmed by re-reading the same tiny LAZ
        # with laspy itself, which gets the full, correct WKT back). Use
        # pyproj's own ``to_wkt()`` instead, straight from the header's own
        # CRS, sidestepping that round trip for this one field; the ExtraBytes
        # VLR and LAS header from the same round trip are unaffected and kept.
        source_crs = header.parse_crs()
        wkt = source_crs.to_wkt() if source_crs is not None else laz_config.wkt
        config = copc.CopcConfigWriter(
            POINT_FORMAT_ID,
            scale=list(scale_axes),
            offset=[mn[0], mn[1], mn[2]],
            wkt=wkt,
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
    header = reader.copc_config.las_header
    size_bytes = os.path.getsize(path)
    uncompressed = header.point_count * header.point_record_length
    return CopcLayout(
        name=name,
        size_bytes=size_bytes,
        num_nodes=len(nodes),
        max_depth=reader.GetMaxDepth(),
        point_count=header.point_count,
        points_per_node=max(counts) if counts else 0,
        extra_dimensions=_copc_extra_dimensions(path),
        codec="laszip",
        compression_ratio=(uncompressed / size_bytes) if size_bytes else 0.0,
    )


def render_copc_lod(
    path: str, out_path: str, *, max_points: int = 120_000, color_by: str = "auto"
) -> str:
    """Render the COPC clustered-octree level-of-detail to ``out_path`` (a PNG).

    Three top-down panels — a coarse overview, a mid level, and the full cloud —
    illustrate how a reader fetches progressively deeper octree nodes: the COPC
    analogue of COG overviews / the tile-footprint diagram, and the point-cloud
    structural artifact of a run. The level cuts are chosen by **point fraction**
    (roughly 10%, 50% and 100% of the cloud), not by a bare depth fraction: point
    density is not uniform across levels, so a depth-based cut can put nearly the
    whole cloud in the last panel and almost nothing in the others. Each panel's
    dot budget is proportional to the share of the cloud it actually holds
    (capped at ``max_points`` overall, for the full-detail panel), so the figure
    shows real densification panel to panel instead of every panel maxing out
    the same fixed dot count regardless of how sparse it is (issue #134's third
    defect). Requires matplotlib (the ``cog`` extra); returns ``out_path``.

    ``color_by`` picks what the scatter is coloured on, since a flat colour makes
    a photogrammetric or coloured-LiDAR cloud unreadable as a *cloud* (no terrain,
    no structure, just a point count):

    - ``"auto"`` (default): the point cloud's own ``red``/``green``/``blue`` when
      the file carries them (a photo-realistic render, the most representative
      choice for CO3D/CARS-family colour clouds), else height (``z``).
    - ``"rgb"``: force RGB; raises if the file carries no red/green/blue.
    - ``"z"``: height, on a continuous colormap with a shared colourbar.
    - any other carried dimension name (e.g. ``"intensity"``,
      ``"classification"``, ``"scan_angle_rank"``): coloured on that dimension's
      values, same colormap/colourbar treatment as height.
    - ``"flat"``: the original single-colour scatter.

    RGB is contrast-stretched once, per channel, against the cloud's own 2nd
    and 98th percentile (not its bare min/max, which a handful of outlier
    pixels can compress the real range against): a photogrammetric or
    natural-scene cloud (rock, vegetation, concrete) rarely spans the full
    0-65535 LAS RGB range writers nominally allow, so normalising against the
    true extremes alone under-uses the colour range and reads as flat,
    washed-out grey. Every other coloured mode shares one min/max
    normalisation across all three panels, computed from the full-detail
    query, so colour is directly comparable panel to panel, not renormalised
    per panel.
    """
    import copclib as copc
    import laspy
    import numpy as np

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display in the runner
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the COPC level-of-detail image requires matplotlib (in the 'cog' "
            "extra); install with `uv sync --extra cog`"
        ) from exc

    reader = laspy.CopcReader.open(path)
    # `dimension_names` lists raw X/Y/Z (uppercase) but not the scaled x/y/z
    # accessors every point record still exposes, so add those explicitly.
    available = set(reader.header.point_format.dimension_names) | {"x", "y", "z"}

    mode = color_by
    if mode == "auto":
        mode = "rgb" if {"red", "green", "blue"}.issubset(available) else "z"
    if mode == "rgb" and not {"red", "green", "blue"}.issubset(available):
        raise ValueError(
            f"render_copc_lod: color_by='rgb' but {path!r} carries no "
            "red/green/blue dimension"
        )
    if mode not in ("rgb", "z", "flat") and mode not in available:
        raise ValueError(
            f"render_copc_lod: color_by={color_by!r} is not a dimension "
            f"{path!r} carries (has: {sorted(available)})"
        )

    depth = copc.FileReader(path).GetMaxDepth()
    # Per-level point counts -> a cumulative-fraction curve, so both the level
    # cuts and each panel's dot budget below are chosen by how many points a
    # level actually adds, not by its bare depth number.
    level_counts = [len(reader.query(level=lvl)) for lvl in range(depth + 1)]
    cum_counts = list(np.cumsum(level_counts))
    total = cum_counts[-1] if cum_counts else 0

    def _cut_for_fraction(frac: float) -> int:
        target = frac * total
        for lvl, c in enumerate(cum_counts):
            if c >= target:
                return lvl
        return depth

    cuts = sorted({_cut_for_fraction(0.1), _cut_for_fraction(0.5), depth})
    labels = ("coarse overview", "mid levels", "full detail")
    rng = np.random.default_rng(0)

    # One full-depth query up front: the "full detail" panel needs it anyway,
    # and it gives a stable colour normalisation shared across all 3 panels.
    full_pts = reader.query(level=range(0, cuts[-1] + 1))
    rgb_lo = np.zeros(3)
    rgb_span = np.ones(3)
    value_range: tuple[float, float] | None = None
    dim_name = "z" if mode == "z" else mode
    if mode == "rgb":
        rgb_full = np.stack(
            [
                np.asarray(full_pts.red, dtype="float64"),
                np.asarray(full_pts.green, dtype="float64"),
                np.asarray(full_pts.blue, dtype="float64"),
            ],
            axis=1,
        )
        rgb_lo = np.percentile(rgb_full, 2, axis=0)
        rgb_hi = np.percentile(rgb_full, 98, axis=0)
        rgb_span = np.maximum(rgb_hi - rgb_lo, 1.0)
    elif mode != "flat":
        values = np.asarray(getattr(full_pts, dim_name), dtype="float64")
        value_range = (float(values.min()), float(values.max()))

    fig, axes = plt.subplots(
        1, len(cuts), figsize=(4.5 * len(cuts), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    mappable = None
    for ax, cut, label in zip(axes, cuts, labels, strict=False):
        pts = full_pts if cut == cuts[-1] else reader.query(level=range(0, cut + 1))
        x = np.asarray(pts.x)
        y = np.asarray(pts.y)
        # This panel's dot budget scales with the share of the cloud it holds
        # (the full-detail panel always gets the full `max_points`), so the
        # coarse panel is visibly sparser rather than padded out to look as
        # dense as the full cloud.
        panel_budget = (
            max(1, round(max_points * cum_counts[cut] / total)) if total else max_points
        )
        idx = None
        if len(x) > panel_budget:
            idx = rng.choice(len(x), panel_budget, replace=False)
            x, y = x[idx], y[idx]
        if mode == "rgb":
            rgb = np.stack(
                [
                    np.asarray(pts.red, dtype="float64"),
                    np.asarray(pts.green, dtype="float64"),
                    np.asarray(pts.blue, dtype="float64"),
                ],
                axis=1,
            )
            if idx is not None:
                rgb = rgb[idx]
            colors = np.clip((rgb - rgb_lo) / rgb_span, 0, 1)
            ax.scatter(x, y, s=0.4, c=colors, alpha=0.8, linewidths=0)
        elif mode == "flat":
            ax.scatter(x, y, s=0.4, c="#1d4ed8", alpha=0.5, linewidths=0)
        else:
            values = np.asarray(getattr(pts, dim_name), dtype="float64")
            if idx is not None:
                values = values[idx]
            mappable = ax.scatter(
                x,
                y,
                s=0.4,
                c=values,
                cmap="viridis",
                vmin=value_range[0],
                vmax=value_range[1],
                alpha=0.8,
                linewidths=0,
            )
        ax.set_title(f"{label}\nlevels 0–{cut} · {len(pts):,} pts", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    if mappable is not None:
        fig.colorbar(mappable, ax=list(axes), shrink=0.75, pad=0.02, label=dim_name)
    color_caption = {
        "rgb": ", coloured by the cloud's own RGB",
        "flat": "",
    }.get(mode, f", coloured by {dim_name}")
    fig.suptitle(
        "COPC clustered-octree level-of-detail"
        f"{color_caption} — a reader fetches only the octree nodes it needs:\n"
        "coarse levels for an overview, deeper levels (or a bbox) for detail "
        "(the source netCDF has no such index)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


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
        x, y, z, extras, crs = _load_points(source)
        if len(x) == 0:
            raise EmptySourceError(f"COPC source {source!r} yielded no finite points")

        span = int(_first(opts.span, DEFAULT_SPAN))
        max_depth_value = _first(opts.max_depth, None)
        max_depth = (
            int(max_depth_value) if max_depth_value is not None else _SAFETY_MAX_DEPTH
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
            crs=crs,
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
