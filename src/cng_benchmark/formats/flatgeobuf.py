"""FlatGeobuf adapter — vector features to a single indexed ``.fgb`` file.

FlatGeobuf is the **second** cloud-native vector candidate of the CNES study
(D2 ch. 2, ch. 4, ch. 6.4). GeoParquet was the only one measured, so the vector
recommendation rested on one measured candidate; this adapter is the other point
on the same source (#83).

The grouping lever for FlatGeobuf is its **packed Hilbert R-tree**. The format
stores one flatbuffer per feature, so the feature is the addressable unit, and the
R-tree written between the header and the feature data is what makes a feature
reachable: a client walks the tree with a bbox predicate, then range-reads only
the features the tree selects. Without that tree the file can only be scanned, so
an unindexed FlatGeobuf is not the candidate under evaluation — the index is on by
default here (``params['spatial_index']``), and writing it also orders the
features along a Hilbert curve, which keeps the selected features contiguous in
the file. Turning it off measures exactly what the index buys.

The tree's branching factor (``index_node_size``, 16) is fixed by the writer and
is not a creation option, so unlike GeoParquet's row-group size the grouping is a
property of the format, not a knob to sweep. What a run can decide is whether the
index exists at all; :func:`describe_flatgeobuf_layout` reports the realised node
size, the feature count and what the tree costs in bytes, so the "what does the
index cost" question is answerable from the result alone.

This adapter writes one ``.fgb`` file per component — a single ``VECTOR_FILE``
object, like the GeoParquet arm — and flows through the same runner paths, with
the read metric a bbox query pushed through the index
(:func:`cng_benchmark.metrics.read.measure_vector_read`). There is no display
surface (a vector table is not a TiTiler raster tile).

Both the write and the source read go through ``pyogrio`` (its manylinux wheel
bundles the GDAL FlatGeobuf driver — no system GDAL), so the whole path is
CI-testable. The layout describer needs no driver at all: it parses the file's
own header flatbuffer and derives the index size from the published packed-R-tree
formula, which is a preamble read rather than a feature scan.
"""

from __future__ import annotations

import os
import struct
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import FlatGeobufLayout
from cng_benchmark.registry import FORMATS

#: The FlatGeobuf magic bytes: ``fgb`` + the spec major version, twice. The second
#: copy's last byte is the patch version, which changes between spec releases, so a
#: file is recognised by this 7-byte prefix only.
MAGIC_PREFIX = b"fgb\x03fgb"

#: File preamble: 8 magic bytes, then a ``uint32`` header length, then the header.
_HEADER_LEN_OFFSET = 8
_HEADER_OFFSET = 12

#: One packed R-tree node on disk: 4 doubles (the node bbox) + a ``uint64`` offset.
NODE_ITEM_BYTES = 40

#: The R-tree branching factor the writer uses. The FlatGeobuf spec defaults to 16
#: and the GDAL driver exposes no creation option for it, so it is a constant of
#: the produced file. A file written without an index records 0 here instead.
DEFAULT_INDEX_NODE_SIZE = 16

#: ``Header`` table field indices (flatbuffer vtable slots) in the FlatGeobuf
#: schema. A field absent from the vtable carries its schema default.
_FIELD_GEOMETRY_TYPE = 2
_FIELD_FEATURES_COUNT = 8
_FIELD_INDEX_NODE_SIZE = 9

#: The FlatGeobuf ``GeometryType`` enum, by value.
GEOMETRY_TYPES = (
    "Unknown",
    "Point",
    "LineString",
    "Polygon",
    "MultiPoint",
    "MultiLineString",
    "MultiPolygon",
    "GeometryCollection",
    "CircularString",
    "CompoundCurve",
    "CurvePolygon",
    "MultiCurve",
    "MultiSurface",
    "Curve",
    "Surface",
    "PolyhedralSurface",
    "TIN",
    "Triangle",
)


class FlatGeobufParams(BaseModel):
    """FlatGeobuf grouping levers, parsed from ``config.params``.

    ``spatial_index`` (default on) writes the packed Hilbert R-tree and orders the
    features along the Hilbert curve. It is the only writer lever the format has:
    the tree's node size is fixed at :data:`DEFAULT_INDEX_NODE_SIZE`, and there is
    no compression to choose. Set it to ``false`` only to measure what the index
    costs and what it buys — an unindexed file is not the candidate format.

    ``null_geometry`` decides how to handle a source that carries features with no
    geometry at all — a prior-database delivery such as SWOT LakeSP Prior reports
    every prior feature in a pass's footprint and leaves the geometry NULL for the
    ones not observed (#98). FlatGeobuf's packed Hilbert R-tree cannot index a NULL
    geometry, so this is a hard format constraint whenever ``spatial_index`` is on:

    * ``"error"`` (default) fails the conversion before any bytes are written,
      naming the count and share.
    * ``"drop"`` writes the non-null subset instead; the dropped count is
      recorded on :class:`~cng_benchmark.models.FlatGeobufLayout` so the result
      self-labels as a content subset.
    * ``"sentinel"`` keeps every feature, substituting a minimal placeholder
      geometry (same broad type as the layer's real content, so
      ``geometry_type`` stays meaningful) positioned just outside the real
      features' extent for the NULL ones, so it can never spuriously satisfy a
      bbox query scoped to the real content — "not searchable as geometry"
      without dropping the row or its attributes. A ``null_geometry`` boolean
      column flags which rows are placeholders, and the count is recorded on
      the layout: those bytes are fabricated, not part of the source, so they
      should not be read into a size comparison against another format's
      (near-free) NULL representation. Only use this when the source carries
      no usable position for the NULL-geometry rows at all — ``"point_from"``
      is the better choice whenever it does.
    * ``"point_from"`` keeps every feature too, but builds a *real* geometry —
      ``Point(lon, lat)`` from ``null_geometry_point_from: [lon_col, lat_col]``
      — rather than a placeholder. Investigating SWOT LakeSP Prior found the
      NULL-geometry rows are not positionless: they carry the prior lake's own
      reference coordinates (``p_lon``/``p_lat``) on every row, geometry or
      not — the location was never missing, only its geometry encoding was
      (#98). The synthesized count is recorded on the layout
      (``features_synthesized``/``geometry_synthesized``) so the run still
      says what happened, without the "fabricated, unreachable" caveat
      ``"sentinel"`` carries — this geometry is real and genuinely queryable.
    """

    model_config = ConfigDict(extra="ignore")

    spatial_index: bool = True
    null_geometry: Literal["error", "drop", "sentinel", "point_from"] = "error"
    #: ``[lon_column, lat_column]``, required when ``null_geometry: point_from``.
    null_geometry_point_from: tuple[str, str] | None = None


class FlatGeobufHeader(NamedTuple):
    """The parsed FlatGeobuf header: what the file says about its own layout."""

    header_bytes: int
    features_count: int
    index_node_size: int
    geometry_type: str


def _table_field(buf: bytes, table: int, index: int) -> int | None:
    """Return the absolute offset of flatbuffer table field ``index``, or ``None``.

    A flatbuffer table starts with a signed offset back to its vtable, which lists
    one ``uint16`` per field: 0, or a field offset beyond the vtable's own length,
    means the field was not written and the reader must use the schema default.
    """
    (soffset,) = struct.unpack_from("<i", buf, table)
    vtable = table - soffset
    (vtable_bytes,) = struct.unpack_from("<H", buf, vtable)
    slot = 4 + 2 * index
    if slot >= vtable_bytes:
        return None
    (offset,) = struct.unpack_from("<H", buf, vtable + slot)
    return table + offset if offset else None


def read_flatgeobuf_header(path: str) -> FlatGeobufHeader:
    """Parse the FlatGeobuf header of the file at ``path``.

    Reads the preamble only — the magic, the header length and the header
    flatbuffer — so this stays cheap on a large file. Returns the feature count,
    the R-tree branching factor (0 when the file carries no index) and the layer
    geometry type. Raises ``ValueError`` when ``path`` is not a FlatGeobuf.
    """
    with open(path, "rb") as fh:
        preamble = fh.read(_HEADER_OFFSET)
        if not preamble.startswith(MAGIC_PREFIX):
            raise ValueError(f"{path!r} is not a FlatGeobuf file (bad magic)")
        (header_bytes,) = struct.unpack_from("<I", preamble, _HEADER_LEN_OFFSET)
        buf = fh.read(header_bytes)

    (root,) = struct.unpack_from("<I", buf, 0)
    count = _table_field(buf, root, _FIELD_FEATURES_COUNT)
    node_size = _table_field(buf, root, _FIELD_INDEX_NODE_SIZE)
    geometry = _table_field(buf, root, _FIELD_GEOMETRY_TYPE)
    geometry_value = struct.unpack_from("<B", buf, geometry)[0] if geometry else 0
    return FlatGeobufHeader(
        header_bytes=header_bytes,
        features_count=struct.unpack_from("<Q", buf, count)[0] if count else 0,
        index_node_size=(
            struct.unpack_from("<H", buf, node_size)[0]
            if node_size
            else DEFAULT_INDEX_NODE_SIZE
        ),
        geometry_type=(
            GEOMETRY_TYPES[geometry_value]
            if geometry_value < len(GEOMETRY_TYPES)
            else "Unknown"
        ),
    )


def packed_rtree_bytes(features_count: int, index_node_size: int) -> int:
    """Return the on-disk size of the packed Hilbert R-tree over ``features_count``.

    The tree is static and complete, so its size follows from the item count and
    the branching factor alone (the FlatGeobuf spec's own formula): the leaf level
    holds one node per feature, each level above holds ``ceil(n / node_size)``
    nodes, and every node occupies :data:`NODE_ITEM_BYTES`. A node size below 2
    means no index — no tree is written and the answer is 0 bytes.
    """
    if features_count < 1 or index_node_size < 2:
        return 0
    level = features_count
    nodes = level
    while level != 1:
        level = -(-level // index_node_size)  # ceil division
        nodes += level
    return nodes * NODE_ITEM_BYTES


def describe_flatgeobuf_layout(
    path: str,
    name: str,
    *,
    features_dropped: int = 0,
    features_sentinel: int = 0,
    features_synthesized: int = 0,
) -> FlatGeobufLayout:
    """Return the :class:`FlatGeobufLayout` of the FlatGeobuf file at ``path``.

    Splits the stored object into its three parts — header, R-tree, feature data —
    from the header alone, so what the index costs is answerable next to the size
    it costs it in. ``compression_ratio`` is 1.0: FlatGeobuf defines no
    compression, so the stored bytes *are* the uncompressed bytes.

    ``features_dropped``/``features_sentinel``/``features_synthesized`` cannot be
    recovered from ``path`` alone (the file only knows what it contains, not what
    was excluded or substituted), so the caller — the adapter that just wrote it
    — passes them through from its own ``null_geometry`` accounting (#98).
    """
    header = read_flatgeobuf_header(path)
    size_bytes = os.path.getsize(path)
    index_bytes = packed_rtree_bytes(header.features_count, header.index_node_size)
    feature_bytes = size_bytes - _HEADER_OFFSET - header.header_bytes - index_bytes
    return FlatGeobufLayout(
        name=name,
        size_bytes=size_bytes,
        geometry_type=header.geometry_type,
        num_features=header.features_count,
        has_spatial_index=index_bytes > 0,
        index_node_size=header.index_node_size,
        header_bytes=header.header_bytes,
        index_bytes=index_bytes,
        feature_bytes=max(0, feature_bytes),
        codec="none",
        compression_ratio=1.0,
        features_dropped=features_dropped,
        content_subset=features_dropped > 0,
        features_sentinel=features_sentinel,
        geometry_fabricated=features_sentinel > 0,
        features_synthesized=features_synthesized,
        geometry_synthesized=features_synthesized > 0,
    )


def _minimal_geometry_like(geom: Any, x: float, y: float, eps: float = 1e-9) -> Any:
    """A minimal, valid, non-empty geometry of ``geom``'s broad type, at ``(x, y)``.

    GDAL's FlatGeobuf writer refuses a NULL *or* an empty geometry when the
    spatial index is on (both tested for #98) — a placeholder has to be real,
    if tiny. Dispatches on the broad OGC family (point-like, line-like,
    everything else) rather than the exact subtype, because OGR promotes a
    ``Polygon`` cleanly into a ``MultiPolygon``-typed layer (verified for #98)
    but not a ``Point`` into either — a mismatched sentinel type would make
    the whole layer's declared ``geometry_type`` report as ``"Unknown"``.
    """
    from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

    if isinstance(geom, Point | MultiPoint):
        return Point(x, y)
    if isinstance(geom, LineString | MultiLineString):
        return LineString([(x, y), (x + eps, y + eps)])
    from shapely.geometry import Polygon

    return Polygon([(x, y), (x + eps, y), (x, y + eps)])


def _unique_column_name(gdf, base: str) -> str:
    """Return ``base``, or ``base`` suffixed until it doesn't collide with ``gdf``."""
    name = base
    while name in gdf.columns:
        name += "_"
    return name


def _flag_null_geometry_rows(gdf, out, null_mask) -> None:
    """Add a boolean column to ``out`` (in place) marking ``null_mask``'s rows.

    Shared by ``sentinel`` and ``point_from``: whichever geometry a NULL row
    was given, the flag says it did not arrive with one. Named ``null_geometry``,
    or a collision-safe variant if the source already has a column by that name.
    """
    flag_col = _unique_column_name(gdf, "null_geometry")
    out[flag_col] = null_mask.to_numpy()


def _substitute_sentinel_geometries(gdf, null_mask) -> Any:
    """Replace every NULL geometry in ``gdf`` with a same-type placeholder.

    Placed just outside the real (non-null) features' bounding box — in the
    layer's own CRS, offset by both a fixed and an extent-proportional margin
    so it holds even for a near-point-sized real extent — so a bbox query scoped
    to the real content can never spuriously select it: "not searchable as
    geometry" by construction, without dropping the row or its attributes.
    """
    out = gdf.copy()
    minx, miny, maxx, maxy = out.loc[~null_mask].total_bounds
    sentinel_x = minx - (maxx - minx) * 0.01 - 1.0
    sentinel_y = miny - (maxy - miny) * 0.01 - 1.0
    template = out.loc[~null_mask].geometry.iloc[0]
    out.loc[null_mask, "geometry"] = _minimal_geometry_like(
        template, sentinel_x, sentinel_y
    )
    _flag_null_geometry_rows(gdf, out, null_mask)
    return out


def _substitute_point_from_columns(gdf, null_mask, lon_col: str, lat_col: str) -> Any:
    """Replace every NULL geometry in ``gdf`` with ``Point(lon_col, lat_col)``.

    Unlike :func:`_substitute_sentinel_geometries`, this is not a fabricated
    placeholder: investigating SWOT LakeSP Prior (#98) found the NULL-geometry
    rows carry the prior lake's own reference coordinates on every row, geometry
    or not — the location was never missing, only its geometry encoding was. A
    named column absent from ``gdf``, or missing a value on a NULL-geometry row,
    means a position genuinely cannot be synthesized for it, and raises naming
    which — silently guessing a position would be worse than any of the other
    policies.
    """
    for col in (lon_col, lat_col):
        if col not in gdf.columns:
            raise ValueError(
                f"null_geometry: point_from names column {col!r}, which is not "
                f"in the source (columns: {', '.join(gdf.columns)})"
            )

    lon = gdf.loc[null_mask, lon_col]
    lat = gdf.loc[null_mask, lat_col]
    unusable = lon.isna() | lat.isna()
    if unusable.any():
        raise ValueError(
            f"{int(unusable.sum())} of {int(null_mask.sum())} NULL-geometry "
            f"features are also missing {lon_col!r}/{lat_col!r}; "
            "null_geometry: point_from cannot synthesize a position for them."
        )

    import geopandas as gpd

    out = gdf.copy()
    out.loc[null_mask, "geometry"] = gpd.points_from_xy(lon, lat, crs=out.crs)
    _flag_null_geometry_rows(gdf, out, null_mask)
    return out


def _apply_null_geometry_policy(
    gdf, opts: FlatGeobufParams
) -> tuple[Any, int, int, int]:
    """Apply ``opts.null_geometry`` to ``gdf``.

    Returns ``(gdf_to_write, features_dropped, features_sentinel,
    features_synthesized)``.

    A prior-database delivery — SWOT LakeSP Prior is the product that surfaced
    this (#98), at 73% of features — reports every prior feature in a pass's
    footprint and leaves the geometry NULL for the ones not observed. FlatGeobuf's
    packed Hilbert R-tree cannot index a NULL geometry, so with the index on (the
    default; an unindexed file is not the candidate) that is a hard format
    constraint, not a bug in the writer:

    * ``"drop"`` writes the non-null subset, regardless of ``spatial_index``.
    * ``"point_from"`` keeps every feature behind a *real* geometry built from
      named lon/lat columns (see :func:`_substitute_point_from_columns`) — the
      right choice whenever the source carries a usable position for its
      NULL-geometry rows.
    * ``"sentinel"`` keeps every feature behind a placeholder geometry (see
      :func:`_substitute_sentinel_geometries`) — no row is dropped, but the
      placeholder bytes are fabricated, not part of the source, so the caller
      records how many for the result to say so. Only for a source with no
      usable position at all; prefer ``"point_from"`` when there is one.
    * ``"error"`` (the default) only raises when the index would actually fail
      to write — an unindexed file holds a NULL geometry without complaint —
      and raises *before* any bytes are written: a GDAL driver message
      partway into a cluster-dispatched write is not something whoever runs
      the campaign can triage.
    """
    null_mask = gdf.geometry.isna()
    null_count = int(null_mask.sum())
    if null_count == 0:
        return gdf, 0, 0, 0

    total = len(gdf)
    if null_count == total:
        raise ValueError(
            f"every one of {total} features has a NULL geometry; there is no "
            "real geometry to index, drop around, or infer a placeholder type from."
        )

    if opts.null_geometry == "drop":
        return gdf.loc[~null_mask].reset_index(drop=True), null_count, 0, 0

    if opts.null_geometry == "point_from":
        if not opts.null_geometry_point_from:
            raise ValueError(
                "null_geometry: point_from requires "
                "null_geometry_point_from: [lon_column, lat_column]"
            )
        lon_col, lat_col = opts.null_geometry_point_from
        out = _substitute_point_from_columns(gdf, null_mask, lon_col, lat_col)
        return out, 0, 0, null_count

    if opts.null_geometry == "sentinel":
        return _substitute_sentinel_geometries(gdf, null_mask), 0, null_count, 0

    if opts.spatial_index:
        share = null_count / total
        raise ValueError(
            f"{null_count} of {total} features ({share:.0%}) have a NULL "
            "geometry; FlatGeobuf's packed Hilbert R-tree cannot index a NULL "
            "geometry. Set params.null_geometry: drop to write the non-null "
            "subset (recorded as a content subset on the layout), "
            "params.null_geometry: point_from (+ null_geometry_point_from: "
            "[lon_col, lat_col]) to keep every feature behind a real position "
            "if the source carries one, params.null_geometry: sentinel to keep "
            "every feature behind a placeholder geometry otherwise (recorded "
            "on the layout, and excluded from spatial queries by "
            "construction), or params.spatial_index: false to keep every "
            "feature without the index."
        )
    return gdf, 0, 0, 0


def _write_flatgeobuf(gdf, target: str, *, spatial_index: bool = True) -> None:
    """Write ``gdf`` to ``target`` as a FlatGeobuf file.

    ``spatial_index`` writes the packed Hilbert R-tree; the driver then also sorts
    the features by their Hilbert index, so a bbox query reads few, contiguous
    ranges. Uses ``pyogrio`` directly (its bundled GDAL carries the FlatGeobuf
    driver) rather than geopandas' engine selection, so the writer is explicit.
    """
    import pyogrio

    pyogrio.write_dataframe(
        gdf,
        target,
        driver="FlatGeobuf",
        SPATIAL_INDEX="YES" if spatial_index else "NO",
    )


@FORMATS.register("flatgeobuf")
class FlatGeobufAdapter(FormatAdapter):
    name = "flatgeobuf"
    object_kind = ObjectKind.VECTOR_FILE

    def __init__(self) -> None:
        # A produced file cannot say how many features were excluded,
        # sentinel-substituted or position-synthesized — that accounting only
        # exists at write time — so :meth:`convert` records it here, keyed by
        # target path, for :meth:`describe_layout` to pick up right after
        # (#98). The runner always calls the two in sequence for the same
        # object before converting anything else through this instance.
        self._null_geometry_stats_by_target: dict[str, tuple[int, int, int]] = {}

    def target_basename(self) -> str:
        return "flatgeobuf.fgb"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (an OGR-readable vector) to a FlatGeobuf file.

        Reads every feature of ``source`` into a GeoDataFrame (a ``/vsizip`` member
        is read in place by the OGR driver), applies the ``null_geometry`` policy
        (#98), then writes it with the spatial-index lever from
        :class:`FlatGeobufParams`.
        """
        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "FlatGeobuf conversion requires the 'flatgeobuf' extra; install "
                "with `uv sync --extra flatgeobuf` "
                "(or `pip install cng-benchmark[flatgeobuf]`)"
            ) from exc

        opts = FlatGeobufParams.model_validate(params)
        gdf = gpd.read_file(source)
        gdf, dropped, sentinel, synthesized = _apply_null_geometry_policy(gdf, opts)
        self._null_geometry_stats_by_target[target] = (dropped, sentinel, synthesized)
        _write_flatgeobuf(gdf, target, spatial_index=opts.spatial_index)

    def describe_grouping_lever(self) -> str:
        return "FlatGeobuf packed Hilbert R-tree index (node size 16, fixed)"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the size (bytes) of the produced FlatGeobuf file — one object."""
        return [os.path.getsize(target)]

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[FlatGeobufLayout]:
        """Return the produced file's spatial-index layout (one object)."""
        dropped, sentinel, synthesized = self._null_geometry_stats_by_target.get(
            target, (0, 0, 0)
        )
        return [
            describe_flatgeobuf_layout(
                target,
                name or self.name,
                features_dropped=dropped,
                features_sentinel=sentinel,
                features_synthesized=synthesized,
            )
        ]
