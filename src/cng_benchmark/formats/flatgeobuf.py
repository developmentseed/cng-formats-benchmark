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
from typing import Any, NamedTuple

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
    """

    model_config = ConfigDict(extra="ignore")

    spatial_index: bool = True


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


def describe_flatgeobuf_layout(path: str, name: str) -> FlatGeobufLayout:
    """Return the :class:`FlatGeobufLayout` of the FlatGeobuf file at ``path``.

    Splits the stored object into its three parts — header, R-tree, feature data —
    from the header alone, so what the index costs is answerable next to the size
    it costs it in. ``compression_ratio`` is 1.0: FlatGeobuf defines no
    compression, so the stored bytes *are* the uncompressed bytes.
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
    )


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

    def target_basename(self) -> str:
        return "flatgeobuf.fgb"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (an OGR-readable vector) to a FlatGeobuf file.

        Reads every feature of ``source`` into a GeoDataFrame (a ``/vsizip`` member
        is read in place by the OGR driver), then writes it with the spatial-index
        lever from :class:`FlatGeobufParams`.
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
        return [describe_flatgeobuf_layout(target, name or self.name)]
