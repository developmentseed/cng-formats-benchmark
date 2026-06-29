"""Result schema for a benchmark run.

These models are the system's serialisable output: a benchmark run produces a
:class:`BenchmarkRun` capturing *what was measured, against what, and when*, so
results from different datasets, formats, and tool versions remain comparable
over time. They are deliberately free of any service or IO dependency — the
harness assembles them, and the deployment (M2) persists them.

The headline payload is the :class:`ObjectSizeProfile`. Object size is a hard
constraint on tiered object stores (see :mod:`cng_benchmark.tiers`), so the
profile is a first-class result rather than an incidental statistic.

Non-metric side-outputs (display chunk-layout PNGs, COPC octree level-of-detail
PNGs, RGB composite VRTs for manual TiTiler inspection) are represented by the
:class:`Artifact` type and accumulated on :attr:`BenchmarkRun.artifacts`, rather
than being smuggled through :class:`MetricResult.detail`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class HistogramBin(BaseModel):
    """A single half-open ``[lower, upper)`` object-size histogram bucket."""

    lower: int
    upper: int
    count: int


class ObjectSizeProfile(BaseModel):
    """Summary of the sizes of the objects a format layout produces.

    Percentiles are reported in bytes. ``p50`` is identical to ``median`` and
    kept alongside the other percentiles for convenience. Tier fitness is
    derived from the *mean* object size against a configured policy: ``tier_fit``
    lists every tier the layout satisfies and ``highest_tier`` is the coldest of
    those (or ``None`` if the objects are too small for any tier).
    """

    count: int
    total_bytes: int
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    min_bytes: int
    max_bytes: int
    histogram: list[HistogramBin]
    tier_fit: list[str]
    highest_tier: str | None


class Artifact(BaseModel):
    """A non-metric file a run produced alongside its measurements.

    A structural or viewer side-output addressed on the store — the COG/Zarr
    display chunk-layout PNG, the COPC octree level-of-detail PNG, or an RGB
    composite VRT for manual TiTiler inspection. ``kind`` groups them, ``uri``
    locates the produced file (``None`` when generation was skipped, reason in
    ``detail``), ``media_type`` is its content type, and ``detail`` carries
    kind-specific extras (e.g. a ready-to-paste TiTiler viewer URL + rescale).
    """

    kind: str  # "chunk_layout" | "octree_lod" | "viewer_vrt"
    name: str  # label, e.g. "natural", "display_chunk_layout"
    uri: str | None = None
    media_type: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """A single named measurement produced by a metric collector."""

    name: str
    value: float
    unit: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class ObjectLayout(BaseModel):
    """Base record of one produced object's partial-access layout.

    Whether bytes are reachable by partial (range) reads is decided by an
    object's *internal* structure, not only its size, and every candidate format
    answers that "can a client fetch part without the whole" question through its
    own structure (COG internal blocks + overviews, Zarr v3 chunks + shards +
    multiscales). This base carries what every format shares — the object's
    ``name`` and ``size_bytes`` — and ``kind`` discriminates the typed per-format
    subclass below, so the structural side of the comparison is in the result
    itself, uniformly, independent of any tile server.
    """

    kind: str
    name: str
    size_bytes: int


class CogLayout(ObjectLayout):
    """A COG's internal tiling layout.

    ``is_tiled`` is true when the block does not span the full raster width (a
    tiled COG, range-read friendly) rather than striped; ``internal_tiles`` is the
    full-resolution block-grid cell count, and ``overview_decimations`` the
    overview levels. The chunk-aware display metric reads the same structure to
    bucket its tiles.
    """

    kind: Literal["cog"] = "cog"
    is_tiled: bool
    block_height: int
    block_width: int
    overview_decimations: list[int] = Field(default_factory=list)
    internal_tiles: int
    codec: str = "none"
    compression_ratio: float = 0.0


class GeoZarrLayout(ObjectLayout):
    """A GeoZarr v3 array's chunk/shard layout.

    The addressable unit is the chunk; the stored object is the shard (one shard
    packs ``chunks_per_shard`` chunks), which is the lever that lifts the mean
    object size into a storage tier. ``multiscale_levels`` is the overview-pyramid
    depth (the GeoZarr analogue of COG overviews) and ``shard_count`` the number of
    shard objects the array produced.
    """

    kind: Literal["geozarr"] = "geozarr"
    chunk_shape: list[int]
    shard_shape: list[int]
    chunks_per_shard: int
    codec: str
    multiscale_levels: int
    shard_count: int
    compression_ratio: float = 0.0


class GeoParquetLayout(ObjectLayout):
    """A GeoParquet file's row-group layout.

    The addressable unit is the **row group**: a reader with a bbox predicate
    fetches only the row groups whose covering bbox overlaps the query, so the row
    group is the vector analogue of a COG block or a Zarr chunk. ``row_group_rows``
    is the largest group's row count (the configured grouping lever),
    ``num_row_groups`` how many the file holds, and ``has_bbox_covering`` whether
    the GeoParquet 1.1 bbox covering column is present — the structure that lets a
    bbox query push down to row groups rather than scan the whole file.
    """

    kind: Literal["geoparquet"] = "geoparquet"
    geometry_column: str
    num_rows: int
    num_row_groups: int
    row_group_rows: int
    has_bbox_covering: bool
    codec: str = "uncompressed"
    compression_ratio: float = 0.0


class CopcLayout(ObjectLayout):
    """A COPC file's octree-node layout.

    The addressable unit is the **octree node**: a reader with a bbox/resolution
    predicate fetches only the nodes that overlap, so the node is the point-cloud
    analogue of a COG block or a Zarr chunk. ``num_nodes`` is how many nodes the
    octree holds, ``max_depth`` its depth (the octree-depth lever), ``point_count``
    the total points, and ``points_per_node`` the largest node's point count — the
    realised per-node point budget (the span lever). A COPC is a single stored
    object; its size already clears the cold tiers, so the lever is about
    preserving range-addressable partial access, not reaching a size floor.
    """

    kind: Literal["copc"] = "copc"
    num_nodes: int
    max_depth: int
    point_count: int
    points_per_node: int
    extra_dimensions: list[str] = Field(default_factory=list)
    codec: str = "laszip"
    #: LASzip compression ratio: the uncompressed LAS point block (point count ×
    #: record length, geometry + carried extra dims) over the stored file size.
    #: Quantifies how much of the size is the format's compression vs the content.
    compression_ratio: float = 0.0


#: Discriminated union over the per-format layouts, so a ``BenchmarkRun`` keeps
#: each layout's subclass fields through pydantic validation and JSON round-trips.
AnyObjectLayout = Annotated[
    CogLayout | GeoZarrLayout | GeoParquetLayout | CopcLayout,
    Field(discriminator="kind"),
]


class BenchmarkRun(BaseModel):
    """The full, serialisable record of one benchmark run.

    Captures the run context needed to interpret and compare results: when it
    ran, the versions of the tools involved, and which dataset/format/params
    were exercised. ``object_profile`` carries the object-size differentiator,
    ``metrics`` holds additional scalar measurements, and ``artifacts`` holds the
    non-metric side-outputs (chunk-layout PNGs, octree LOD, RGB VRTs) produced
    alongside the measurements.
    """

    timestamp: datetime
    tool_versions: dict[str, str] = Field(default_factory=dict)
    dataset_id: str
    format_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    object_profile: ObjectSizeProfile | None = None
    object_layouts: list[AnyObjectLayout] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
