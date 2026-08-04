"""GeoParquet adapter — vector features to a single GeoParquet file.

The grouping lever for GeoParquet is its **row-group size** plus **spatial
partitioning**: the row group is the addressable unit a bbox query fetches, so
how many features land in a group (``row_group_rows``) and whether features are
spatially ordered first (so each group's covering bbox is tight) together decide
how few row groups a spatial query must read. This adapter writes one GeoParquet
file per component — a single ``VECTOR_FILE`` object, the vector analogue of the
COG arm — and flows through the same runner paths as COG, with the read metric a
bbox/row-group spatial query (:func:`cng_benchmark.metrics.read.measure_vector_read`)
rather than a raster window. There is no display surface (a vector table is not a
TiTiler raster tile).

The write core (:func:`_write_geoparquet`, :func:`enumerate_objects`,
:func:`describe_geoparquet_layout`) depends only on ``geopandas`` + ``pyarrow`` +
``shapely``, so the lever / enumerate / layout logic is unit-testable on a
synthetic in-memory GeoDataFrame. Only :meth:`GeoParquetAdapter.convert`'s source
read needs an OGR driver (``pyogrio``, the ``geoparquet`` extra), imported lazily.

The per-column codec (``params['compression']``) defaults to ``zstd``, not
geopandas'/pyarrow's own ``snappy`` default — the CNES D2 review's leading
hypothesis for LakeSP's GeoParquet coming out **larger** than the zipped
shapefile source was the row-group codec (#73). ``compression_level`` and
``data_page_size`` round out the sweep once the codec name itself is fixed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import GeoParquetLayout
from cng_benchmark.registry import FORMATS

#: Default rows per row group when the config carries no lever value. A row group
#: is the addressable unit, so this is the GeoParquet analogue of COG's block size.
DEFAULT_ROW_GROUP_ROWS = 50_000


class GeoParquetParams(BaseModel):
    """GeoParquet grouping levers, parsed from ``config.params``.

    ``row_group_rows`` sets the rows per row group (the addressable unit);
    it tolerates a swept *list* of sizes, taking the first so a swept lever
    degrades to a single run, mirroring COG's ``block_size``.
    ``spatial_partitioning`` (default on) spatially orders the features before
    writing so each row group's covering bbox is compact — the lever that makes a
    bbox query touch fewer groups. ``compression`` is the per-column codec
    (``zstd`` default — geopandas/pyarrow's own default is ``snappy``, the codec
    the CNES D2 review's +14%-vs-shapefile hypothesis pinned as the culprit
    (#73); ``compression_level`` and ``data_page_size`` are the finer knobs
    (codec effort, and the parquet-internal page granularity within a row
    group) a codec sweep can tune once the row-group lever is fixed.
    """

    model_config = ConfigDict(extra="ignore")

    row_group_rows: Any = None
    spatial_partitioning: bool = True
    compression: str = "zstd"
    compression_level: int | None = None
    data_page_size: int | None = None


def _row_group_rows(value: Any, default: int = DEFAULT_ROW_GROUP_ROWS) -> int:
    """Normalise the ``row_group_rows`` lever to a positive int.

    Tolerates ``None``/empty (falls back to ``default``) and a swept *list* of
    sizes (takes the first), like COG's ``block_size``.
    """
    if value is None or value == []:
        return default
    if isinstance(value, list | tuple):
        value = value[0]
    return max(1, int(value))


def _spatial_sort(gdf):
    """Order features along a Hilbert curve so neighbouring rows are near in space.

    Spatially compact row groups have tight covering bboxes, so a bbox query
    overlaps fewer of them — this is the "spatial partitioning" half of the lever.
    Falls back to a lower-left-corner sort if Hilbert ordering is unavailable
    (e.g. an empty geometry extent).
    """
    try:
        order = gdf.geometry.hilbert_distance()
    except Exception:  # noqa: BLE001 - degrade to a coarse bounds sort
        bounds = gdf.geometry.bounds
        order = bounds["miny"].rank(method="first") * (len(gdf) + 1) + bounds[
            "minx"
        ].rank(method="first")
    return gdf.iloc[order.argsort(kind="stable").to_numpy()].reset_index(drop=True)


def _write_geoparquet(
    gdf,
    target: str,
    *,
    row_group_rows: int,
    spatial_partitioning: bool,
    compression: str = "zstd",
    compression_level: int | None = None,
    data_page_size: int | None = None,
) -> None:
    """Write ``gdf`` to ``target`` as a GeoParquet file with the grouping lever.

    Features are (optionally) spatially sorted, then written with the configured
    ``row_group_rows`` per row group and a GeoParquet 1.1 covering-bbox column
    (``write_covering_bbox=True``) so a reader can push a bbox predicate down to
    the overlapping row groups. ``compression_level`` and ``data_page_size`` are
    forwarded to pyarrow's writer as-is (``None`` uses the codec's / pyarrow's
    own default) — the finer half of the codec-sweep lever (#73), on top of the
    codec name itself. Pure ``geopandas`` + ``pyarrow`` — no OGR — so it is
    CI-testable on a synthetic GeoDataFrame.
    """
    if spatial_partitioning and len(gdf) > 1:
        gdf = _spatial_sort(gdf)
    kwargs: dict[str, Any] = {}
    if compression_level is not None:
        kwargs["compression_level"] = compression_level
    if data_page_size is not None:
        kwargs["data_page_size"] = data_page_size
    gdf.to_parquet(
        target,
        index=False,
        compression=compression,
        row_group_size=row_group_rows,
        write_covering_bbox=True,
        **kwargs,
    )


def _geo_metadata(path: str) -> dict:
    """Return the GeoParquet ``geo`` file-metadata mapping (empty if absent)."""
    import pyarrow.parquet as pq

    raw = pq.read_metadata(path).metadata or {}
    blob = raw.get(b"geo")
    return json.loads(blob) if blob else {}


def describe_geoparquet_layout(path: str, name: str) -> GeoParquetLayout:
    """Return the :class:`GeoParquetLayout` of the GeoParquet file at ``path``.

    Reads the parquet footer (cheap, no feature scan): the row count, the row
    groups and their largest member (the effective ``row_group_rows`` lever), and
    — from the GeoParquet ``geo`` metadata — the primary geometry column and
    whether a bbox covering column is present for spatial pushdown.
    """
    import pyarrow.parquet as pq

    md = pq.read_metadata(path)
    row_group_rows = max(
        (md.row_group(i).num_rows for i in range(md.num_row_groups)), default=0
    )
    geo = _geo_metadata(path)
    primary = geo.get("primary_column", "geometry")
    has_bbox = bool(geo.get("columns", {}).get(primary, {}).get("covering"))

    codec = "uncompressed"
    total_uncompressed = 0
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            col = rg.column(j)
            if i == 0 and j == 0:
                codec = col.compression.lower()
            total_uncompressed += col.total_uncompressed_size
    file_size = os.path.getsize(path)
    compression_ratio = total_uncompressed / file_size if file_size else 0.0

    return GeoParquetLayout(
        name=name,
        size_bytes=file_size,
        geometry_column=primary,
        num_rows=md.num_rows,
        num_row_groups=md.num_row_groups,
        row_group_rows=int(row_group_rows),
        has_bbox_covering=has_bbox,
        codec=codec,
        compression_ratio=compression_ratio,
    )


@FORMATS.register("geoparquet")
class GeoParquetAdapter(FormatAdapter):
    name = "geoparquet"
    object_kind = ObjectKind.VECTOR_FILE

    def target_basename(self) -> str:
        return "geoparquet.parquet"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (an OGR-readable vector) to a GeoParquet file.

        Reads every feature of ``source`` into a GeoDataFrame (a ``/vsizip``
        member is read in place by the OGR driver), then writes it with the
        row-group-size and spatial-partitioning lever from :class:`GeoParquetParams`.
        """
        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "GeoParquet conversion requires the 'geoparquet' extra; install "
                "with `uv sync --extra geoparquet` "
                "(or `pip install cng-benchmark[geoparquet]`)"
            ) from exc

        opts = GeoParquetParams.model_validate(params)
        gdf = gpd.read_file(source)
        _write_geoparquet(
            gdf,
            target,
            row_group_rows=_row_group_rows(opts.row_group_rows),
            spatial_partitioning=opts.spatial_partitioning,
            compression=opts.compression,
            compression_level=opts.compression_level,
            data_page_size=opts.data_page_size,
        )

    def describe_grouping_lever(self) -> str:
        return "GeoParquet row-group size and spatial partitioning"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the size (bytes) of the produced GeoParquet file — one object."""
        return [os.path.getsize(target)]

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[GeoParquetLayout]:
        """Return the produced file's row-group layout (one object)."""
        return [describe_geoparquet_layout(target, name or self.name)]
