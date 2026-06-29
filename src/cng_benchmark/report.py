"""Result artifacts: the JSON record and a human-readable Markdown summary.

A deployed run produces two artifacts under its configured output location: the
machine-readable ``result.json`` (the full :class:`~cng_benchmark.models.BenchmarkRun`)
and a compact ``summary.md`` for humans skimming a results bucket. Rendering is
pure and stdlib-only, so it is fully unit-testable; persistence is delegated to
:mod:`cng_benchmark.storage`, which handles both local paths and S3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cng_benchmark import storage
from cng_benchmark.models import BenchmarkRun

if TYPE_CHECKING:
    from cng_benchmark.runner import ProductSetResult

RESULT_FILENAME = "result.json"
SUMMARY_FILENAME = "summary.md"


def _format_bytes(n: float) -> str:
    """Render a byte count with a binary unit suffix (KiB, MiB, …)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} {units[-1]}"  # pragma: no cover - unreachable


def render_markdown_summary(run: BenchmarkRun) -> str:
    """Render a compact Markdown summary of a :class:`BenchmarkRun`."""
    lines: list[str] = [
        f"# Benchmark result: {run.dataset_id} → {run.format_id}",
        "",
        f"- **Timestamp:** {run.timestamp.isoformat()}",
        f"- **Dataset:** `{run.dataset_id}`",
        f"- **Format:** `{run.format_id}`",
    ]
    versions = ", ".join(f"{k} {v}" for k, v in sorted(run.tool_versions.items()))
    if versions:
        lines.append(f"- **Tool versions:** {versions}")

    profile = run.object_profile
    if profile is not None:
        lines += [
            "",
            "## Object-size profile",
            "",
            f"- **Objects:** {profile.count}",
            f"- **Total:** {_format_bytes(profile.total_bytes)}",
            f"- **Mean:** {_format_bytes(profile.mean)}",
            f"- **Median / p90 / p99:** {_format_bytes(profile.median)} / "
            f"{_format_bytes(profile.p90)} / {_format_bytes(profile.p99)}",
            f"- **Min / max:** {_format_bytes(profile.min_bytes)} / "
            f"{_format_bytes(profile.max_bytes)}",
            f"- **Tier fit:** {', '.join(profile.tier_fit) or 'none'}"
            f" (highest: {profile.highest_tier or 'none'})",
        ]

    if run.object_layouts:
        lines += _render_object_layouts(run.object_layouts)

    if run.metrics:
        lines += ["", "## Metrics", ""]
        lines += ["| Metric | Value | Unit |", "| --- | --- | --- |"]
        for m in run.metrics:
            lines.append(f"| {m.name} | {m.value:g} | {m.unit or ''} |")

    lines += _render_artifacts(run.artifacts)

    lines.append("")
    return "\n".join(lines)


def _render_artifacts(artifacts: list) -> list[str]:
    """Render the run's non-metric side-outputs (chunk-layout/LOD PNGs, VRTs).

    A produced artifact shows its store URI (plus a ready-to-paste TiTiler viewer
    URL for a composite VRT); a skipped one shows why. Empty when there are none.
    """
    if not artifacts:
        return []
    lines = ["", "## Artifacts", ""]
    for a in artifacts:
        if a.uri:
            line = f"- **{a.name}** (`{a.kind}`): `{a.uri}`"
            titiler = a.detail.get("titiler_url")
            if titiler:
                line += f" — TiTiler viewer: `{titiler}`"
            lines.append(line)
        else:
            reason = a.detail.get("skipped_reason", "skipped")
            lines.append(f"- **{a.name}** (`{a.kind}`): skipped — {reason}")
    return lines


def _render_object_layouts(layouts: list) -> list[str]:
    """Render the per-object partial-access layout, one table per format kind.

    Each format answers "can a client fetch part without the whole" through its own
    structure, so COG objects get a "Tiling layout" table (block size, overviews)
    and GeoZarr arrays a "Chunk/shard layout" table (chunk, shard, codec,
    multiscale levels) — the structural side of the comparison, beside the sizes.
    """
    cog = [ly for ly in layouts if ly.kind == "cog"]
    geozarr = [ly for ly in layouts if ly.kind == "geozarr"]
    geoparquet = [ly for ly in layouts if ly.kind == "geoparquet"]
    copc = [ly for ly in layouts if ly.kind == "copc"]
    lines: list[str] = []
    if cog:
        lines += _render_tiling_layout(cog)
    if geozarr:
        lines += _render_chunk_shard_layout(geozarr)
    if geoparquet:
        lines += _render_row_group_layout(geoparquet)
    if copc:
        lines += _render_octree_layout(copc)
    return lines


def _render_tiling_layout(layouts: list) -> list[str]:
    """Render the COG per-object tiling layout: a coverage line plus a table.

    Leads with how many objects are internally tiled (range-read friendly) vs
    striped, then one row per object (block size, overview levels, internal
    tiles) — the structural side of the partial-access story, beside the sizes.
    """
    tiled = sum(1 for ly in layouts if ly.is_tiled)
    lines = [
        "",
        "## Tiling layout",
        "",
        f"- **Internally tiled:** {tiled}/{len(layouts)} objects "
        f"({len(layouts) - tiled} striped)",
        "",
        "| Object | Tiled | Block | Overviews | Internal tiles |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ovr = len(ly.overview_decimations)
        lines.append(
            f"| {ly.name} | {'yes' if ly.is_tiled else 'no'} | "
            f"{ly.block_width}×{ly.block_height} | {ovr} | {ly.internal_tiles} |"
        )
    return lines


def _render_chunk_shard_layout(layouts: list) -> list[str]:
    """Render the GeoZarr per-array chunk/shard layout: a coverage line plus a table.

    Leads with the total shard-object count (the stored, tier-judged objects), then
    one row per array (chunk = addressable unit, shard = stored object,
    chunks/shard, codec, multiscale levels) — the GeoZarr answer to the same
    partial-access question COG answers with internal tiling.
    """
    shards = sum(ly.shard_count for ly in layouts)
    lines = [
        "",
        "## Chunk/shard layout",
        "",
        f"- **Shard objects:** {shards} across {len(layouts)} array(s)",
        "",
        "| Array | Chunk | Shard | Chunks/shard | Codec | Levels | Shards |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        chunk = "×".join(str(v) for v in ly.chunk_shape)
        shard = "×".join(str(v) for v in ly.shard_shape)
        lines.append(
            f"| {ly.name} | {chunk} | {shard} | {ly.chunks_per_shard} | "
            f"{ly.codec} | {ly.multiscale_levels} | {ly.shard_count} |"
        )
    return lines


def _render_row_group_layout(layouts: list) -> list[str]:
    """Render the GeoParquet per-file row-group layout: a coverage line plus a table.

    Leads with whether the bbox covering is present (whether a bbox query can push
    down to row groups at all), then one row per file (geometry column, feature
    count, row groups, rows/group) — the GeoParquet answer to the same
    partial-access question COG answers with internal tiling.
    """
    with_bbox = sum(1 for ly in layouts if ly.has_bbox_covering)
    lines = [
        "",
        "## Row-group layout",
        "",
        f"- **Bbox covering:** {with_bbox}/{len(layouts)} file(s) "
        "(spatial pushdown to row groups)",
        "",
        "| Object | Geometry | Features | Row groups | Rows/group | Bbox covering |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        lines.append(
            f"| {ly.name} | {ly.geometry_column} | {ly.num_rows} | "
            f"{ly.num_row_groups} | {ly.row_group_rows} | "
            f"{'yes' if ly.has_bbox_covering else 'no'} |"
        )
    return lines


def _render_octree_layout(layouts: list) -> list[str]:
    """Render the COPC per-file octree layout: a coverage line plus a table.

    Leads with the total octree-node count (the range-addressable units), then one
    row per file (node count, octree depth, total points, largest node) — the COPC
    answer to the same partial-access question COG answers with internal tiling.
    """
    nodes = sum(ly.num_nodes for ly in layouts)
    carried = sorted({d for ly in layouts for d in ly.extra_dimensions})
    ratios = [ly.compression_ratio for ly in layouts if ly.compression_ratio]
    lines = [
        "",
        "## Octree layout",
        "",
        f"- **Octree nodes:** {nodes} across {len(layouts)} file(s)",
        f"- **Carried point variables:** {len(carried)} extra dimension(s)"
        + (f" ({', '.join(carried)})" if carried else " — geometry only"),
    ]
    if ratios:
        lines.append(
            "- **LASzip compression:** "
            + ", ".join(f"{r:.2f}×" for r in ratios)
            + " (uncompressed LAS point block / stored size)"
        )
    lines += [
        "",
        "| Object | Nodes | Depth | Points | Points/node | Extra dims | Compression |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        lines.append(
            f"| {ly.name} | {ly.num_nodes} | {ly.max_depth} | "
            f"{ly.point_count} | {ly.points_per_node} | "
            f"{len(ly.extra_dimensions)} | {ratio} |"
        )
    return lines


def write_artifacts(run: BenchmarkRun, output_uri: str) -> dict[str, str]:
    """Write ``result.json`` and ``summary.md`` under ``output_uri``.

    ``output_uri`` is treated as a directory/prefix (local path, ``file://`` or
    ``s3://bucket/prefix``). Returns the URIs of the artifacts written.
    """
    result_uri = storage.join(output_uri, RESULT_FILENAME)
    summary_uri = storage.join(output_uri, SUMMARY_FILENAME)
    storage.write_text(result_uri, run.model_dump_json(indent=2))
    storage.write_text(summary_uri, render_markdown_summary(run))
    return {"result": result_uri, "summary": summary_uri}


def render_product_set_summary(result: ProductSetResult) -> str:
    """Render a top-level summary: a per-product table plus the roll-up.

    One row per product (object count, total, mean, tier fit) followed by the
    pooled roll-up row, so a reader skimming a run sees the per-scene
    distribution and the honest set-level distribution at a glance.
    """
    lines = [
        f"# Benchmark result: {result.rollup.dataset_id} → {result.rollup.format_id}"
        " (product set)",
        "",
        f"- **Timestamp:** {result.rollup.timestamp.isoformat()}",
        f"- **Products:** {len(result.per_product)}",
        "",
        "## Per-product object-size profiles",
        "",
        "| Product | Objects | Total | Mean | Highest tier | Layout |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def _layout(run) -> str:
        layouts = run.object_layouts
        if not layouts:
            return "-"
        # A format-agnostic structural digest: COG reports its range-read-friendly
        # (tiled) fraction; GeoZarr reports its shard-object count; GeoParquet, the
        # row-group count (its addressable units) — each format's own answer to the
        # partial-access question.
        cog = [ly for ly in layouts if ly.kind == "cog"]
        if cog:
            return f"{sum(1 for ly in cog if ly.is_tiled)}/{len(cog)} tiled"
        geoparquet = [ly for ly in layouts if ly.kind == "geoparquet"]
        if geoparquet:
            groups = sum(ly.num_row_groups for ly in geoparquet)
            return f"{groups} row groups"
        copc = [ly for ly in layouts if ly.kind == "copc"]
        if copc:
            return f"{sum(ly.num_nodes for ly in copc)} octree nodes"
        shards = sum(getattr(ly, "shard_count", 0) for ly in layouts)
        return f"{shards} shards"

    for run in result.per_product:
        p = run.object_profile
        product_id = run.params.get("product_id", run.dataset_id)
        if p is None:  # pragma: no cover - profile always present here
            lines.append(f"| {product_id} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {product_id} | {p.count} | {_format_bytes(p.total_bytes)} | "
            f"{_format_bytes(p.mean)} | {p.highest_tier or 'none'} | {_layout(run)} |"
        )
    roll = result.rollup.object_profile
    if roll is not None:
        lines.append(
            f"| **roll-up** | **{roll.count}** | **{_format_bytes(roll.total_bytes)}** "
            f"| **{_format_bytes(roll.mean)}** | **{roll.highest_tier or 'none'}** "
            f"| **{_layout(result.rollup)}** |"
        )
    lines += _render_artifacts(result.rollup.artifacts)
    lines.append("")
    return "\n".join(lines)


def write_product_set_artifacts(
    result: ProductSetResult, output_uri: str
) -> dict[str, str]:
    """Write the product-set run tree under ``output_uri``.

    Lays out ``product/<id>/{result.json,summary.md}`` per scene,
    ``rollup/{result.json,summary.md}`` for the pooled distribution, and a
    top-level ``summary.md`` (per-product table + roll-up). Returns a map of the
    artifact URIs written.
    """
    written: dict[str, str] = {}
    for run in result.per_product:
        product_id = run.params.get("product_id", run.dataset_id)
        product_dir = storage.join(storage.join(output_uri, "product"), str(product_id))
        paths = write_artifacts(run, product_dir)
        written[f"product/{product_id}/result"] = paths["result"]
        written[f"product/{product_id}/summary"] = paths["summary"]

    rollup_paths = write_artifacts(result.rollup, storage.join(output_uri, "rollup"))
    written["rollup/result"] = rollup_paths["result"]
    written["rollup/summary"] = rollup_paths["summary"]

    summary_uri = storage.join(output_uri, SUMMARY_FILENAME)
    storage.write_text(summary_uri, render_product_set_summary(result))
    written["summary"] = summary_uri
    return written
