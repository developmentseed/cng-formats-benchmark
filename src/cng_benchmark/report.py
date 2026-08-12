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

    lines += _render_conditions(run.conditions)
    lines += _render_artifacts(run.artifacts)

    lines.append("")
    return "\n".join(lines)


def _render_conditions(conditions: list) -> list[str]:
    """Render the run-protocol condition matrix (issue #87).

    One row per ``(phase, cache, concurrency, metric)`` — mean, median and
    spread across the condition's replicates — the "cold isolated / warm
    isolated / cold concurrent" table the study wants per format. Empty when
    the run carries no ``run_protocol`` (the common case).
    """
    if not conditions:
        return []
    lines = [
        "",
        "## Run protocol",
        "",
        "| Phase | Cache | Concurrency | Replicates | Metric | Mean | Median"
        " | Spread | Unit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in conditions:
        isolation = (
            "isolated" if c.concurrency == 1 else f"concurrent ({c.concurrency})"
        )
        n = len(c.replicates)
        for m in c.aggregate:
            med = m.detail.get("median")
            spread = m.detail.get("stdev")
            med_str = f"{med:g}" if isinstance(med, int | float) else "—"
            spread_str = f"{spread:g}" if isinstance(spread, int | float) else "—"
            lines.append(
                f"| {c.phase} | {c.cache} | {isolation} | {n} | {m.name} | "
                f"{m.value:g} | {med_str} | {spread_str} | {m.unit or ''} |"
            )
    return lines


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
    flatgeobuf = [ly for ly in layouts if ly.kind == "flatgeobuf"]
    copc = [ly for ly in layouts if ly.kind == "copc"]
    lines: list[str] = []
    if cog:
        lines += _render_tiling_layout(cog)
    if geozarr:
        lines += _render_chunk_shard_layout(geozarr)
    if geoparquet:
        lines += _render_row_group_layout(geoparquet)
    if flatgeobuf:
        lines += _render_spatial_index_layout(flatgeobuf)
    if copc:
        lines += _render_octree_layout(copc)
    return lines


def _render_tiling_layout(layouts: list) -> list[str]:
    """Render the COG per-object tiling layout: a coverage line plus a table.

    Leads with how many objects are internally tiled (range-read friendly) vs
    striped, then one row per object (block size, overview levels, internal
    tiles) — the structural side of the partial-access story, beside the sizes.

    A bundled multi-band file (several components stacked into one COG, #102)
    carries ``band_names`` — flagged with a coverage line and a "Bands" column,
    so a table row for one object can still say which components it holds.
    """
    tiled = sum(1 for ly in layouts if ly.is_tiled)
    bundled = [ly for ly in layouts if ly.band_names]
    lines = [
        "",
        "## Tiling layout",
        "",
        f"- **Internally tiled:** {tiled}/{len(layouts)} objects "
        f"({len(layouts) - tiled} striped)",
    ]
    if bundled:
        lines.append(
            f"- **Bundled:** {len(bundled)} file(s) hold more than one "
            "component, one band each"
        )
    lines += [
        "",
        "| Object | Tiled | Block | Overviews | Internal tiles | Codec"
        " | Compression | Bands |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ovr = len(ly.overview_decimations)
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        bands = ", ".join(ly.band_names) if ly.band_names else "—"
        lines.append(
            f"| {ly.name} | {'yes' if ly.is_tiled else 'no'} | "
            f"{ly.block_width}×{ly.block_height} | {ovr} | {ly.internal_tiles} | "
            f"{ly.codec} | {ratio} | {bands} |"
        )
    return lines


def _render_chunk_shard_layout(layouts: list) -> list[str]:
    """Render the GeoZarr per-array chunk/shard layout: a coverage line plus a table.

    Leads with the total shard-object count (the stored, tier-judged objects), then
    one row per array (chunk = addressable unit, shard = stored object,
    chunks/shard, codec, multiscale levels) — the GeoZarr answer to the same
    partial-access question COG answers with internal tiling.

    Arrays sharing a ``grid_group`` (several components bundled into one
    store, #102) share that group's coordinate arrays and pyramid metadata —
    flagged with a coverage line and a "Grid" column naming each array's group.
    """
    shards = sum(ly.shard_count for ly in layouts)
    grid_groups = {ly.grid_group for ly in layouts if ly.grid_group}
    lines = [
        "",
        "## Chunk/shard layout",
        "",
        f"- **Shard objects:** {shards} across {len(layouts)} array(s)",
    ]
    if grid_groups:
        lines.append(
            f"- **Bundled:** {len(layouts)} array(s) share {len(grid_groups)} "
            "grid group(s) — one set of coordinate arrays and pyramid "
            "metadata per group, not per array"
        )
    lines += [
        "",
        "| Array | Chunk | Shard | Chunks/shard | Codec | Levels | Shards"
        " | Compression | Value encoding | Grid | Native vs. derived |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        chunk = "×".join(str(v) for v in ly.chunk_shape)
        shard = "×".join(str(v) for v in ly.shard_shape)
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        # Whether the array itself applies the scale, or leaves that to the
        # client — the encoding-semantics contrast with COG.
        if ly.scale_offset:
            encoding = "scale_offset"
            if ly.stored_dtype:
                encoding += f" → {ly.stored_dtype}"
        else:
            encoding = ly.stored_dtype or "—"
        # A multi-resolution bundle's unified pyramid (#112) can start a
        # component at any level, not just 0 — say plainly which of its
        # levels are its own measured data and which are derived, rather
        # than leaving a reader to assume level 0 is always native.
        if ly.native_level is None:
            native = "—"
        elif ly.multiscale_levels:
            last = ly.native_level + ly.multiscale_levels
            native = f"{ly.native_level} native, {ly.native_level + 1}-{last} derived"
        else:
            native = f"{ly.native_level} native"
        lines.append(
            f"| {ly.name} | {chunk} | {shard} | {ly.chunks_per_shard} | "
            f"{ly.codec} | {ly.multiscale_levels} | {ly.shard_count} | {ratio} "
            f"| {encoding} | {ly.grid_group or '—'} | {native} |"
        )
    overview = sum(ly.overview_bytes for ly in layouts)
    total = sum(ly.size_bytes for ly in layouts)
    if overview:
        share = f"{100 * overview / total:.1f}%" if total else "—"
        lines += [
            "",
            f"> **Overviews:** {_format_bytes(overview)} of {_format_bytes(total)}"
            f" ({share}) is the multiscale pyramid — the levels a zoomed-out tile"
            " is served from, instead of downsampling full-resolution chunks on"
            " the fly. A COG carries its overviews in the same objects, so the"
            " size comparison above is like-for-like.",
        ]
    elif any(ly.multiscale_levels == 0 for ly in layouts):
        lines += [
            "",
            "> **No overviews:** these arrays are single-level, so a tile server"
            " must read full-resolution chunks for every zoomed-out tile. The"
            " display numbers are not comparable with an arm that has a pyramid.",
        ]
    if any(ly.scale_offset for ly in layouts):
        lines += [
            "",
            "> `scale_offset` arrays apply the source's scale in the array's own"
            " codec pipeline, so any Zarr reader returns physical units without"
            " unscaling them itself. The packed integer is still what occupies"
            " the shard, so the sizes above stay comparable with the COG arm —"
            " where the same scale is out-of-band metadata the client must find"
            " and apply.",
        ]
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
        "| Object | Geometry | Features | Row groups | Rows/group"
        " | Bbox covering | Codec | Compression |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        lines.append(
            f"| {ly.name} | {ly.geometry_column} | {ly.num_rows} | "
            f"{ly.num_row_groups} | {ly.row_group_rows} | "
            f"{'yes' if ly.has_bbox_covering else 'no'} | {ly.codec} | {ratio} |"
        )
    return lines


def _render_spatial_index_layout(layouts: list) -> list[str]:
    """Render the FlatGeobuf per-file index layout: a coverage line plus a table.

    Leads with how many files carry the packed Hilbert R-tree (whether a bbox query
    can select features at all, rather than scan), then one row per file (geometry
    type, feature count, node size, what the index costs) — the FlatGeobuf answer to
    the same partial-access question COG answers with internal tiling. The index
    share is quoted because it is the format's own overhead, and the only part of a
    FlatGeobuf that is not the data itself.
    """
    indexed = sum(1 for ly in layouts if ly.has_spatial_index)
    index_bytes = sum(ly.index_bytes for ly in layouts)
    total = sum(ly.size_bytes for ly in layouts)
    share = f" ({100 * index_bytes / total:.1f}% of the stored bytes)" if total else ""
    subset_files = [ly for ly in layouts if ly.content_subset]
    fabricated_files = [ly for ly in layouts if ly.geometry_fabricated]
    synthesized_files = [ly for ly in layouts if ly.geometry_synthesized]
    lines = [
        "",
        "## Spatial-index layout",
        "",
        f"- **Packed Hilbert R-tree:** {indexed}/{len(layouts)} file(s) "
        "(bbox selection of features, not a scan)",
        f"- **Index cost:** {_format_bytes(index_bytes)}{share}",
    ]
    if subset_files:
        dropped = sum(ly.features_dropped for ly in subset_files)
        lines.append(
            f"- **Content subset:** {len(subset_files)} file(s) dropped "
            f"{dropped} feature(s) with a NULL geometry — not content-complete"
        )
    if synthesized_files:
        synthesized = sum(ly.features_synthesized for ly in synthesized_files)
        lines.append(
            f"- **Synthesized geometry:** {len(synthesized_files)} file(s) built "
            f"{synthesized} geometry(ies) from attribute columns for features "
            "with no real one"
        )
    if fabricated_files:
        sentinel = sum(ly.features_sentinel for ly in fabricated_files)
        lines.append(
            f"- **Fabricated geometry:** {len(fabricated_files)} file(s) hold "
            f"{sentinel} placeholder geometry(ies) for features with no real one"
        )
    lines += [
        "",
        "| Object | Geometry | Features | Dropped | Synthesized | Sentinel | Indexed"
        " | Node size | Index | Features bytes | Codec | Compression |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        lines.append(
            f"| {ly.name} | {ly.geometry_type} | {ly.num_features} | "
            f"{ly.features_dropped} | {ly.features_synthesized} | "
            f"{ly.features_sentinel} | "
            f"{'yes' if ly.has_spatial_index else 'no'} | {ly.index_node_size} | "
            f"{_format_bytes(ly.index_bytes)} | {_format_bytes(ly.feature_bytes)} | "
            f"{ly.codec} | {ratio} |"
        )
    lines += [
        "",
        "> FlatGeobuf stores raw flatbuffers and defines no compression, so the"
        " stored bytes are the uncompressed bytes (1.00×). That is the number to"
        " read a GeoParquet arm's compression ratio against: the two are the"
        " vector candidates on the same source.",
    ]
    if subset_files:
        lines += [
            "",
            "> **Content subset:** a NULL geometry cannot be indexed by the packed"
            " R-tree, so `null_geometry: drop` wrote the non-null subset — the"
            " `Features`/`Dropped` columns above are not the full source, and"
            " this arm's size is not comparable to a content-complete arm on the"
            " same source without accounting for the drop.",
        ]
    if synthesized_files:
        lines += [
            "",
            "> **Synthesized geometry:** `null_geometry: point_from` kept every"
            " feature by building a real geometry from named lon/lat columns for"
            " the ones with no real one (see the `Synthesized` column) — a"
            " reshaping of source attributes, not fabricated data, but still"
            " worth knowing which rows it applies to.",
        ]
    if fabricated_files:
        lines += [
            "",
            "> **Fabricated geometry:** `null_geometry: sentinel` kept every"
            " feature by substituting a placeholder geometry, positioned outside"
            " the real content's extent, for the ones with no real geometry (see"
            " the `Sentinel` column). Those bytes do not exist in the source, so"
            " this arm's size is not comparable to a GeoParquet arm's — a NULL"
            " geometry costs that format near nothing.",
        ]
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
        "| Object | Nodes | Depth | Points | Points/node | Extra dims"
        " | Codec | Compression |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ly in layouts:
        ratio = f"{ly.compression_ratio:.2f}×" if ly.compression_ratio else "—"
        lines.append(
            f"| {ly.name} | {ly.num_nodes} | {ly.max_depth} | "
            f"{ly.point_count} | {ly.points_per_node} | "
            f"{len(ly.extra_dimensions)} | {ly.codec} | {ratio} |"
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
        # row-group count (its addressable units); FlatGeobuf, the indexed fraction
        # — each format's own answer to the partial-access question.
        cog = [ly for ly in layouts if ly.kind == "cog"]
        if cog:
            return f"{sum(1 for ly in cog if ly.is_tiled)}/{len(cog)} tiled"
        geoparquet = [ly for ly in layouts if ly.kind == "geoparquet"]
        if geoparquet:
            groups = sum(ly.num_row_groups for ly in geoparquet)
            return f"{groups} row groups"
        flatgeobuf = [ly for ly in layouts if ly.kind == "flatgeobuf"]
        if flatgeobuf:
            indexed = sum(1 for ly in flatgeobuf if ly.has_spatial_index)
            return f"{indexed}/{len(flatgeobuf)} indexed"
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
    lines += _render_conditions(result.rollup.conditions)
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
