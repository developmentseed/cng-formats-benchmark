"""Chunk-aware tile selection for the display metric (geo, ``cog`` extra).

Finds WebMercator tiles that deliberately exercise different chunk-crossing
scenarios — a tile that lands inside one internal block, one that straddles a
block boundary (2), a corner that straddles both (4), and an over-zoomed-out tile
that spans 9+ blocks. Timing those tiles (in :mod:`cng_benchmark.metrics.display`)
shows how display latency scales with the number of blocks TiTiler must read.

Selection is analytical: it reads the local COG's block size and overview
decimations and reproduces GDAL/rio-tiler's overview pick + window math to count
how many block-grid cells each tile intersects. That is faithful enough for
*bucketing* tiles; the measured TiTiler latency is the real signal. This module
needs rasterio + morecantile (both in the ``cog`` extra), kept apart from the
HTTP-only display collector so the latter stays import-light.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from cng_benchmark.metrics.display import TileSpec


class _TileWindow(NamedTuple):
    """A tile's footprint in dataset full-res pixel space.

    ``clipped`` is the read window intersected with the raster (what GDAL/TiTiler
    actually reads, used for chunk counting); ``raw`` is the full WebMercator tile
    TiTiler serves projected into pixel space (used for drawing the served tile);
    ``interior`` is ``True`` when ``raw`` lies fully inside the raster.
    """

    clipped: tuple[float, float, float, float]
    raw: tuple[float, float, float, float]
    interior: bool


#: Default chunk-count buckets to find tiles for.
DEFAULT_TARGETS: tuple[int, ...] = (1, 2, 4, 9)


class _Grid(NamedTuple):
    """The block/chunk grid of a produced object, in full-resolution pixel space.

    The format-agnostic input to tile selection and the layout image: a COG yields
    it from its internal block size + overviews, a GeoZarr store from its chunk
    shape + multiscale levels. ``block_w``/``block_h`` is the addressable unit
    (COG block / Zarr chunk); ``decimations`` the available overview/level
    decimations (``[1, …]``, real ratios read off the store's own pyramid, not
    assumed to double — #116); ``level_paths`` the group each entry of
    ``decimations`` is actually served from (``None`` for COG, which has no
    server-side group concept — rio-tiler resolves its overviews itself);
    ``native_resolution`` the object's own native pixel size, in its own CRS'
    units (metres for a projected CRS) — the same units a ``multiscale_levels``/
    ``display_target_resolutions`` config value is given in, *not* the
    WebMercator-projected resolution tile selection otherwise works in
    (#116; the two differ by a latitude-dependent scale factor, so a target
    resolution given in native units has to be converted before it can be
    compared against a WebMercator zoom's cell size); ``inv`` the inverse
    affine (world → pixel).
    """

    block_w: int
    block_h: int
    decimations: list[int]
    native_resolution: float
    crs: object
    inv: object
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    level_paths: list[str | None]


def _require_geo():
    """Import the geo stack, raising a clear error if the ``cog`` extra is absent."""
    try:
        import morecantile
        import rasterio
        from rasterio.warp import transform_bounds
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "chunk-aware tile selection requires the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return morecantile, rasterio, transform_bounds


def _read_cog_grid(cog_path: str) -> _Grid:
    """Read the block/overview grid of a COG with rasterio."""
    _morecantile, rasterio, _tb = _require_geo()
    with rasterio.open(cog_path) as src:
        block_h, block_w = src.block_shapes[0]
        decimations = [1, *src.overviews(1)]
        return _Grid(
            block_w=int(block_w),
            block_h=int(block_h),
            decimations=decimations,
            native_resolution=abs(src.transform.a),
            crs=src.crs,
            inv=~src.transform,
            width=src.width,
            height=src.height,
            bounds=tuple(src.bounds),
            # rio-tiler resolves a COG's overview by zoom internally -- no
            # server-side group to name, unlike the stock GeoZarr router.
            level_paths=[None] * len(decimations),
        )


def _entry_resolution(entry: dict, fallback: float) -> float:
    """Ground resolution (native's own units) one multiscales layout entry
    declares via its own ``spatial:transform`` — ``fallback`` if absent."""
    tr = entry.get("spatial:transform")
    if not tr or len(tr) < 5:
        return fallback
    return min(abs(tr[0]), abs(tr[4]))


def _resolve_multiscale_layout(
    open_group, attrs: dict, group: str | None
) -> tuple[list[dict], str]:
    """Return a GeoZarr store's real pyramid layout for ``group``, and the
    path prefix each entry's ``asset`` resolves against.

    Tries ``group``'s own ``zarr_conventions.multiscales.layout`` first — a
    non-batched store (``group=None``, the doc is on the store root) or a
    ``#102`` single-resolution bundle's grid subtree (``group="gridN"``)
    both declare it directly on the group being addressed, describing
    exactly that component's own levels, nested as its children. Falls back
    to the *store root*'s own doc when ``group`` has none of its own — a
    ``#114`` unified multi-resolution pyramid, where every level is written
    as a top-level group at the store root and ``group`` just names the
    level a component's own data *starts* at (its
    :meth:`~cng_benchmark.formats.geozarr.GeoZarrAdapter.component_locator`)
    — scoped to that component's own contiguous chain by slicing the root
    layout from the entry whose ``asset`` matches ``group`` onward, never
    the levels below it where that component has no data (by construction
    of how ``#114`` writes it).
    """
    layout = (attrs.get("multiscales") or {}).get("layout")
    if layout:
        return list(layout), (group or "")
    if not group:
        return [], ""
    root_attrs = dict(open_group("").attrs)
    root_layout = (root_attrs.get("multiscales") or {}).get("layout") or []
    start = next(
        (i for i, e in enumerate(root_layout) if e.get("asset") == group), None
    )
    if start is None:
        return [], ""
    return list(root_layout[start:]), ""


def _read_zarr_grid(
    store: str,
    role: str = "sink",
    *,
    group: str | None = None,
    var_name: str | None = None,
) -> _Grid:
    """Read the chunk/multiscale grid of a GeoZarr store (chunk = addressable unit).

    Chunk shape is the partial-read unit; decimations and the group serving
    each level are read from the store's own ``zarr_conventions.multiscales.
    layout`` (:func:`_resolve_multiscale_layout`) — the doc a spec-compliant
    reader (GeoZarrReader included) already uses — rather than re-derived by
    walking directory structure, which silently under-counts a ``#114``
    unified multi-resolution bundle's levels (#116). The CRS and the affine
    come from the store's own ``geo-proj``/``spatial`` convention attributes.
    ``group``/``var_name`` address one component of a bundled store
    (#102/#114) — ``None`` (the defaults) reproduce today's single-component
    addressing.
    """
    import zarr
    from affine import Affine
    from rasterio.crs import CRS

    from cng_benchmark.formats.geozarr import DATA_VAR
    from cng_benchmark.storage import fsspec_storage_options, is_s3

    name = var_name or DATA_VAR
    so = fsspec_storage_options(role) if is_s3(store) else None

    def _open(path: str):
        return zarr.open_group(store, mode="r", storage_options=so, path=path)

    root = _open(group or "")
    attrs = dict(root.attrs)
    crs_def = attrs.get("proj:code") or attrs.get("proj:wkt2") or ""
    coeffs = [float(v) for v in attrs.get("spatial:transform", [])]
    # Tile selection has to project the store into WebMercator, so both the CRS
    # and a 6-coefficient affine must be present; fail clearly if not (an
    # ungeoreferenced store has no map tiles to time).
    if not crs_def or len(coeffs) != 6:
        raise RuntimeError(
            f"GeoZarr store {store!r} is not georeferenced for display "
            f"(proj:code/proj:wkt2={'set' if crs_def else 'missing'}, "
            f"spatial:transform has {len(coeffs)} of 6 coefficients); "
            "cannot select map tiles"
        )
    # `spatial:transform` is already in rasterio/affine coefficient order
    # ([a, b, c, d, e, f]) -- `a` is the pixel width, this group's own native
    # resolution regardless of whether a multiscales doc exists.
    native_resolution = abs(coeffs[0])

    layout, level_prefix = _resolve_multiscale_layout(_open, attrs, group)
    if layout:
        level_paths = [
            f"{level_prefix}/{e['asset']}" if level_prefix else str(e["asset"])
            for e in layout
        ]
        arr = _open(level_paths[0])[name]
        decimations = [
            max(1, round(_entry_resolution(e, native_resolution) / native_resolution))
            for e in layout
        ]
    else:
        # No multiscales doc at all: a flat, single-level store or component.
        arr = root[name]
        level_paths = [group]
        decimations = [1]

    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    block_h, block_w = int(arr.chunks[-2]), int(arr.chunks[-1])
    crs = CRS.from_user_input(crs_def)
    transform = Affine(*coeffs)
    inv = ~transform
    # All four raster corners, so a rotated/sheared transform still bounds right.
    corners = [(0, 0), (width, 0), (0, height), (width, height)]
    pts = [transform * (cx, cy) for cx, cy in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    return _Grid(
        block_w,
        block_h,
        decimations,
        native_resolution,
        crs,
        inv,
        width,
        height,
        bounds,
        level_paths,
    )


def _blocks_spanned(lo: float, hi: float, block: float) -> int:
    """Number of ``block``-sized grid cells the pixel span ``[lo, hi)`` touches."""
    if hi <= lo:
        return 1
    first = math.floor(lo / block)
    last = math.floor((hi - 1e-9) / block)
    return max(1, last - first + 1)


def _pick_decimation(decimations: list[int], native_res: float, tile_res: float) -> int:
    """Reproduce GDAL overview selection: coarsest level still finer than the tile.

    Returns the largest decimation whose resolution is ``<= tile_res`` (GDAL reads
    a finer-or-equal overview and downsamples). If the tile is finer than full
    resolution, use full res (decimation 1); if it is coarser than the coarsest
    overview, GDAL is stuck with that coarsest level.
    """
    eligible = [d for d in decimations if native_res * d <= tile_res]
    return max(eligible) if eligible else min(decimations)


def _tile_pixel_window(
    tile, tms, transform_bounds, crs, inv, width, height
) -> _TileWindow | None:
    """Project a WebMercator tile into dataset pixel space.

    The tile's mercator bounds (the exact extent TiTiler serves) are reprojected
    into the dataset CRS and turned into a pixel bounding box (robust to any
    affine orientation). Returns ``None`` when the tile does not overlap the
    dataset at all; otherwise both the clipped read window and the raw served-tile
    window, plus whether the tile is interior. See :class:`_TileWindow`.
    """
    tb = tms.xy_bounds(tile)
    left, bottom, right, top = transform_bounds(
        tms.crs, crs, tb.left, tb.bottom, tb.right, tb.top
    )
    cols, rows = [], []
    for x in (left, right):
        for y in (bottom, top):
            c, r = inv * (x, y)
            cols.append(c)
            rows.append(r)
    raw_c0, raw_c1 = min(cols), max(cols)
    raw_r0, raw_r1 = min(rows), max(rows)
    col0 = max(0.0, raw_c0)
    col1 = min(float(width), raw_c1)
    row0 = max(0.0, raw_r0)
    row1 = min(float(height), raw_r1)
    if col1 <= col0 or row1 <= row0:
        return None
    interior = (
        raw_c0 >= -0.5
        and raw_c1 <= width + 0.5
        and raw_r0 >= -0.5
        and raw_r1 <= height + 0.5
    )
    return _TileWindow(
        (col0, col1, row0, row1), (raw_c0, raw_c1, raw_r0, raw_r1), interior
    )


def _count_chunks(
    tile, tms, transform_bounds, crs, inv, width, height, block_w, block_h, decim
) -> tuple[int, bool] | None:
    """Blocks rendering ``tile`` reads and whether it is an interior tile.

    Maps the (clipped) tile read window into the chosen overview's pixel space and
    counts the block-grid cells it intersects. Returns ``None`` when the tile does
    not overlap the dataset.
    """
    window = _tile_pixel_window(tile, tms, transform_bounds, crs, inv, width, height)
    if window is None:
        return None
    col0, col1, row0, row1 = window.clipped
    n_x = _blocks_spanned(col0 / decim, col1 / decim, block_w)
    n_y = _blocks_spanned(row0 / decim, row1 / decim, block_h)
    return n_x * n_y, window.interior


def select_chunk_tiles(
    cog_path: str,
    *,
    tile_matrix_set: str = "WebMercatorQuad",
    targets: tuple[int, ...] = DEFAULT_TARGETS,
    max_tiles_per_zoom: int = 4000,
) -> list[TileSpec]:
    """Select one tile per chunk-count bucket in ``targets`` for the COG."""
    return _select_from_grid(
        _read_cog_grid(cog_path),
        tile_matrix_set=tile_matrix_set,
        targets=targets,
        max_tiles_per_zoom=max_tiles_per_zoom,
    )


def select_zarr_chunk_tiles(
    store: str,
    *,
    role: str = "sink",
    tile_matrix_set: str = "WebMercatorQuad",
    targets: tuple[int, ...] = DEFAULT_TARGETS,
    max_tiles_per_zoom: int = 4000,
    group: str | None = None,
    var_name: str | None = None,
) -> list[TileSpec]:
    """Select one tile per chunk-count bucket for the GeoZarr ``store``.

    The store counterpart to :func:`select_chunk_tiles`: a tile's cost is how many
    Zarr *chunks* its footprint straddles, the same partial-access question COG
    answers with internal blocks. ``group``/``var_name`` address one component
    of a bundled store (#102); ``None`` reproduces today's addressing.
    """
    return _select_from_grid(
        _read_zarr_grid(store, role, group=group, var_name=var_name),
        tile_matrix_set=tile_matrix_set,
        targets=targets,
        max_tiles_per_zoom=max_tiles_per_zoom,
    )


def select_resolution_tiles(
    cog_path: str,
    *,
    target_resolutions: list[float] | None = None,
    tile_matrix_set: str = "WebMercatorQuad",
    max_tiles_per_zoom: int = 4000,
) -> list[TileSpec]:
    """Select one tile per target ground resolution for the COG (#116).

    Pass the *same* ``target_resolutions`` as the matched GeoZarr arm's
    :func:`select_zarr_resolution_tiles` call so ``"res_60m"`` means the
    same physical zoom regardless of format — comparing a format-relative
    overview/level index instead would silently compare different ground
    resolutions and call it a format comparison (#116). ``None`` derives
    resolutions from the COG's own overview decimations (today's implicit
    per-format behaviour), for an arm where cross-format alignment isn't
    the point.
    """
    return _select_by_resolution(
        _read_cog_grid(cog_path),
        target_resolutions,
        tile_matrix_set=tile_matrix_set,
        max_tiles_per_zoom=max_tiles_per_zoom,
    )


def select_zarr_resolution_tiles(
    store: str,
    *,
    role: str = "sink",
    target_resolutions: list[float] | None = None,
    tile_matrix_set: str = "WebMercatorQuad",
    max_tiles_per_zoom: int = 4000,
    group: str | None = None,
    var_name: str | None = None,
) -> list[TileSpec]:
    """Select one tile per target ground resolution for the GeoZarr ``store`` (#116).

    Each tile carries the ``.group`` that actually serves its resolution,
    read from the store's real pyramid (:func:`_read_zarr_grid`, correctly
    scoped to one component even for a `#114` unified multi-resolution
    bundle) — :func:`~cng_benchmark.metrics.display.measure_display`'s
    per-tile query override uses it so the stock router reads the
    resolution-appropriate level instead of always the native one.
    ``group``/``var_name`` address one component of a bundled store
    (#102/#114); ``None`` reproduces today's addressing.
    """
    return _select_by_resolution(
        _read_zarr_grid(store, role, group=group, var_name=var_name),
        target_resolutions,
        tile_matrix_set=tile_matrix_set,
        max_tiles_per_zoom=max_tiles_per_zoom,
    )


def _select_by_resolution(
    grid: _Grid,
    target_resolutions: list[float] | None,
    *,
    tile_matrix_set: str = "WebMercatorQuad",
    max_tiles_per_zoom: int = 4000,
) -> list[TileSpec]:
    """Select one representative tile per target ground resolution for ``grid``.

    Unlike :func:`_select_from_grid` (bucketed by how many chunks a tile
    touches — a different, still-useful question), this is bucketed by
    *zoom range coverage*: one tile per entry of ``target_resolutions``
    (in ``grid.native_resolution``'s units — metres for a projected CRS,
    matching the ``multiscale_levels``/``display_target_resolutions``
    convention), or, when not given, one per the grid's own decimations
    (today's implicit per-format behaviour). Each resolution maps to the
    coarsest decimation still finer-or-equal to it (:func:`_pick_decimation`
    — the same GDAL-overview-style rule the tiler itself effectively
    applies) and thus to the group that decimation's level is actually
    served from (``grid.level_paths``, ``None`` for COG). Labeled by
    resolution (``"res_{r:g}m"``), not a format-relative level index, so the
    label is comparable across formats.
    """
    morecantile, _rasterio, transform_bounds = _require_geo()
    block_w, block_h = grid.block_w, grid.block_h
    decimations = grid.decimations
    native_resolution = grid.native_resolution
    crs, inv = grid.crs, grid.inv
    width, height = grid.width, grid.height
    bounds = grid.bounds

    tms = morecantile.tms.get(tile_matrix_set)
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
    mleft, _, mright, _ = transform_bounds(crs, tms.crs, *bounds)
    native_res_webmercator = (mright - mleft) / width
    # `target_resolutions` are in the object's own native CRS units, but
    # `tms.zoom_for_res` (and the chunk-counting math below) work in
    # WebMercator metres, which differ from native-CRS metres by a
    # latitude-dependent scale factor -- derived once here from the two
    # already-known native resolutions, rather than reprojecting per tile.
    to_webmercator = native_res_webmercator / native_resolution

    resolutions = target_resolutions or [native_resolution * d for d in decimations]

    selected: list[TileSpec] = []
    for res in resolutions:
        decim = _pick_decimation(decimations, native_resolution, res)
        group = grid.level_paths[decimations.index(decim)]
        z = tms.zoom_for_res(res * to_webmercator)

        best: TileSpec | None = None
        best_interior = False
        for i, tile in enumerate(tms.tiles(west, south, east, north, [z])):
            if i >= max_tiles_per_zoom:
                break
            counted = _count_chunks(
                tile,
                tms,
                transform_bounds,
                crs,
                inv,
                width,
                height,
                block_w,
                block_h,
                decim,
            )
            if counted is None:
                continue
            chunks, interior = counted
            if best is None or (interior and not best_interior):
                best = TileSpec(
                    f"res_{res:g}m", tile.z, tile.x, tile.y, chunks, group=group
                )
                best_interior = interior
                if interior:
                    break
        if best is not None:
            selected.append(best)
    return selected


def _select_from_grid(
    grid: _Grid,
    *,
    tile_matrix_set: str = "WebMercatorQuad",
    targets: tuple[int, ...] = DEFAULT_TARGETS,
    max_tiles_per_zoom: int = 4000,
) -> list[TileSpec]:
    """Select one tile per chunk-count bucket in ``targets`` for ``grid``.

    Scans WebMercator zooms from the object's native zoom downward — finer zooms
    expose small chunk counts (1/2/4); coarser-than-the-coarsest-overview zooms
    force larger reads (9+). Returns one :class:`TileSpec` per reachable target
    (label ``"{n}chunk"``); unreachable buckets are silently skipped. The ``9``
    bucket accepts the smallest count ``>= 9``.
    """
    morecantile, _rasterio, transform_bounds = _require_geo()
    block_w, block_h = grid.block_w, grid.block_h
    decimations = grid.decimations
    crs, inv = grid.crs, grid.inv
    width, height = grid.width, grid.height
    bounds = grid.bounds

    tms = morecantile.tms.get(tile_matrix_set)
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds)
    mleft, _, mright, _ = transform_bounds(crs, tms.crs, *bounds)
    native_res = (mright - mleft) / width
    native_zoom = tms.zoom_for_res(native_res)

    # Go from native zoom down past the coarsest overview into the over-zoom
    # regime, where the per-tile read grows and high chunk counts appear.
    z_lo = max(0, native_zoom - len(decimations) - 4)

    # chunk count -> (is_interior, representative tile). Interior tiles (footprint
    # fully inside the raster) are preferred over edge tiles that read mostly
    # out-of-bounds, since the latter make for misleading timings and pictures.
    best: dict[int, tuple[bool, TileSpec]] = {}
    for z in range(native_zoom, z_lo - 1, -1):
        tile_res = tms.matrix(z).cellSize
        decim = _pick_decimation(decimations, native_res, tile_res)
        for i, tile in enumerate(tms.tiles(west, south, east, north, [z])):
            if i >= max_tiles_per_zoom:
                break
            counted = _count_chunks(
                tile,
                tms,
                transform_bounds,
                crs,
                inv,
                width,
                height,
                block_w,
                block_h,
                decim,
            )
            if counted is None:
                continue
            chunks, interior = counted
            prev = best.get(chunks)
            if prev is None or (interior and not prev[0]):
                best[chunks] = (interior, TileSpec("", tile.z, tile.x, tile.y, chunks))
        if all(_match(best, t) is not None for t in targets):
            break

    selected: list[TileSpec] = []
    for t in targets:
        chosen = _match(best, t)
        if chosen is None:
            continue
        approx = chosen.chunks != t
        selected.append(chosen._replace(label=f"{t}chunk", approx=approx))
    return selected


def _match(best: dict[int, tuple[bool, TileSpec]], target: int) -> TileSpec | None:
    """Pick the tile for ``target``: exact count, or smallest ``>=`` for ``9+``."""
    if target in best:
        return best[target][1]
    if target >= 9:
        over = sorted(c for c in best if c >= target)
        if over:
            return best[over[0]][1]
    return None


def _require_viz():
    """Import matplotlib (Agg backend), raising a clear error if it is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed in the runner
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "the chunk-layout image requires matplotlib (in the 'cog' extra); "
            "install with `uv sync --extra cog`"
        ) from exc
    return plt, Rectangle


def render_chunk_layout(
    cog_path: str,
    tiles: list[TileSpec],
    out_path: str,
    *,
    tile_matrix_set: str = "WebMercatorQuad",
) -> str:
    """Render the block grid with each selected tile's footprint to ``out_path``.

    One panel per scenario tile: the object's internal block/chunk grid (at that
    tile's overview resolution) over the dataset extent, with the tile's read
    footprint highlighted and annotated with its chunk count — the "1 vs 2 vs 4 vs
    9+ chunks per tile" picture, mirroring the reference notebook. Returns
    ``out_path``. Requires matplotlib (the ``cog`` extra).
    """
    return _render_from_grid(
        _read_cog_grid(cog_path), tiles, out_path, tile_matrix_set=tile_matrix_set
    )


def render_zarr_chunk_layout(
    store: str,
    tiles: list[TileSpec],
    out_path: str,
    *,
    role: str = "sink",
    tile_matrix_set: str = "WebMercatorQuad",
    group: str | None = None,
    var_name: str | None = None,
) -> str:
    """Render the GeoZarr store's chunk grid with each tile's footprint.

    ``group``/``var_name`` address one component of a bundled store (#102);
    ``None`` reproduces today's addressing.
    """
    return _render_from_grid(
        _read_zarr_grid(store, role, group=group, var_name=var_name),
        tiles,
        out_path,
        tile_matrix_set=tile_matrix_set,
    )


def _render_from_grid(
    grid: _Grid,
    tiles: list[TileSpec],
    out_path: str,
    *,
    tile_matrix_set: str = "WebMercatorQuad",
) -> str:
    """Render ``grid``'s block/chunk grid with each selected tile's footprint."""
    morecantile, _rasterio, transform_bounds = _require_geo()
    plt, Rectangle = _require_viz()

    block_w, block_h = grid.block_w, grid.block_h
    decimations = grid.decimations
    crs, inv = grid.crs, grid.inv
    width, height = grid.width, grid.height
    bounds = grid.bounds

    tms = morecantile.tms.get(tile_matrix_set)
    mleft, _, mright, _ = transform_bounds(crs, tms.crs, *bounds)
    native_res = (mright - mleft) / width

    panels = tiles or []
    fig, axes = plt.subplots(
        1, max(1, len(panels)), figsize=(4.2 * max(1, len(panels)), 4.4), squeeze=False
    )
    for ax, spec in zip(axes[0], panels, strict=False):
        tile_res = tms.matrix(spec.z).cellSize
        decim = _pick_decimation(decimations, native_res, tile_res)
        # Block spacing in full-res pixels at the overview the tile is read from.
        step_x = block_w * decim
        step_y = block_h * decim

        ax.add_patch(
            Rectangle((0, 0), width, height, fill=False, edgecolor="0.4", linewidth=1.0)
        )
        gx = 0
        while gx <= width:
            ax.axvline(gx, color="0.8", linewidth=0.6, zorder=0)
            gx += step_x
        gy = 0
        while gy <= height:
            ax.axhline(gy, color="0.8", linewidth=0.6, zorder=0)
            gy += step_y

        window = _tile_pixel_window(
            morecantile.Tile(spec.x, spec.y, spec.z),
            tms,
            transform_bounds,
            crs,
            inv,
            width,
            height,
        )
        if window is not None:
            # Draw the actual WebMercator tile TiTiler serves (raw bounds).
            rc0, rc1, rr0, rr1 = window.raw
            ax.add_patch(
                Rectangle(
                    (rc0, rr0),
                    rc1 - rc0,
                    rr1 - rr0,
                    facecolor="tab:orange",
                    alpha=0.45,
                    edgecolor="tab:red",
                    linewidth=1.5,
                    zorder=2,
                    label="TiTiler tile",
                )
            )
            # Zoom to the tile's neighbourhood (± ~1.5 blocks) so the
            # chunk-crossing is legible instead of a speck on the full extent.
            cx, cy = (rc0 + rc1) / 2, (rr0 + rr1) / 2
            half = max(rc1 - rc0, rr1 - rr0) / 2 + 1.5 * max(step_x, step_y)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy + half, cy - half)  # image row order (y down)
        else:
            ax.set_xlim(-0.02 * width, 1.02 * width)
            ax.set_ylim(1.02 * height, -0.02 * height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        approx = " (approx)" if spec.approx else ""
        ax.set_title(
            f"{spec.label}: {spec.chunks} chunk(s){approx}\n"
            f"z/x/y={spec.z}/{spec.x}/{spec.y}, overview ÷{decim}",
            fontsize=9,
        )

    if not panels:
        axes[0][0].text(
            0.5,
            0.5,
            "no chunk scenarios reachable",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0][0].set_axis_off()

    fig.suptitle("Tile footprint vs internal block (chunk) grid", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
