"""GeoZarr v3 adapter — 2D, per-component, sharded, multiscale.

The grouping lever for Zarr v3 is its chunk and shard shape: the chunk is the
addressable (range-read) unit, and a *shard* packs many chunks into one stored
object, so shard size is the knob that lifts the mean object size into a storage
tier (ADR 0001). This adapter converts one baseline raster to one sharded 2D
GeoZarr store, the per-component analogue of the COG arm — it flows through the
same runner paths as COG. Time-stacking the source scenes into a 3D cube is a
separate, deferred concern (see the M2.5 plan): nothing here stacks.

The store carries an **overview pyramid by default** — the direct analogue of
COG overviews, without which a tile server has to read full-resolution chunks
for every zoomed-out tile and the display comparison measures the missing
pyramid rather than the format (#71). The sibling
:mod:`~cng_benchmark.formats.geozarr_multiscales` module builds the metadata
that describes it.

The store-writing core (:func:`_write_sharded`, :func:`enumerate_store_objects`,
:func:`describe_store_layout`) depends only on ``zarr`` + ``xarray`` + ``numpy``
+ ``zarr_cm``, so the sharding-lever / enumerate / layout logic is unit-testable
on synthetic in-memory arrays. Only :meth:`GeoZarrAdapter.convert`'s source read
needs ``rioxarray`` (the ``geozarr`` extra), imported lazily.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats import geozarr_multiscales as ms
from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.formats.grid import GridKey, group_by_grid
from cng_benchmark.models import GeoZarrLayout
from cng_benchmark.registry import FORMATS

if TYPE_CHECKING:
    from cng_benchmark.datasets.base import SourceObject

#: The single data variable each per-component store holds (the display and read
#: metrics address the array by this name).
DATA_VAR = "data"

#: Defaults when the config carries no lever value (spatial y, x).
DEFAULT_CHUNK = (1024, 1024)
DEFAULT_SHARD = (2048, 2048)


def finest_level_group(
    store: str, *, group: str | None = None, var_name: str | None = None
) -> str | None:
    """The zarr group holding the native-resolution data array.

    ``None`` for a flat store (the array at the pyramid group's own root); the
    lowest-numbered multiscale level group (``"0"``, native resolution — see
    :mod:`~cng_benchmark.formats.geozarr_multiscales`) otherwise. TiTiler's
    stock xarray router has no multiscale awareness and needs this explicitly
    as its ``group=`` query; a reader that resolves the pyramid itself (e.g.
    ``GeoZarrReader``) does not. Mirrors the array lookup
    :func:`cng_benchmark.metrics.read._open_zarr_array` does for the read
    metric. ``group``/``var_name`` address one component of a bundled store
    (#102) — ``None`` (the defaults) reproduce today's single-component
    addressing (:data:`DATA_VAR` at the store root).
    """
    import zarr

    name = var_name or DATA_VAR
    root = zarr.open_group(store, mode="r", path=group or "")
    if name in root:
        return None
    level_keys = sorted((k for k in root.group_keys()), key=int)
    return level_keys[0]


class GeoZarrParams(BaseModel):
    """Zarr v3 sharding levers, parsed from ``config.params``.

    ``chunk_shape`` / ``shard_shape`` accept a 2D ``[y, x]`` or a 3D
    ``[t, y, x]`` shape (the trailing two, spatial, dims are used — a leading time
    dim is ignored in the 2D regime) and tolerate a swept *list of shapes*, taking
    the first so a swept lever degrades to a single run, mirroring COG's
    ``block_size``. ``codec`` is the per-chunk compressor (``zstd`` default;
    ``none`` for raw).

    ``multiscale_levels`` is the overview-pyramid depth. ``"auto"`` (the default)
    coarsens by ``/2`` down to roughly one tile, the same rule GDAL follows for
    COG overviews, so the display comparison between the two arms is like-for-like
    — an arm without a pyramid makes the tile server read full-resolution chunks
    for every zoomed-out tile (#71). An explicit integer overrides it; ``0``
    writes the base array alone, which is the flat-store comparison, not a
    representative GeoZarr.

    ``scale_offset`` switches how a packed source (one with a ``scale_factor``)
    is encoded. Off, the CF ``scale_factor``/``add_offset`` attributes are
    written and the array still reads back as the raw stored count — a client
    has to know to unscale, the same fragmented situation COG is in. On, the
    ``scale_offset`` + ``cast_value`` codec chain goes into the array's codec
    pipeline: the array declares a float dtype so *every* reader gets physical
    units unambiguously, while the packed integer is what lands on disk (#54).

    ``standard_name`` is the CF standard name of the quantity the pixels hold
    (e.g. ``surface_bidirectional_reflectance``). GeoZarr requires it on a data
    array; the dataset reader normally supplies it per component, so this is the
    override.
    """

    model_config = ConfigDict(extra="ignore")

    chunk_shape: Any = None
    shard_shape: Any = None
    codec: str = "zstd"
    multiscale_levels: Any = "auto"
    scale_offset: bool = False
    standard_name: str | None = None


def _spatial_pair(shape: Any, default: tuple[int, int]) -> tuple[int, int]:
    """Normalise a config shape to a spatial ``(y, x)`` pair.

    Tolerates a scalar ``1024`` (square, like COG's ``block_size``), a swept *list
    of shapes* (takes the first), a 3D ``[t, y, x]`` shape (takes the trailing
    two), and a 2D ``[y, x]`` shape; falls back to ``default``.
    """
    if shape is None or shape == []:
        return default
    if isinstance(shape, int | float):
        return (int(shape), int(shape))
    if isinstance(shape[0], list | tuple):
        shape = shape[0]
    vals = [int(v) for v in shape]
    if len(vals) >= 2:
        return (vals[-2], vals[-1])
    if len(vals) == 1:
        return (vals[0], vals[0])
    return default


def _multiscale_depth(value: Any, shape: tuple[int, int]) -> int:
    """Resolve the configured pyramid depth against the array's shape.

    ``None`` / ``"auto"`` derives the COG-style depth from the shape; anything
    else is an explicit level count. A depth deeper than the array can carry is
    harmless — :func:`_levels` stops once a level would be smaller than 2 px.
    """
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        return ms.auto_depth(shape)
    return max(0, int(value))


def _fit_chunk(chunk: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    """Clamp a chunk to the array shape (a chunk may not exceed the array)."""
    return tuple(max(1, min(c, n)) for c, n in zip(chunk, shape, strict=True))


def _fit_shard(
    shard: tuple[int, int], chunk: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    """Align a shard to a whole multiple of the chunk, clamped to the array.

    Zarr v3 requires the shard shape to be an integer multiple of the chunk shape;
    a shard also may not exceed the array. Rounds each dim down to a chunk multiple
    (at least one chunk).
    """
    out: list[int] = []
    for s, c, n in zip(shard, chunk, shape, strict=True):
        s = min(s, n)
        out.append(max(c, (s // c) * c))
    return (out[0], out[1])


def _compressor(name: str):
    """Map a codec name to a zarr v3 compressor instance (or ``None`` for raw)."""
    import zarr

    key = (name or "zstd").lower()
    if key in ("none", "raw", ""):
        return None
    table = {
        "zstd": zarr.codecs.ZstdCodec,
        "gzip": zarr.codecs.GzipCodec,
        "blosc": zarr.codecs.BloscCodec,
    }
    if key not in table:
        raise ValueError(f"unknown geozarr codec {name!r}; expected {sorted(table)}")
    return table[key]()


#: The float dtype a scale_offset-encoded array declares. float32 holds a 16-bit
#: packed count times its scale without loss, so the round-trip is exact.
_UNPACKED_DTYPE = "float32"


def _scale_offset_filters(scale_factor: float, add_offset: float, stored_dtype: str):
    """Return the ``scale_offset`` + ``cast_value`` chain for a CF-packed array.

    CF says ``physical = stored * scale_factor + add_offset``. Zarr's
    ``scale_offset`` codec encodes ``(value - offset) * scale``, so the CF scale
    inverts to ``scale = 1 / scale_factor`` (the same mapping EOPF's
    ``scale_offset_from_cf`` uses). That codec alone cannot change the dtype —
    the spec requires it to use "the arithmetic semantics of the input array's
    data type" — so ``cast_value`` is chained after it to put the packed integer
    on disk. Together they are the modern replacement for the legacy
    ``numcodecs.fixedscaleoffset`` ``astype``.
    """
    from zarr.codecs import CastValue, ScaleOffset

    return [
        ScaleOffset(offset=add_offset, scale=1.0 / scale_factor),
        CastValue(data_type=stored_dtype),
    ]


def _levels(data, n_levels: int, fill_value: float | None = None) -> list:
    """Return the multiscale pyramid: the base array plus ``n_levels`` coarsenings.

    Each extra level halves both spatial dims (mean-coarsened, trimming a ragged
    edge), the Zarr analogue of COG overviews. ``n_levels == 0`` returns just the
    base array.

    When ``fill_value`` is known the mean is taken over the *valid* pixels of each
    2x2 block, and a block that is entirely fill stays fill. Averaging a fill
    pixel as though it were data drags the overview towards the fill value — with
    MAJA's -10000 that darkens every swath edge — and GDAL already excludes
    nodata when it builds COG overviews, so doing it here keeps the two arms
    comparable.
    """
    import numpy as np

    arrays = [data]
    cur = data
    for _ in range(max(0, int(n_levels))):
        h, w = cur.shape
        if h < 2 or w < 2:
            break
        trimmed = cur[: h - (h % 2), : w - (w % 2)]
        blocks = trimmed.reshape(trimmed.shape[0] // 2, 2, trimmed.shape[1] // 2, 2)
        if fill_value is None:
            cur = blocks.mean(axis=(1, 3)).astype(data.dtype)
        else:
            valid = blocks != fill_value
            n_valid = valid.sum(axis=(1, 3))
            total = np.where(valid, blocks, 0).sum(axis=(1, 3), dtype="float64")
            cur = np.where(
                n_valid > 0, total / np.maximum(n_valid, 1), fill_value
            ).astype(data.dtype)
        arrays.append(np.ascontiguousarray(cur))
    return arrays


def _write_sharded(
    store: str,
    data,
    *,
    chunk: tuple[int, int],
    shard: tuple[int, int],
    codec: str,
    crs_wkt: str = "",
    geotransform: str = "",
    multiscale_levels: Any = "auto",
    fill_value: float | None = None,
    scale_factor: float | None = None,
    add_offset: float | None = None,
    scale_offset: bool = False,
    standard_name: str | None = None,
    group: str | None = None,
    var_name: str = DATA_VAR,
    write_shared: bool = True,
) -> None:
    """Write a 2D array to ``store`` as a sharded, multiscale GeoZarr v3 store.

    Each pyramid level is its own integer-named group holding the array
    (``<level>/{var_name}``, native resolution at ``0``), and the group that
    describes the pyramid — ``store`` itself by default, or ``group`` when
    given (a bundled write's grid subtree, #102) — carries the Zarr
    ``multiscales`` / ``spatial`` / ``geo-proj`` conventions (see
    :mod:`~cng_benchmark.formats.geozarr_multiscales`). Every level carries its
    *own* affine transform, grid description and cell-centre coordinates, so a
    reader can georeference an overview without deriving it. ``multiscale_levels
    == 0`` degrades to the flat store — the array alone under ``var_name`` at
    the pyramid group, no pyramid — which is the comparison case, not the
    default.

    Georeferencing lives in the conventions and nowhere else: no CF
    ``spatial_ref`` grid-mapping variable, no ``grid_mapping`` attribute, and no
    ``_ARRAY_DIMENSIONS`` (Zarr v3's own ``dimension_names`` names the axes).
    Pure ``xarray`` + ``zarr`` + ``numpy`` — no rioxarray — so it is CI-testable.

    ``fill_value`` declares the no-data value, ``scale_factor`` / ``add_offset``
    the CF packing that recovers physical units (``physical = stored *
    scale_factor + add_offset``). ``scale_offset`` chooses how that packing is
    encoded — as CF attributes, or as the codec chain (see
    :class:`GeoZarrParams`) — and :func:`_data_var` / :func:`_encoding` carry it
    out. ``standard_name`` is the CF name of the quantity the pixels hold.

    ``group``/``var_name``/``write_shared`` are the bundled-write seam (#102):
    a batched adapter shares one grid's coordinate arrays and pyramid metadata
    across several components by calling this once per component, with the
    *first* component of a grid using the defaults (``write_shared=True``,
    which writes the coords and grid attrs) and every later component of the
    *same* grid passing ``write_shared=False`` (only its own array, added as a
    sibling — no coords, no attrs rewritten) plus its own ``var_name``. A
    non-batched call (the default ``group=None``, ``var_name=DATA_VAR``,
    ``write_shared=True``) is byte-for-byte what this function has always done.
    """
    import numpy as np
    import xarray as xr

    depth = _multiscale_depth(multiscale_levels, data.shape)
    # Coarsen in the *packed* domain: the fill compares exactly there, which it
    # would not after a float multiply.
    levels = _levels(data, depth, fill_value)
    compressors = _compressor(codec)
    gt = ms.parse_geotransform(geotransform)
    y_attrs, x_attrs = ms.coordinate_attrs(crs_wkt)

    packed_dtype = str(data.dtype)
    offset = add_offset or 0.0
    use_codec = scale_offset and scale_factor is not None
    filters = None
    array_fill = fill_value

    if use_codec:
        # Re-express the counts as physical floats and let the codec chain pack
        # them back down on write, so the array *declares* physical units while
        # the packed integer is what occupies the shard.
        def _physical(arr):
            return (arr.astype(_UNPACKED_DTYPE) * scale_factor + offset).astype(
                _UNPACKED_DTYPE
            )

        levels = [np.ascontiguousarray(_physical(level)) for level in levels]
        filters = _scale_offset_filters(scale_factor, offset, packed_dtype)
        if fill_value is not None:
            # Zarr returns `fill_value` verbatim for an unwritten chunk — it does
            # not travel through the codecs — so it has to be stated in the
            # array's declared (physical) units, via the same expression as the
            # data so the two match exactly.
            array_fill = float(_physical(np.array([fill_value], dtype=packed_dtype))[0])
        else:
            # A float array defaults to a NaN fill, which `cast_value` cannot
            # cast down to the packed integer dtype. Pin it to 0 — the same fill
            # the un-encoded arm gets when no no-data is declared.
            array_fill = 0.0

    def _level_gt(level: int) -> tuple[float, ...] | None:
        """The geotransform of level ``level`` — the native one, coarsened."""
        return None if gt is None else ms.decimate(gt, 2**level)

    def _data_var(level: int, level_data) -> xr.DataArray:
        da = xr.DataArray(level_data, dims=("y", "x"))
        level_gt = _level_gt(level)
        # The array describes its own grid, not just its parent group's: xarray
        # drops a dataset's attributes when a variable is selected out of it, so
        # a client holding the array alone (a tile server addressing a variable)
        # would otherwise have nothing to georeference with.
        if level_gt is None:
            da.attrs.update(ms.dimensions_attrs())
        else:
            da.attrs.update(ms.grid_attrs(crs_wkt, level_gt, level_data.shape))
        if standard_name:
            da.attrs["standard_name"] = standard_name
        # Without the codec, CF packing travels as *attributes* — never as
        # xarray encoding keys, where xarray would apply the inverse transform
        # on write and overflow the int16 counts. With the codec the array
        # already reads back physical, so emitting them too would make a CF
        # reader scale a second time.
        if not use_codec:
            if scale_factor is not None:
                da.attrs["scale_factor"] = scale_factor
            if add_offset is not None:
                da.attrs["add_offset"] = add_offset
        return da

    def _level_dataset(level: int, level_data) -> xr.Dataset:
        """One pyramid level: the array, its grid attrs, and (when sharing) coords."""
        ds = xr.Dataset({var_name: _data_var(level, level_data)})
        level_gt = _level_gt(level)
        if level_gt is None:
            return ds
        if write_shared:
            coords = ms.level_coords(level_gt, level_data.shape)
            if coords is not None:
                y, x = coords
                ds = ds.assign_coords(y=("y", y, y_attrs), x=("x", x, x_attrs))
        # Every component of a bundled write's group re-declares the same grid
        # attrs (identical across components on one grid) on this same level
        # group's `to_zarr(mode="a", ...)` call -- an `xr.Dataset` with no
        # `.attrs` silently *wipes* a group's existing attributes rather than
        # leaving them alone (verified empirically for #102), and for a flat
        # (unpyramided) bundled store this level group *is* the pyramid group
        # every component's write lands on, not just the first's.
        ds.attrs.update(ms.grid_attrs(crs_wkt, level_gt, level_data.shape))
        return ds

    def _encoding(level_data) -> dict:
        c = _fit_chunk(chunk, level_data.shape)
        s = _fit_shard(shard, c, level_data.shape)
        enc: dict = {"chunks": c, "shards": s, "compressors": compressors}
        if filters is not None:
            enc["filters"] = filters
        # The two are not interchangeable: `fill_value` is the Zarr v3 array fill
        # in zarr.json (what an unwritten chunk reads as), `_FillValue` the CF
        # attribute a reader actually masks on. Declaring only the former leaves
        # the no-data invisible to xarray/rioxarray/GDAL — and declaring the
        # latter when the source named no no-data would mask valid zeros.
        if array_fill is not None:
            enc["fill_value"] = array_fill
        if fill_value is not None:
            enc["_FillValue"] = array_fill
        return {var_name: enc}

    def _level_group(level: int, flat: bool) -> str | None:
        """The zarr group path for pyramid level ``level``, nested under ``group``."""
        lvl = None if flat else str(level)
        if group is None:
            return lvl
        return group if lvl is None else f"{group}/{lvl}"

    # A write into an existing bundled store must never truncate a sibling
    # grid's subtree, and a fresh, not-yet-existing store must still work —
    # "a" (create-if-missing, add/override only the addressed group) is safe
    # either way (verified empirically for #102); "w" stays reserved for the
    # first write of a non-batched store, matching this function's original,
    # still-tested behaviour exactly.
    def _mode(is_first_level: bool) -> Literal["w", "a"]:
        if group is None and write_shared and is_first_level:
            return "w"
        return "a"

    if len(levels) == 1:
        # Flat store: no pyramid to describe, so the array sits at the pyramid
        # group and that group carries only its grid and CRS description.
        ds = _level_dataset(0, levels[0])
        ds.to_zarr(
            store,
            mode=_mode(True),
            group=_level_group(0, flat=True),
            zarr_format=3,
            consolidated=False,
            encoding=_encoding(levels[0]),
        )
        return

    # Multiscale: each level is its own group "0".."N", native resolution at 0.
    for i, level_data in enumerate(levels):
        _level_dataset(i, level_data).to_zarr(
            store,
            mode=_mode(i == 0),
            group=_level_group(i, flat=False),
            zarr_format=3,
            consolidated=False,
            encoding=_encoding(level_data),
        )
    if write_shared:
        # The pyramid group is what declares the pyramid: which group holds
        # each level, how it was derived, and where every level sits on the
        # ground. Written once per grid, by its first component.
        root_attrs = ms.group_attrs(crs_wkt, gt, [tuple(lv.shape) for lv in levels])
        xr.Dataset(attrs=root_attrs).to_zarr(
            store, mode="a", group=group, zarr_format=3, consolidated=False
        )


def _shard_data_files(
    store: str, *, group: str | None = None, var_name: str | None = None
) -> list[str]:
    """Return shard object paths (chunk data under ``c/``) in ``store``.

    Zarr v3 lays an array's chunk/shard data out under ``<array>/c/...``.
    ``var_name`` scopes the match to one array's own shards (used for a
    component's *own* structural layout, see :func:`describe_store_layout`);
    ``None`` matches *every* array's shards, coordinate arrays (``x``/``y``)
    included — they are real physical objects the store puts on S3, and the
    undercount they were previously excluded from was exactly the harness/
    reality mismatch #102 is about: CNES's own S3 inspection counted them
    (the "min 82 B" object in the finding that opened #102 *is* a coordinate
    chunk), so this harness's own object-size profile has to as well. ``group``
    further scopes to one grid's subtree (a bundled store's ``grid0``,
    ``grid1``, …). Excludes ``zarr.json`` metadata either way.
    """
    prefix = f"/{group}/" if group else None
    files: list[str] = []
    for root, _dirs, names in os.walk(store):
        for n in names:
            if n == "zarr.json":
                continue
            path = os.path.join(root, n)
            rel = "/" + os.path.relpath(path, store).replace(os.sep, "/")
            if prefix is not None and not rel.startswith(prefix):
                continue
            segments = rel.split("/")
            if "c" not in segments:
                continue
            array_name = segments[segments.index("c") - 1]
            if var_name is not None and array_name != var_name:
                continue
            files.append(path)
    return files


def enumerate_store_objects(store: str) -> list[int]:
    """Return the byte sizes of every shard object in ``store``, coordinate
    arrays included — the store's real, whole physical object population (the
    number a size/tier profile has to be honest about; #102).
    """
    return [os.path.getsize(p) for p in _shard_data_files(store)]


def _overview_bytes(
    store: str, *, group: str | None = None, var_name: str = DATA_VAR
) -> int:
    """Return the bytes one array's *overview* levels occupy (below level 0).

    The pyramid is what a zoomed-out tile is served from, and it is also what the
    store pays for it — the same trade a COG makes with its overviews. Splitting
    it out keeps the size comparison between the two arms readable. Scoped to
    one component's own array via ``group``/``var_name`` (#102) — a bundled
    store's shared coordinate-array bytes are never an "overview" of anyone's
    data, so they are excluded by construction (``_shard_data_files`` already
    scopes to ``var_name``).
    """
    total = 0
    prefix = f"{group}/" if group else ""
    for path in _shard_data_files(store, group=group, var_name=var_name):
        rel = os.path.relpath(path, store).replace(os.sep, "/")
        rel_in_group = rel[len(prefix) :] if prefix and rel.startswith(prefix) else rel
        head = rel_in_group.split("/", 1)[0]
        if head.isdigit() and int(head) > 0:
            total += os.path.getsize(path)
    return total


def _read_array_meta(
    store: str, *, group: str | None = None, var_name: str = DATA_VAR
) -> dict:
    """Read one array's chunk/shard shape, codec and shard count from ``store``.

    Opens ``store``'s finest array named ``var_name`` (the array itself, or
    group ``0`` for a multiscale store) — at the store root by default, or
    under ``group`` for one component of a bundled store (#102, via zarr's own
    ``path=`` group navigation) — and reports its chunk grid, shard grid
    (chunks-per-shard), the configured compressor name, and how many shard
    objects that one array's own chain of levels produced.

    Also reports whether the ``scale_offset`` codec chain is active and which
    dtype actually reaches disk. That distinction matters for the compression
    ratio: a scale_offset array *declares* float32 but stores the packed integer,
    so sizing "uncompressed" from the declared dtype would double the baseline
    and flatter the ratio against the COG arm.
    """
    import zarr

    root = zarr.open_group(store, mode="r", path=group or "")
    if var_name in root:
        arrays = [root[var_name]]
    else:  # multiscale: levels live in integer-named groups
        level_keys = sorted((k for k in root.group_keys()), key=lambda k: int(k))
        arrays = [root[k][var_name] for k in level_keys]
    levels = len(arrays) - 1
    arr = arrays[0]

    chunk = list(arr.chunks)
    shards = list(arr.shards) if arr.shards is not None else list(arr.chunks)
    chunks_per_shard = 1
    for s, c in zip(shards, chunk, strict=True):
        chunks_per_shard *= max(1, s // c)
    import numpy as np

    codec = "none"
    for c in getattr(arr, "compressors", ()) or ():
        codec = type(c).__name__.replace("Codec", "").lower() or codec

    # `filters` are the array-to-array codecs: scale_offset unpacks, cast_value
    # names the dtype that reaches disk. Read them through `to_dict()` — the
    # on-disk spec form — rather than off the instances, whose attribute names
    # and dtype objects are zarr-internal.
    scale_offset = False
    stored_dtype = str(arr.dtype)
    for f in getattr(arr, "filters", ()) or ():
        spec = f.to_dict() if hasattr(f, "to_dict") else {}
        name = spec.get("name", "")
        if name == "scale_offset":
            scale_offset = True
        elif name == "cast_value":
            cast_to = spec.get("configuration", {}).get("data_type")
            stored_dtype = str(cast_to) if cast_to else stored_dtype
    try:
        itemsize = np.dtype(stored_dtype).itemsize
    except TypeError:  # an exotic cast target; fall back to the declared dtype
        stored_dtype = str(arr.dtype)
        itemsize = np.dtype(stored_dtype).itemsize
    # Every level's raw bytes, not just the base array's: the stored size counts
    # the whole pyramid, so a base-only baseline would report the pyramid's cost
    # as if it were poor compression.
    uncompressed_bytes = int(sum(int(np.prod(a.shape)) for a in arrays) * itemsize)
    return {
        "scale_offset": scale_offset,
        "stored_dtype": stored_dtype,
        "chunk_shape": chunk,
        "shard_shape": shards,
        "chunks_per_shard": int(chunks_per_shard),
        "codec": codec,
        "multiscale_levels": int(levels),
        "shard_count": len(_shard_data_files(store, group=group, var_name=var_name)),
        "uncompressed_bytes": uncompressed_bytes,
    }


def describe_store_layout(
    store: str, name: str, *, group: str | None = None, var_name: str = DATA_VAR
) -> GeoZarrLayout:
    """Return the :class:`GeoZarrLayout` of one array in the GeoZarr store at ``store``.

    ``group``/``var_name`` address one component of a bundled store (#102);
    the defaults describe the whole (non-batched) store, unchanged from before.
    ``size_bytes`` is that array's *own* shard bytes only — never the shared
    ``x``/``y`` coordinate bytes a bundled store's components hold in common —
    so a component's reported size is what it alone costs, not an arbitrary
    slice of shared overhead.
    """
    meta = _read_array_meta(store, group=group, var_name=var_name)
    total = sum(
        os.path.getsize(p)
        for p in _shard_data_files(store, group=group, var_name=var_name)
    )
    uncompressed = meta.pop("uncompressed_bytes", 0)
    compression_ratio = uncompressed / total if total else 0.0
    return GeoZarrLayout(
        name=name,
        size_bytes=total,
        compression_ratio=compression_ratio,
        overview_bytes=_overview_bytes(store, group=group, var_name=var_name),
        grid_group=group,
        **meta,
    )


def _require_rioxarray():
    try:
        import rioxarray  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "GeoZarr conversion requires the 'geozarr' extra; install with "
            "`uv sync --extra geozarr` (or `pip install cng-benchmark[geozarr]`)"
        ) from exc


def _read_component_array(source: str, params: dict, opts: GeoZarrParams) -> dict:
    """Read ``source``'s first band as CRS+geotransform-tagged pixel data.

    Shared by :meth:`GeoZarrAdapter.convert` and :meth:`GeoZarrAdapter.convert_batch`
    (#102) — the same per-source rioxarray read and nodata/scale/offset/
    standard_name precedence (an explicit param wins; otherwise the source's
    own embedded value), whether there is one source or several. ``opts`` is
    the *already-merged* :class:`GeoZarrParams` for this source, so the batched
    caller can pass one parsed per component (its own nodata/scale_factor/
    standard_name folded in) while the shared levers (chunk/shard/codec/…)
    stay whatever the run configured.
    """
    import numpy as np
    import rioxarray

    da = rioxarray.open_rasterio(source, masked=False)
    # Reduce to a single 2D (y, x) band; rioxarray yields (band, y, x).
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    data = np.ascontiguousarray(da.values)
    crs_wkt = da.rio.crs.to_wkt() if da.rio.crs else ""
    t = da.rio.transform()
    geotransform = " ".join(str(v) for v in (t.c, t.a, t.b, t.f, t.d, t.e))

    # An explicit param wins; otherwise fall back to what the source
    # embedded (rioxarray surfaces a GDAL band's nodata/scale/offset as
    # `_FillValue`, `scale_factor` and `add_offset` attributes). MAJA
    # reflectance declares none of them in the file, so the dataset reader
    # supplies them — see `SourceObject.nodata` / `.scale_factor`.
    def _param(key: str, source_value) -> float | None:
        value = params.get(key, source_value)
        return None if value is None else float(value)

    fill_value = _param("nodata", da.rio.nodata)
    scale_factor = _param("scale_factor", da.attrs.get("scale_factor"))
    add_offset = _param("add_offset", da.attrs.get("add_offset"))
    # rioxarray reports the identity packing (1.0 / 0.0) for a band that
    # carries none; writing it out would only add noise.
    if scale_factor == 1.0:
        scale_factor = None
    if add_offset == 0.0:
        add_offset = None

    # Same precedence for the CF quantity name, which a source read from a
    # CF file (SWOT's netCDF variables) already carries and a delivered
    # GeoTIFF (MAJA, S1 RTC) does not — there the reader names it.
    standard_name = opts.standard_name or da.attrs.get("standard_name")

    return {
        "data": data,
        "crs_wkt": crs_wkt,
        "geotransform": geotransform,
        "fill_value": fill_value,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "standard_name": standard_name,
    }


@FORMATS.register("geozarr")
class GeoZarrAdapter(FormatAdapter):
    name = "geozarr"
    object_kind = ObjectKind.ZARR_STORE
    supports_batch = True

    def __init__(self) -> None:
        # Batch-time bookkeeping: which grid group each component's array
        # landed in, keyed by target store path — describe_layout and
        # component_locator read it back for a target this instance just
        # wrote via convert_batch (#102, mirrors the FlatGeobuf
        # null_geometry bookkeeping pattern from #98).
        self._batch_grid_by_target: dict[str, dict[str, str]] = {}

    def target_basename(self) -> str:
        return "geozarr.zarr"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (a GDAL/rioxarray-readable raster) to a sharded store.

        Reads the source's first band into a 2D array (with its CRS and
        geotransform) and writes it as a Zarr v3 store whose chunk/shard shape,
        codec and multiscale depth come from :class:`GeoZarrParams`.
        """
        _require_rioxarray()
        opts = GeoZarrParams.model_validate(params)
        read = _read_component_array(source, params, opts)
        data = read["data"]

        chunk = _fit_chunk(_spatial_pair(opts.chunk_shape, DEFAULT_CHUNK), data.shape)
        shard = _fit_shard(
            _spatial_pair(opts.shard_shape, DEFAULT_SHARD), chunk, data.shape
        )
        _write_sharded(
            target,
            data,
            chunk=chunk,
            shard=shard,
            codec=opts.codec,
            crs_wkt=read["crs_wkt"],
            geotransform=read["geotransform"],
            multiscale_levels=opts.multiscale_levels,
            fill_value=read["fill_value"],
            scale_factor=read["scale_factor"],
            add_offset=read["add_offset"],
            scale_offset=opts.scale_offset,
            standard_name=read["standard_name"],
        )

    def convert_batch(
        self, sources: list[SourceObject], target: str, params: dict[str, Any]
    ) -> None:
        """Convert every one of ``sources`` into one bundled store at ``target`` (#102).

        Reads each source the same way :meth:`convert` reads its one, groups
        them by grid equality (:func:`~cng_benchmark.formats.grid.group_by_grid`
        — shape, geotransform, CRS), and writes one grid's group of components
        sharing the grid's coordinate arrays and pyramid metadata (written once,
        by the first component) with each component as a sibling array named
        for itself. A product whose components span more than one grid gets one
        ``grid0``/``grid1``/… subtree per grid, all in this one store — bundling
        never silently drops or misplaces a component onto the wrong grid.

        When those grids sit at more than one *real* native pixel size — S2/
        MAJA's 10 m reflectance next to its 20 m masks, per #107 — grid
        equality already separates them one group per native resolution for
        free; what this method adds is anchoring the pyramid on that fact:
        only the coarsest native group gets a synthetic overview pyramid
        (:meth:`_resolution_chain`), every finer native group is written flat,
        and one combined chain doc describing the whole 10 m -> 20 m -> …
        anchored pyramid is written once at the store root
        (:func:`~cng_benchmark.formats.geozarr_multiscales.chain_group_attrs`)
        — a naive per-group pyramid would otherwise have every native group
        independently fabricate its own unrelated overview levels, the
        fragmentation #107 is about. A single-grid bundle (one native
        resolution) is unaffected: every group keeps its configured
        ``multiscale_levels`` exactly as before.
        """
        _require_rioxarray()
        import xarray as xr

        reads: dict[str, dict] = {}
        keys: list[GridKey] = []
        for src in sources:
            component_params = dict(params)
            for key, value in (
                ("nodata", src.nodata),
                ("scale_factor", src.scale_factor),
                ("standard_name", src.standard_name),
            ):
                if value is not None:
                    component_params.setdefault(key, value)
            component_opts = GeoZarrParams.model_validate(component_params)
            read = _read_component_array(src.uri, component_params, component_opts)
            reads[src.name] = read
            gt_tuple = ms.parse_geotransform(read["geotransform"]) or ()
            keys.append(
                GridKey(
                    name=src.name,
                    shape=tuple(read["data"].shape),
                    geotransform=gt_tuple,
                    crs=read["crs_wkt"],
                )
            )

        # The shared, run-level levers (chunk/shard/codec/multiscale/scale_offset)
        # are identical for every component — only nodata/scale_factor/
        # standard_name vary per source, already folded into each `read` above.
        opts = GeoZarrParams.model_validate(params)

        groups = group_by_grid(keys)
        grid_ids = {name: f"grid{i}" for i, grp in enumerate(groups) for name in grp}
        group_by_id = {f"grid{i}": grp for i, grp in enumerate(groups)}

        # S2/MAJA-style anchoring (#107): when the bundle's grid-groups sit at
        # more than one *real* native pixel size, a finer group's own overview
        # levels would be redundant — the next coarser native group already
        # is the next zoomed-out tier — so only the coarsest group keeps a
        # synthetic pyramid past its own native resolution. A single-grid
        # bundle (today's common case, e.g. one product's same-grid variables)
        # is unaffected: every group keeps `opts.multiscale_levels` unchanged.
        pixel_sizes: dict[str, float] | None = {}
        for grid_id, names in group_by_id.items():
            gt = ms.parse_geotransform(reads[names[0]]["geotransform"])
            if gt is None:
                pixel_sizes = None
                break
            pixel_sizes[grid_id] = ms.pixel_size(gt)
        is_multi_resolution = pixel_sizes is not None and (
            len({round(p, 6) for p in pixel_sizes.values()}) > 1
        )
        coarsest_pixel_size = max(pixel_sizes.values()) if pixel_sizes else None

        for group_names in groups:
            grid_id = grid_ids[group_names[0]]
            group_levels = opts.multiscale_levels
            if is_multi_resolution and pixel_sizes[grid_id] < coarsest_pixel_size:  # type: ignore[operator]
                group_levels = 0
            for i, name in enumerate(group_names):
                read = reads[name]
                data = read["data"]
                chunk = _fit_chunk(
                    _spatial_pair(opts.chunk_shape, DEFAULT_CHUNK), data.shape
                )
                shard = _fit_shard(
                    _spatial_pair(opts.shard_shape, DEFAULT_SHARD), chunk, data.shape
                )
                _write_sharded(
                    target,
                    data,
                    chunk=chunk,
                    shard=shard,
                    codec=opts.codec,
                    crs_wkt=read["crs_wkt"],
                    geotransform=read["geotransform"],
                    multiscale_levels=group_levels,
                    fill_value=read["fill_value"],
                    scale_factor=read["scale_factor"],
                    add_offset=read["add_offset"],
                    scale_offset=opts.scale_offset,
                    standard_name=read["standard_name"],
                    group=grid_id,
                    var_name=name,
                    write_shared=(i == 0),
                )

        # Harness-internal bookkeeping only — not part of the published
        # zarr_conventions — so a reader (or describe_layout) doesn't have to
        # re-derive grid equality itself.
        root_attrs: dict[str, Any] = {"cng_benchmark:components": grid_ids}
        if is_multi_resolution:
            assert pixel_sizes is not None
            chain = self._resolution_chain(target, group_by_id, reads, pixel_sizes)
            finest_id = min(pixel_sizes, key=lambda gid: pixel_sizes[gid])  # type: ignore[index]
            crs_wkt = reads[group_by_id[finest_id][0]]["crs_wkt"]
            root_attrs.update(ms.chain_group_attrs(crs_wkt, chain))

        self._batch_grid_by_target[target] = grid_ids
        # A second `to_zarr(mode="a")` write to the *same* group with a
        # different `.attrs` silently replaces the group's existing attrs
        # rather than merging (the bug fixed in #103 for a bundled flat
        # store's grid attrs) — so the bookkeeping and the resolution-chain
        # doc are folded into one `.attrs` dict and written in one call here.
        xr.Dataset(attrs=root_attrs).to_zarr(
            target, mode="a", zarr_format=3, consolidated=False
        )

    def _resolution_chain(
        self,
        target: str,
        group_by_id: dict[str, list[str]],
        reads: dict[str, dict],
        pixel_sizes: dict[str, float],
    ) -> list[ms.ChainEntry]:
        """Build the cross-group anchored-pyramid chain for a multi-resolution
        bundle (#107): finest native tier first.

        Every non-coarsest grid-group contributes one flat entry (its "next
        level down" is the next native tier's group, not a synthetic
        downsample of its own array). The coarsest grid-group contributes its
        own base entry *and* one entry per overview level it actually wrote —
        read back from the store this call just finished writing, rather than
        recomputed, so the chain's metadata can never drift from what is
        actually on disk.
        """
        import zarr

        ordered = sorted(pixel_sizes, key=lambda gid: pixel_sizes[gid])
        coarsest_id = ordered[-1]

        entries: list[ms.ChainEntry] = []
        for grid_id in ordered[:-1]:
            rep = reads[group_by_id[grid_id][0]]
            gt = ms.parse_geotransform(rep["geotransform"])
            assert gt is not None  # guaranteed by the pixel-size pass below
            entries.append(
                ms.ChainEntry(asset=grid_id, gt=gt, shape=tuple(rep["data"].shape))
            )

        coarsest_rep = reads[group_by_id[coarsest_id][0]]
        coarsest_gt = ms.parse_geotransform(coarsest_rep["geotransform"])
        assert coarsest_gt is not None
        var_name = group_by_id[coarsest_id][0]
        root = zarr.open_group(target, mode="r", path=coarsest_id)
        if var_name in root:
            # Flat: this bundle's coarsest tier got no overview levels either
            # (e.g. `multiscale_levels: 0`, or it is already tile-sized).
            entries.append(
                ms.ChainEntry(
                    asset=coarsest_id,
                    gt=coarsest_gt,
                    shape=tuple(coarsest_rep["data"].shape),
                )
            )
        else:
            for lvl_key in sorted((k for k in root.group_keys()), key=int):
                level = int(lvl_key)
                shape = tuple(root[lvl_key][var_name].shape)
                entries.append(
                    ms.ChainEntry(
                        asset=f"{coarsest_id}/{lvl_key}",
                        gt=ms.decimate(coarsest_gt, 2**level),
                        shape=shape,
                    )
                )
        return entries

    def component_locator(self, target: str, name: str) -> str | None:
        """The grid group ``name``'s array lives under, for a batched ``target``."""
        return self._batch_grid_by_target.get(target, {}).get(name)

    def describe_grouping_lever(self) -> str:
        return "Zarr v3 chunk and shard shape"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the sizes of every shard data object the store produced."""
        return enumerate_store_objects(target)

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[GeoZarrLayout]:
        """Return the produced store's chunk/shard layout.

        One array for a non-batched target; one per component, each scoped to
        its own grid group, for a target :meth:`convert_batch` just wrote.
        """
        grid_ids = self._batch_grid_by_target.get(target)
        if grid_ids is None:
            return [describe_store_layout(target, name or self.name)]
        return [
            describe_store_layout(target, comp_name, group=grid_id, var_name=comp_name)
            for comp_name, grid_id in grid_ids.items()
        ]
