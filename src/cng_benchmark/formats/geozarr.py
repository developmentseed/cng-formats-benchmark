"""GeoZarr v3 adapter — 2D, per-component, sharded.

The grouping lever for Zarr v3 is its chunk and shard shape: the chunk is the
addressable (range-read) unit, and a *shard* packs many chunks into one stored
object, so shard size is the knob that lifts the mean object size into a storage
tier (ADR 0001). This adapter converts one baseline raster to one sharded 2D
GeoZarr store, the per-component analogue of the COG arm — it flows through the
same runner paths as COG. Time-stacking the source scenes into a 3D cube is a
separate, deferred concern (see the M2.5 plan): nothing here stacks.

The store-writing core (:func:`_write_sharded`, :func:`enumerate_store_objects`,
:func:`describe_store_layout`) depends only on ``zarr`` + ``xarray`` + ``numpy``,
so the sharding-lever / enumerate / layout logic is unit-testable on synthetic
in-memory arrays. Only :meth:`GeoZarrAdapter.convert`'s source read needs
``rioxarray`` (the ``geozarr`` extra), imported lazily.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.models import GeoZarrLayout
from cng_benchmark.registry import FORMATS

#: The single data variable each per-component store holds (titiler-xarray and the
#: read metric address the array by this name).
DATA_VAR = "data"

#: Defaults when the config carries no lever value (spatial y, x).
DEFAULT_CHUNK = (1024, 1024)
DEFAULT_SHARD = (2048, 2048)


class GeoZarrParams(BaseModel):
    """Zarr v3 sharding levers, parsed from ``config.params``.

    ``chunk_shape`` / ``shard_shape`` accept a 2D ``[y, x]`` or a 3D
    ``[t, y, x]`` shape (the trailing two, spatial, dims are used — a leading time
    dim is ignored in the 2D regime) and tolerate a swept *list of shapes*, taking
    the first so a swept lever degrades to a single run, mirroring COG's
    ``block_size``. ``codec`` is the per-chunk compressor (``zstd`` default;
    ``none`` for raw). ``multiscale_levels`` is the overview-pyramid depth (0 = the
    base array only).
    """

    model_config = ConfigDict(extra="ignore")

    chunk_shape: Any = None
    shard_shape: Any = None
    codec: str = "zstd"
    multiscale_levels: int = 0


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
    multiscale_levels: int = 0,
    fill_value: float | None = None,
    scale_factor: float | None = None,
    add_offset: float | None = None,
) -> None:
    """Write a 2D array to ``store`` as a sharded GeoZarr v3 store.

    With no pyramid (``multiscale_levels <= 0``) the array is written under
    :data:`DATA_VAR` at the store root. With a pyramid, each level is its own
    integer-named group holding the array (``<level>/{DATA_VAR}``, finest at
    ``0``), with a ``multiscales`` attribute on the root listing the level paths.
    CRS and the GDAL geotransform travel in a CF ``spatial_ref`` grid-mapping
    variable so a reader (rioxarray, GDAL, titiler-xarray) can georeference the
    array. Pure ``xarray`` + ``zarr`` + ``numpy`` — no rioxarray — so it is
    CI-testable.

    ``fill_value`` declares the no-data value, ``scale_factor`` / ``add_offset``
    the CF packing that recovers physical units (``physical = stored *
    scale_factor + add_offset``) — see :func:`_data_var` and :func:`_encoding`
    for how each is written.
    """
    import xarray as xr

    levels = _levels(data, multiscale_levels, fill_value)
    compressors = _compressor(codec)

    def _grid_mapping() -> xr.DataArray:
        ref = xr.DataArray(0).astype("int32")
        ref.attrs.update(
            crs_wkt=crs_wkt, spatial_ref=crs_wkt, GeoTransform=geotransform
        )
        return ref

    def _data_var(level_data) -> xr.DataArray:
        da = xr.DataArray(level_data, dims=("y", "x"))
        da.attrs["grid_mapping"] = "spatial_ref"
        # CF packing travels as *attributes*, never as xarray encoding keys: in
        # encoding, xarray applies the inverse transform on write (dividing the
        # stored integers by scale_factor), which overflows an int16 DN array.
        # As attributes the DNs are stored verbatim — byte-comparable with the
        # COG arm — and a CF reader still recovers physical units. The GeoZarr
        # spec recommends exactly these CF attributes.
        if scale_factor is not None:
            da.attrs["scale_factor"] = scale_factor
        if add_offset is not None:
            da.attrs["add_offset"] = add_offset
        return da

    def _encoding(level_data) -> dict:
        c = _fit_chunk(chunk, level_data.shape)
        s = _fit_shard(shard, c, level_data.shape)
        enc: dict = {"chunks": c, "shards": s, "compressors": compressors}
        if fill_value is not None:
            # Both halves are needed: `fill_value` is the Zarr v3 array fill in
            # zarr.json, `_FillValue` the CF attribute a reader actually masks
            # on. Setting only the former leaves the no-data undeclared to
            # xarray/rioxarray/GDAL.
            enc["fill_value"] = fill_value
            enc["_FillValue"] = fill_value
        return {DATA_VAR: enc}

    if multiscale_levels <= 0:
        ds = xr.Dataset(
            {DATA_VAR: _data_var(levels[0]), "spatial_ref": _grid_mapping()}
        )
        ds.to_zarr(
            store,
            mode="w",
            zarr_format=3,
            consolidated=False,
            encoding=_encoding(levels[0]),
        )
        return

    # Multiscale: each level is its own group "0".."N"; the root records the paths.
    for i, level_data in enumerate(levels):
        ds = xr.Dataset(
            {DATA_VAR: _data_var(level_data), "spatial_ref": _grid_mapping()}
        )
        ds.to_zarr(
            store,
            mode="w" if i == 0 else "a",
            group=str(i),
            zarr_format=3,
            consolidated=False,
            encoding=_encoding(level_data),
        )
    paths = [{"path": str(i)} for i in range(len(levels))]
    root = xr.Dataset(attrs={"multiscales": [{"datasets": paths}]})
    root.to_zarr(store, mode="a", zarr_format=3, consolidated=False)


def _shard_data_files(store: str) -> list[str]:
    """Return the data array's shard object paths (its chunk data under ``c/``).

    Zarr v3 lays an array's chunk/shard data out under ``<array>/c/...``. Only the
    main :data:`DATA_VAR` array's shards are the tier-judged objects, so this keeps
    files under ``.../{DATA_VAR}/c/`` — matching both the root array
    (``{DATA_VAR}/c/...``) and each multiscale level (``<level>/{DATA_VAR}/c/...``)
    — and excludes ``zarr.json`` metadata and the scalar ``spatial_ref``
    grid-mapping variable, which would otherwise skew the size profile and
    ``shard_count``.
    """
    marker = f"/{DATA_VAR}/c/"
    files: list[str] = []
    for root, _dirs, names in os.walk(store):
        for n in names:
            if n == "zarr.json":
                continue
            path = os.path.join(root, n)
            rel = "/" + os.path.relpath(path, store).replace(os.sep, "/")
            if marker in rel:
                files.append(path)
    return files


def enumerate_store_objects(store: str) -> list[int]:
    """Return the byte sizes of every shard data object in ``store``."""
    return [os.path.getsize(p) for p in _shard_data_files(store)]


def _read_array_meta(store: str) -> dict:
    """Read the base array's chunk/shard shape, codec and shard count from ``store``.

    Opens the store's finest array (the root array, or group ``0`` for a multiscale
    store) and reports its chunk grid, shard grid (chunks-per-shard), the configured
    compressor name, and the number of shard objects across the whole store.
    """
    import zarr

    root = zarr.open_group(store, mode="r")
    levels = 0
    if DATA_VAR in root:
        arr = root[DATA_VAR]
    else:  # multiscale: levels live in integer-named groups
        level_keys = sorted((k for k in root.group_keys()), key=lambda k: int(k))
        levels = max(0, len(level_keys) - 1)
        arr = root[level_keys[0]][DATA_VAR]

    chunk = list(arr.chunks)
    shards = list(arr.shards) if arr.shards is not None else list(arr.chunks)
    chunks_per_shard = 1
    for s, c in zip(shards, chunk, strict=True):
        chunks_per_shard *= max(1, s // c)
    import numpy as np

    codec = "none"
    for c in getattr(arr, "compressors", ()) or ():
        codec = type(c).__name__.replace("Codec", "").lower() or codec
    uncompressed_bytes = int(np.prod(arr.shape) * np.dtype(str(arr.dtype)).itemsize)
    return {
        "chunk_shape": chunk,
        "shard_shape": shards,
        "chunks_per_shard": int(chunks_per_shard),
        "codec": codec,
        "multiscale_levels": int(levels),
        "shard_count": len(_shard_data_files(store)),
        "uncompressed_bytes": uncompressed_bytes,
    }


def describe_store_layout(store: str, name: str) -> GeoZarrLayout:
    """Return the :class:`GeoZarrLayout` of the GeoZarr store at ``store``."""
    meta = _read_array_meta(store)
    total = sum(enumerate_store_objects(store))
    uncompressed = meta.pop("uncompressed_bytes", 0)
    compression_ratio = uncompressed / total if total else 0.0
    return GeoZarrLayout(
        name=name, size_bytes=total, compression_ratio=compression_ratio, **meta
    )


@FORMATS.register("geozarr")
class GeoZarrAdapter(FormatAdapter):
    name = "geozarr"
    object_kind = ObjectKind.ZARR_STORE

    def target_basename(self) -> str:
        return "geozarr.zarr"

    def convert(self, source: str, target: str, params: dict[str, Any]) -> None:
        """Convert ``source`` (a GDAL/rioxarray-readable raster) to a sharded store.

        Reads the source's first band into a 2D array (with its CRS and
        geotransform) and writes it as a Zarr v3 store whose chunk/shard shape,
        codec and multiscale depth come from :class:`GeoZarrParams`.
        """
        try:
            import rioxarray  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
            raise RuntimeError(
                "GeoZarr conversion requires the 'geozarr' extra; install with "
                "`uv sync --extra geozarr` (or `pip install cng-benchmark[geozarr]`)"
            ) from exc
        import numpy as np
        import rioxarray

        opts = GeoZarrParams.model_validate(params)
        da = rioxarray.open_rasterio(source, masked=False)
        # Reduce to a single 2D (y, x) band; rioxarray yields (band, y, x).
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        data = np.ascontiguousarray(da.values)

        chunk = _fit_chunk(_spatial_pair(opts.chunk_shape, DEFAULT_CHUNK), data.shape)
        shard = _fit_shard(
            _spatial_pair(opts.shard_shape, DEFAULT_SHARD), chunk, data.shape
        )
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

        _write_sharded(
            target,
            data,
            chunk=chunk,
            shard=shard,
            codec=opts.codec,
            crs_wkt=crs_wkt,
            geotransform=geotransform,
            multiscale_levels=opts.multiscale_levels,
            fill_value=fill_value,
            scale_factor=scale_factor,
            add_offset=add_offset,
        )

    def describe_grouping_lever(self) -> str:
        return "Zarr v3 chunk and shard shape"

    def enumerate_objects(self, target: str) -> list[int]:
        """Return the sizes of every shard data object the store produced."""
        return enumerate_store_objects(target)

    def describe_layout(
        self, target: str, *, name: str | None = None
    ) -> list[GeoZarrLayout]:
        """Return the produced store's chunk/shard layout (one array)."""
        return [describe_store_layout(target, name or self.name)]
