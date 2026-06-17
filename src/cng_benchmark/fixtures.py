"""Synthetic fixture generation — test scaffolding, not benchmark logic.

The deployable stack must prove it works end-to-end without any external data,
so it needs a small, *valid* Cloud-Optimized GeoTIFF it can seed into object
storage. This module produces one in-memory: a few-hundred-KiB, tiled,
overview-bearing COG generated with rasterio + rio-cogeo (the ``cog`` extra).

It lives apart from :mod:`cng_benchmark.formats.cog` on purpose — generating a
disposable fixture is deployment scaffolding, whereas the format adapter's
conversion is the thing under test. The heavy geo imports are deferred so the
core harness stays installable without them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: Default fixture geometry — small but enough for internal tiling + overviews.
DEFAULT_SIZE = 512
DEFAULT_BLOCKSIZE = 256


def _require_geo():
    """Import the geo stack, raising a clear error if the ``cog`` extra is absent."""
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from rio_cogeo.cogeo import cog_translate
        from rio_cogeo.profiles import cog_profiles
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "fixture generation requires the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return np, rasterio, from_bounds, cog_translate, cog_profiles


def generate_cog_bytes(
    *,
    size: int = DEFAULT_SIZE,
    blocksize: int = DEFAULT_BLOCKSIZE,
    overview_levels: int = 2,
) -> bytes:
    """Generate a small, valid 3-band COG and return its bytes.

    The raster is a deterministic RGB gradient over a global EPSG:4326 extent,
    written as an internally tiled, deflate-compressed COG with overviews — a
    realistic-enough object to seed storage and exercise the read/display path.
    """
    np, rasterio, from_bounds, cog_translate, cog_profiles = _require_geo()

    rows = np.linspace(0, 255, size, dtype="uint8")
    cols = np.linspace(0, 255, size, dtype="uint8")
    red = np.tile(cols, (size, 1))
    green = np.tile(rows.reshape(-1, 1), (1, size))
    blue = ((red.astype("uint16") + green) // 2).astype("uint8")
    data = np.stack([red, green, blue])

    src_profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 3,
        "height": size,
        "width": size,
        "crs": "EPSG:4326",
        "transform": from_bounds(-180, -90, 180, 90, size, size),
    }
    output_profile = cog_profiles.get("deflate")
    output_profile.update(blockxsize=blocksize, blockysize=blocksize)

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.tif")
        dst_path = os.path.join(tmp, "cog.tif")
        with rasterio.open(src_path, "w", **src_profile) as src:
            src.write(data)
        cog_translate(
            src_path,
            dst_path,
            output_profile,
            overview_level=overview_levels,
            quiet=True,
        )
        return Path(dst_path).read_bytes()
