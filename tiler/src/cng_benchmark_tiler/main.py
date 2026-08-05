"""cng-benchmark-tiler application.

One FastAPI process, three tile routers, each bound to a different reader, so
the *only* variable between two display measurements is which reader served
them:

* ``/cog``      -- ``titiler.core.factory.TilerFactory``, rasterio/GDAL: the
  COG baseline.
* ``/zarr``     -- ``titiler.xarray.factory.TilerFactory``, stock xarray
  defaults (default indexes on): "TiTiler out of the box".
* ``/geozarr``  -- ``titiler.eopf.factory.TilerFactory``, ``GeoZarrReader``
  (``create_default_indexes=False``, zoom-matched multiscale level, cached
  datatree open).

``/geozarr`` reuses ``titiler.eopf``'s own factory wholesale -- the only
override is ``path_dependency``: that package's default resolves a STAC
collection/item path, which this bench has no use for, so it's swapped for
the same ``url=`` query-param resolver ``/cog`` uses. Every other route
(``tile``, ``info``, ``tilejson``, ``point``, ``preview``, ``part``, the
``variables``/``sel`` query contract) is untouched -- a binding, not a fork.

``/stac`` and ``/mosaicjson`` (present in ``titiler.application``) are
dropped to keep the surface small; TileMatrixSets, Algorithms and ColorMaps
are kept since tiling needs them.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import version as pkg_version

# `titiler.eopf.dependencies` (its STAC collection/item path resolver, which
# this app overrides via `path_dependency` and never calls) instantiates a
# required `DataStoreSettings` at *import* time. Satisfy it with a placeholder
# before importing anything from `titiler.eopf` -- the value is never read.
os.environ.setdefault("TITILER_EOPF_STORE_URL", "https://unused.invalid/")

import jinja2  # noqa: E402
import rasterio  # noqa: E402
import xarray  # noqa: E402
import zarr  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402
from starlette.templating import Jinja2Templates  # noqa: E402
from starlette_cramjam.middleware import CompressionMiddleware  # noqa: E402
from titiler.core.dependencies import DatasetPathParams  # noqa: E402
from titiler.core.errors import (  # noqa: E402
    DEFAULT_STATUS_CODES,
    add_exception_handlers,
)
from titiler.core.factory import (  # noqa: E402
    AlgorithmFactory,
    ColorMapFactory,
    TilerFactory,
    TMSFactory,
)
from titiler.core.middleware import CacheControlMiddleware  # noqa: E402
from titiler.core.utils import update_openapi  # noqa: E402
from titiler.eopf.factory import TilerFactory as GeoZarrTilerFactory  # noqa: E402
from titiler.eopf.reader import open_dataset as geozarr_open_dataset  # noqa: E402
from titiler.xarray.factory import TilerFactory as XarrayTilerFactory  # noqa: E402

from cng_benchmark_tiler.settings import ApiSettings  # noqa: E402

logging.getLogger("botocore.credentials").disabled = True
logging.getLogger("rasterio.session").setLevel(logging.ERROR)

settings = ApiSettings()

templates = Jinja2Templates(
    env=jinja2.Environment(
        autoescape=jinja2.select_autoescape(["html"]),
        loader=jinja2.ChoiceLoader(
            [
                jinja2.PackageLoader("titiler.core", "templates"),
                jinja2.PackageLoader("titiler.eopf", "templates"),
            ]
        ),
    )
)

app = FastAPI(
    title=settings.name,
    description=settings.description,
    openapi_url="/api",
    docs_url="/api.html",
    version=pkg_version("cng-benchmark-tiler"),
)
update_openapi(app)

cog = TilerFactory(router_prefix="/cog", templates=templates)
app.include_router(cog.router, prefix="/cog", tags=["COG"])

zarr_tiler = XarrayTilerFactory(router_prefix="/zarr", templates=templates)
app.include_router(zarr_tiler.router, prefix="/zarr", tags=["Zarr"])

geozarr = GeoZarrTilerFactory(
    path_dependency=DatasetPathParams,
    router_prefix="/geozarr",
    templates=templates,
)
app.include_router(geozarr.router, prefix="/geozarr", tags=["GeoZarr"])

tms = TMSFactory(templates=templates)
app.include_router(tms.router, tags=["Tiling Schemes"])

algorithms = AlgorithmFactory(templates=templates)
app.include_router(algorithms.router, tags=["Algorithms"])

cmaps = ColorMapFactory(templates=templates)
app.include_router(cmaps.router, tags=["ColorMaps"])

add_exception_handlers(app, DEFAULT_STATUS_CODES)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_allow_methods,
        allow_headers=["*"],
    )

app.add_middleware(
    CompressionMiddleware,
    minimum_size=1,
    exclude_mediatype={
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/jp2",
        "image/webp",
    },
    compression_level=6,
)

app.add_middleware(
    CacheControlMiddleware,
    cachecontrol=settings.cachecontrol,
    exclude_path={r"/healthz"},
)


@app.get("/healthz", tags=["Health Check"])
def healthz() -> dict:
    """Health check + resolved tool versions.

    A benchmark run reads this once and records it in ``result.json``'s
    ``tool_versions``, so no display metric is ambiguous about which reader
    build produced it.
    """
    return {
        "versions": {
            "titiler_core": pkg_version("titiler.core"),
            "titiler_xarray": pkg_version("titiler.xarray"),
            "titiler_eopf": pkg_version("titiler.eopf"),
            "rasterio": rasterio.__version__,
            "gdal": rasterio.__gdal_version__,
            "proj": rasterio.__proj_version__,
            "zarr": zarr.__version__,
            "xarray": xarray.__version__,
        }
    }


@app.post("/_mgmt/geozarr-cache/clear", tags=["Cache Management"])
def clear_geozarr_cache() -> dict:
    """Clear ``GeoZarrReader``'s in-process opened-datatree cache.

    The reader memoizes opened datatrees, and would share that cache across
    replicas if Redis were configured. Redis stays off by default here (no
    ``EOPF_CACHE__*`` env is set), but the in-process memo alone still makes
    a warm replicate measure something different from a cold one -- this
    endpoint makes it resettable between replicates.
    """
    geozarr_open_dataset.cache_clear()
    return {"cleared": True}
