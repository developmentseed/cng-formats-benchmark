"""Per-role GDAL/`/vsis3` configuration for reads.

GDAL's S3 configuration is process-global (the ``AWS_*`` environment), so a
single run that reads its source from one provider and its sink from another
cannot rely on one static environment. :func:`gdal_session` scopes the
configuration to a ``with`` block using :class:`rasterio.Env`, applying the
:class:`~cng_benchmark.storage.S3Profile` for a given role — endpoint
(host:port, path-style), HTTPS, and a private CA bundle — so the conversion can
read its source and the read metric can read the sink, each with its own
credentials and CA, in the same process.

rasterio is the ``cog`` extra; it is imported lazily so the core stays light.

A subtlety the env overlay below handles: not every GDAL consumer goes through
rasterio. ``pyogrio`` (the ``geoparquet`` extra) bundles its *own* libgdal in its
manylinux wheel, so :class:`rasterio.Env`'s thread-local config does not reach
it. To keep the per-role endpoint and credentials authoritative for *any* GDAL
binding in the block (pyogrio's OGR source read as much as rasterio's raster
read), the same options are also overlaid onto ``os.environ`` for the duration
and restored on exit — GDAL reads the process environment when no thread-local
override is present.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from urllib.parse import urlparse

from cng_benchmark import storage


def _require_geo():
    try:
        import rasterio
        from rasterio.session import AWSSession
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "GDAL/vsis3 reads require the 'cog' extra; install with "
            "`uv sync --extra cog` (or `pip install cng-benchmark[cog]`)"
        ) from exc
    return rasterio, AWSSession


@contextmanager
def _environ_overlay(overlay: dict[str, str]):
    """Apply ``overlay`` to ``os.environ`` for the block, restoring prior values.

    Used so a GDAL binding that does not honour :class:`rasterio.Env` (notably
    ``pyogrio``, with its own bundled libgdal) still sees the active role's
    endpoint/credentials. Keys absent before are removed on exit; keys present are
    restored to their prior value, so a nested or subsequent session is unaffected.
    """
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in overlay}
    os.environ.update(overlay)
    try:
        yield
    finally:
        for key, prior in saved.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


@contextmanager
def gdal_session(role: str = "sink"):
    """Scope GDAL ``/vsis3`` config to ``role`` for the duration of the block.

    Reads the role's :class:`~cng_benchmark.storage.S3Profile` from the
    environment. With no S3 endpoint configured (e.g. a local read, or the
    synthetic path before any ``AWS_*`` is set) it is effectively a no-op beyond
    GDAL's directory-listing tweak.

    The config is applied two ways for the block: via :class:`rasterio.Env`
    (rasterio's GDAL) and as an ``os.environ`` overlay (every other GDAL binding,
    e.g. pyogrio's OGR read for the GeoParquet source), both restored on exit.
    """
    rasterio, AWSSession = _require_geo()
    p = storage.s3_profile(role)

    options: dict[str, str] = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if p.endpoint:
        # GDAL's AWS_S3_ENDPOINT wants host[:port] only — strip scheme and any
        # path component (a trailing path would make /vsis3 requests fail).
        options["AWS_S3_ENDPOINT"] = urlparse(p.endpoint).netloc or p.endpoint
        options["AWS_VIRTUAL_HOSTING"] = "FALSE"
        options["AWS_HTTPS"] = "YES" if p.endpoint.startswith("https") else "NO"
    if p.ca_bundle:
        options["GDAL_HTTP_CAINFO"] = p.ca_bundle

    session = None
    if p.access_key and p.secret_key:
        session = AWSSession(
            aws_access_key_id=p.access_key,
            aws_secret_access_key=p.secret_key,
            region_name=p.region,
        )

    # Mirror the GDAL config (plus the role's credentials/region) into os.environ
    # so a non-rasterio binding in the block reads the same per-role S3 settings.
    env_overlay = dict(options)
    if p.access_key and p.secret_key:
        env_overlay["AWS_ACCESS_KEY_ID"] = p.access_key
        env_overlay["AWS_SECRET_ACCESS_KEY"] = p.secret_key
        if p.region:
            env_overlay["AWS_DEFAULT_REGION"] = p.region

    with _environ_overlay(env_overlay), rasterio.Env(session, **options):
        yield
