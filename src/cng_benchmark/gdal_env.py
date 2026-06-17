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
"""

from __future__ import annotations

from contextlib import contextmanager

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
def gdal_session(role: str = "sink"):
    """Scope GDAL ``/vsis3`` config to ``role`` for the duration of the block.

    Reads the role's :class:`~cng_benchmark.storage.S3Profile` from the
    environment. With no S3 endpoint configured (e.g. a local read, or the
    synthetic path before any ``AWS_*`` is set) it is effectively a no-op beyond
    GDAL's directory-listing tweak.
    """
    rasterio, AWSSession = _require_geo()
    p = storage.s3_profile(role)

    options: dict[str, str] = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
    if p.endpoint:
        options["AWS_S3_ENDPOINT"] = p.endpoint.split("://", 1)[-1]
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

    with rasterio.Env(session, **options):
        yield
