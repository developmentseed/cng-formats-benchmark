"""Tests for the per-role GDAL session context manager (requires the `cog` extra)."""

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("rio_cogeo")

from cng_benchmark.fixtures import generate_cog_bytes  # noqa: E402
from cng_benchmark.gdal_env import gdal_session  # noqa: E402


def test_gdal_session_scopes_a_local_read(tmp_path, monkeypatch):
    # No S3 endpoint configured: the session is a near no-op and a local read
    # still works inside it.
    for k in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "SOURCE_AWS_ENDPOINT_URL"):
        monkeypatch.delenv(k, raising=False)
    cog = tmp_path / "cog.tif"
    cog.write_bytes(generate_cog_bytes(size=128, blocksize=128))

    import rasterio

    with gdal_session("sink"):
        with rasterio.open(cog) as src:
            assert src.count == 3


def test_gdal_session_applies_source_endpoint_options(monkeypatch):
    # The source profile drives the GDAL options inside the context.
    monkeypatch.setenv("SOURCE_AWS_ENDPOINT_URL", "https://s3.datalake.cnes.fr")
    monkeypatch.setenv("SOURCE_AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("SOURCE_AWS_SECRET_ACCESS_KEY", "s")

    import rasterio

    with gdal_session("source"):
        opts = rasterio.env.getenv()
        assert opts["AWS_S3_ENDPOINT"] == "s3.datalake.cnes.fr"
        assert opts["AWS_HTTPS"] == "YES"
        assert opts["AWS_VIRTUAL_HOSTING"] == "FALSE"
