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


def test_gdal_session_strips_path_from_endpoint(monkeypatch):
    # AWS_S3_ENDPOINT must be host[:port] only, even if the URL carries a path.
    monkeypatch.setenv("SOURCE_AWS_ENDPOINT_URL", "https://host.example:9000/tenant/")
    monkeypatch.setenv("SOURCE_AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("SOURCE_AWS_SECRET_ACCESS_KEY", "s")

    import rasterio

    with gdal_session("source"):
        assert rasterio.env.getenv()["AWS_S3_ENDPOINT"] == "host.example:9000"


def test_gdal_session_overlays_os_environ_for_non_rasterio_bindings(monkeypatch):
    # A binding with its own libgdal (e.g. pyogrio) does not honour rasterio.Env,
    # so the per-role endpoint + credentials must also be on os.environ inside the
    # block — and removed again on exit (they were absent before).
    import os

    monkeypatch.setenv("SOURCE_AWS_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
    monkeypatch.setenv("SOURCE_AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("SOURCE_AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.delenv("AWS_S3_ENDPOINT", raising=False)

    assert "AWS_S3_ENDPOINT" not in os.environ
    with gdal_session("source"):
        assert os.environ["AWS_S3_ENDPOINT"] == "s3.fr-par.scw.cloud"
        assert os.environ["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert os.environ["AWS_ACCESS_KEY_ID"] == "k"
        assert os.environ["AWS_SECRET_ACCESS_KEY"] == "s"
    # Restored on exit: the key absent before the block is gone again.
    assert "AWS_S3_ENDPOINT" not in os.environ


def test_gdal_session_restores_prior_os_environ_values(monkeypatch):
    # A key present before the block is restored to its prior value, not dropped,
    # so a sink-then-source sequence leaves the outer environment intact.
    import os

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "sink-key")
    monkeypatch.setenv("SOURCE_AWS_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
    monkeypatch.setenv("SOURCE_AWS_ACCESS_KEY_ID", "source-key")
    monkeypatch.setenv("SOURCE_AWS_SECRET_ACCESS_KEY", "s")

    with gdal_session("source"):
        assert os.environ["AWS_ACCESS_KEY_ID"] == "source-key"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "sink-key"
