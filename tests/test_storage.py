"""Tests for the URI-addressed storage layer (local + S3)."""

import pytest

from cng_benchmark import storage


def test_list_object_sizes_local_directory(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 30)

    assert sorted(storage.list_object_sizes(str(tmp_path))) == [10, 30]


def test_list_object_sizes_local_single_file(tmp_path):
    f = tmp_path / "one.bin"
    f.write_bytes(b"z" * 42)
    assert storage.list_object_sizes(str(f)) == [42]


def test_list_object_sizes_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.list_object_sizes(str(tmp_path / "nope"))


def test_list_object_sizes_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="no objects"):
        storage.list_object_sizes(str(tmp_path))


def test_write_read_round_trip_local(tmp_path):
    uri = str(tmp_path / "nested" / "out.txt")
    storage.write_text(uri, "hello")
    assert storage.read_bytes(uri) == b"hello"


def test_write_read_round_trip_file_uri(tmp_path):
    uri = (tmp_path / "via-file-uri.txt").as_uri()
    storage.write_text(uri, "hi")
    assert storage.read_bytes(uri) == b"hi"


def test_to_gdal_path_maps_schemes():
    assert storage.to_gdal_path("s3://bucket/key.tif") == "/vsis3/bucket/key.tif"
    assert storage.to_gdal_path("file:///tmp/x.tif") == "/tmp/x.tif"
    assert storage.to_gdal_path("/local/x.tif") == "/local/x.tif"
    # An already-composed archive VSI path is passed through untouched.
    composed = "/vsizip//vsis3/bucket/a.zip/inner.tif"
    assert storage.to_gdal_path(composed) == composed


def test_s3_profile_sink_reads_bare_env(monkeypatch):
    for k in ("SOURCE_AWS_ENDPOINT_URL", "SOURCE_AWS_ACCESS_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "sinkkey")
    monkeypatch.setenv("AWS_CA_BUNDLE", "/etc/ssl/sink.pem")

    p = storage.s3_profile("sink")
    assert p.endpoint == "http://minio:9000"
    assert p.access_key == "sinkkey"
    assert p.ca_bundle == "/etc/ssl/sink.pem"


def test_s3_profile_source_falls_back_then_overrides(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "shared")
    monkeypatch.delenv("SOURCE_AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("SOURCE_AWS_ACCESS_KEY_ID", raising=False)

    # With no SOURCE_*, source falls back to the bare AWS_* (single-endpoint).
    assert storage.s3_profile("source").endpoint == "http://minio:9000"
    assert storage.s3_profile("source").access_key == "shared"

    # SOURCE_* overrides for the source role only; sink is unaffected.
    monkeypatch.setenv("SOURCE_AWS_ENDPOINT_URL", "https://s3.datalake.cnes.fr")
    monkeypatch.setenv("SOURCE_AWS_ACCESS_KEY_ID", "datalakekey")
    assert storage.s3_profile("source").endpoint == "https://s3.datalake.cnes.fr"
    assert storage.s3_profile("source").access_key == "datalakekey"
    assert storage.s3_profile("sink").endpoint == "http://minio:9000"


def test_s3_profile_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown S3 role"):
        storage.s3_profile("souce")  # typo must fail fast, not fall back to sink


def test_object_size_local_and_missing(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"a" * 17)
    assert storage.object_size(str(f)) == 17
    assert storage.object_size(str(tmp_path / "missing.bin")) is None
    assert storage.object_size(str(tmp_path)) is None  # a directory, not an object


def test_join_local_and_s3():
    assert storage.join("s3://bucket/results", "result.json") == (
        "s3://bucket/results/result.json"
    )
    assert storage.join("s3://bucket/results/", "result.json") == (
        "s3://bucket/results/result.json"
    )
    assert storage.join("/tmp/out", "summary.md").endswith("out/summary.md")


# --- S3 path, exercised against an in-memory moto server ---------------------

moto = pytest.importorskip("moto")


@pytest.fixture
def s3_bucket(monkeypatch):
    """A moto-backed S3 bucket with credentials set in the environment."""
    from moto import mock_aws

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    with mock_aws():
        import boto3

        boto3.client("s3").create_bucket(Bucket="bench")
        yield "bench"


def test_s3_write_list_read_round_trip(s3_bucket):
    storage.write_bytes(f"s3://{s3_bucket}/fixtures/a.tif", b"a" * 100)
    storage.write_bytes(f"s3://{s3_bucket}/fixtures/b.tif", b"b" * 200)

    sizes = storage.list_object_sizes(f"s3://{s3_bucket}/fixtures/")
    assert sorted(sizes) == [100, 200]
    assert storage.read_bytes(f"s3://{s3_bucket}/fixtures/a.tif") == b"a" * 100


def test_s3_list_empty_prefix_raises(s3_bucket):
    with pytest.raises(ValueError, match="no objects"):
        storage.list_object_sizes(f"s3://{s3_bucket}/empty/")


def test_s3_write_to_prefix_rejected(s3_bucket):
    with pytest.raises(ValueError, match="object key"):
        storage.write_bytes(f"s3://{s3_bucket}/prefix/", b"x")


def test_s3_read_from_prefix_rejected(s3_bucket):
    with pytest.raises(ValueError, match="object key"):
        storage.read_bytes(f"s3://{s3_bucket}/prefix/")
