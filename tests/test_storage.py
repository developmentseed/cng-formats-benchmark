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


def test_upload_tree_local_preserves_paths(tmp_path):
    src = tmp_path / "store.zarr"
    (src / "c" / "0").mkdir(parents=True)
    (src / "zarr.json").write_bytes(b"{}")
    (src / "c" / "0" / "0").write_bytes(b"shard")
    dest = tmp_path / "out"
    storage.upload_tree(str(src), str(dest))
    assert (dest / "zarr.json").read_bytes() == b"{}"
    assert (dest / "c" / "0" / "0").read_bytes() == b"shard"


def test_upload_tree_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files to upload"):
        storage.upload_tree(str(empty), str(tmp_path / "out"))


def test_upload_tree_missing_dir_raises_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.upload_tree(str(tmp_path / "nope"), str(tmp_path / "out"))


def test_upload_tree_local_clears_stale_destination(tmp_path):
    src = tmp_path / "store.zarr"
    (src / "c").mkdir(parents=True)
    (src / "zarr.json").write_bytes(b"{}")
    (src / "c" / "0").write_bytes(b"shard")
    dest = tmp_path / "out"
    storage.upload_tree(str(src), str(dest))
    # A stale object from a previous run sitting under the reused prefix.
    (dest / "c" / "stale").write_bytes(b"old")
    storage.upload_tree(str(src), str(dest))
    assert not (dest / "c" / "stale").exists()
    assert (dest / "c" / "0").read_bytes() == b"shard"


def test_upload_tree_s3_clears_stale_destination(s3_bucket):
    import tempfile
    from pathlib import Path

    uri = f"s3://{s3_bucket}/objects/store.zarr"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "store.zarr"
        (root / "c").mkdir(parents=True)
        (root / "zarr.json").write_bytes(b"meta")
        (root / "c" / "0").write_bytes(b"shard")
        storage.upload_tree(str(root), uri)
        storage.write_bytes(f"{uri}/c/stale", b"old")  # leftover from a prior run
        storage.upload_tree(str(root), uri)

    keys = {u.rsplit("/", 1)[-1] for u in storage.list_uris(f"{uri}/", role="sink")}
    assert "stale" not in keys


def test_upload_tree_s3_preserves_paths(s3_bucket):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "store.zarr"
        (root / "c").mkdir(parents=True)
        (root / "zarr.json").write_bytes(b"meta")
        (root / "c" / "0").write_bytes(b"shard-bytes")
        storage.upload_tree(str(root), f"s3://{s3_bucket}/objects/store.zarr")

    sizes = storage.list_object_sizes(f"s3://{s3_bucket}/objects/store.zarr/")
    assert sorted(sizes) == [len(b"meta"), len(b"shard-bytes")]
    body = storage.read_bytes(f"s3://{s3_bucket}/objects/store.zarr/c/0")
    assert body == b"shard-bytes"


def test_fsspec_storage_options_shapes_role_profile(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://s3.example.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("AWS_CA_BUNDLE", "/etc/ssl/ca.pem")
    opts = storage.fsspec_storage_options("sink")
    assert opts["key"] == "k" and opts["secret"] == "s"
    assert opts["client_kwargs"]["endpoint_url"] == "https://s3.example.com"
    assert opts["client_kwargs"]["verify"] == "/etc/ssl/ca.pem"


def test_s3_list_empty_prefix_raises(s3_bucket):
    with pytest.raises(ValueError, match="no objects"):
        storage.list_object_sizes(f"s3://{s3_bucket}/empty/")


def test_s3_write_to_prefix_rejected(s3_bucket):
    with pytest.raises(ValueError, match="object key"):
        storage.write_bytes(f"s3://{s3_bucket}/prefix/", b"x")


def test_s3_read_from_prefix_rejected(s3_bucket):
    with pytest.raises(ValueError, match="object key"):
        storage.read_bytes(f"s3://{s3_bucket}/prefix/")


def test_list_uris_local_prefix_suffix_limit(tmp_path):
    root = tmp_path / "T31TCJ"
    root.mkdir()
    for name in ("2015_a.zip", "2016_b.zip", "2016_c.zip", "2016_note.txt"):
        (root / name).write_bytes(b"")

    # suffix narrows to zips; prefix is a path-prefix under the root.
    assert [p.split("/")[-1] for p in storage.list_uris(str(root), suffix=".zip")] == [
        "2015_a.zip",
        "2016_b.zip",
        "2016_c.zip",
    ]
    got = storage.list_uris(str(root), prefix="2016", suffix=".zip")
    assert [p.split("/")[-1] for p in got] == ["2016_b.zip", "2016_c.zip"]
    # limit stops early in sorted order.
    bounded = storage.list_uris(str(root), prefix="2016", suffix=".zip", limit=1)
    assert [p.split("/")[-1] for p in bounded] == ["2016_b.zip"]


def test_s3_list_uris_prefix_is_server_side_and_path_prefix(s3_bucket):
    for key in (
        "T31TCJ/2015_a.zip",
        "T31TCJ/2016_b.zip",
        "T31TCJ/2016_c.zip",
        "T31TCJ/2016_note.txt",
        "T31TCJ/x2016_decoy.zip",  # substring "2016" but not a path prefix
    ):
        storage.write_bytes(f"s3://{s3_bucket}/{key}", b"x")

    got = storage.list_uris(f"s3://{s3_bucket}/T31TCJ/", prefix="2016", suffix=".zip")
    assert got == [
        f"s3://{s3_bucket}/T31TCJ/2016_b.zip",
        f"s3://{s3_bucket}/T31TCJ/2016_c.zip",
    ]
    bounded = storage.list_uris(
        f"s3://{s3_bucket}/T31TCJ/", prefix="2016", suffix=".zip", limit=1
    )
    assert bounded == [f"s3://{s3_bucket}/T31TCJ/2016_b.zip"]


def test_s3_list_uris_root_without_trailing_slash(s3_bucket):
    storage.write_bytes(f"s3://{s3_bucket}/T31TCJ/2016_b.zip", b"x")
    # A root key with no trailing slash is still treated as a directory root,
    # so the prefix doesn't glue onto the last path segment.
    got = storage.list_uris(f"s3://{s3_bucket}/T31TCJ", prefix="2016", suffix=".zip")
    assert got == [f"s3://{s3_bucket}/T31TCJ/2016_b.zip"]


def _zip_bytes(names: list[str]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, b"x" * 32)
    return buf.getvalue()


def test_open_seekable_s3_lists_zip_members_and_closes(s3_bucket):
    # Regression: listing a remote zip's members reads only the central
    # directory through a ranged seekable reader, and the caller closes it.
    # The S3 reader must implement close() or zip-delivery datasets fail with
    # AttributeError on every real (S3) archive.
    import zipfile

    members = ["A_FRE_B2.tif", "MASKS/A_CLM_R1.tif"]
    storage.write_bytes(f"s3://{s3_bucket}/scenes/A.zip", _zip_bytes(members))

    fileobj = storage.open_seekable(f"s3://{s3_bucket}/scenes/A.zip")
    try:
        with zipfile.ZipFile(fileobj) as zf:
            assert sorted(zf.namelist()) == sorted(members)
    finally:
        fileobj.close()  # must not raise

    # Also usable as a context manager.
    with storage.open_seekable(f"s3://{s3_bucket}/scenes/A.zip") as f2:
        assert zipfile.ZipFile(f2).namelist()


def test_list_uris_s3_filters_by_suffix(s3_bucket):
    storage.write_bytes(f"s3://{s3_bucket}/t/2015/a.zip", b"z")
    storage.write_bytes(f"s3://{s3_bucket}/t/2015/a.xml", b"m")
    storage.write_bytes(f"s3://{s3_bucket}/t/2016/b.zip", b"z")
    uris = storage.list_uris(f"s3://{s3_bucket}/t/", suffix=".zip")
    assert uris == [f"s3://{s3_bucket}/t/2015/a.zip", f"s3://{s3_bucket}/t/2016/b.zip"]
