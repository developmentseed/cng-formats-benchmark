"""URI-addressed object IO for the deployable runner.

The harness core is service-free, but the *deployed* runner has to read the
objects it profiles and persist its result artifact somewhere. This module is
that thin seam: a handful of functions that operate on a location given as a
URI, dispatching to the local filesystem for a path or ``file://`` URI and to
S3 for an ``s3://`` URI.

The S3 path is the same code whether it talks to a real provider (the Scaleway
lab bucket) or the MinIO stand-in used by docker-compose and CI — only the
endpoint and credentials differ, and those come from the standard ``AWS_*``
environment (``AWS_ENDPOINT_URL`` / ``AWS_ENDPOINT_URL_S3`` point boto3 at
MinIO). boto3 is an optional dependency (the ``s3`` extra); importing it is
deferred so the core stays installable and unit-testable without it.

A real run spans two providers at once — it reads its *source* from one (the
private-CA CNES Datalake, read-only) and writes its *sink* to another (Scaleway,
read-write). Configuration is therefore resolved per **role**: the ``sink`` role
reads the bare ``AWS_*`` environment, the ``source`` role reads ``SOURCE_AWS_*``
and falls back to the bare ``AWS_*``. So the synthetic single-endpoint path
(source and sink both MinIO) needs no ``SOURCE_*`` and behaves exactly as before.
GDAL ``/vsis3`` reads (the read metric, and the conversion reading its source)
apply the same per-role profile through :mod:`cng_benchmark.gdal_env`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class _S3Uri:
    bucket: str
    key: str  # may be "" (bucket root) or a prefix/object key


def _parse_s3(uri: str) -> _S3Uri:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    if not bucket:
        raise ValueError(f"s3 URI {uri!r} has no bucket")
    return _S3Uri(bucket=bucket, key=parsed.path.lstrip("/"))


def is_s3(uri: str) -> bool:
    """Return ``True`` if ``uri`` is an ``s3://`` location."""
    return uri.startswith("s3://")


def to_gdal_path(uri: str) -> str:
    """Map a URI to a GDAL-openable path.

    ``s3://bucket/key`` becomes ``/vsis3/bucket/key`` and ``file://`` is reduced
    to its local path; anything else (a local path, or an already-composed GDAL
    VSI path such as ``/vsizip//vsis3/…`` for an archive) is passed through.
    """
    if is_s3(uri):
        return "/vsis3/" + uri[len("s3://") :]
    if uri.startswith("file://"):
        return urlparse(uri).path
    return uri


@dataclass(frozen=True)
class S3Profile:
    """Per-role S3 connection settings resolved from the environment."""

    endpoint: str | None
    region: str | None
    access_key: str | None
    secret_key: str | None
    ca_bundle: str | None


def _role_env(role: str, name: str) -> str | None:
    """Read ``name`` for ``role``: ``source`` prefers ``SOURCE_`` then bare."""
    if role == "source":
        return os.getenv("SOURCE_" + name) or os.getenv(name)
    return os.getenv(name)


def s3_profile(role: str = "sink") -> S3Profile:
    """Resolve the S3 settings for ``role`` (``"sink"`` or ``"source"``)."""
    return S3Profile(
        endpoint=_role_env(role, "AWS_ENDPOINT_URL_S3")
        or _role_env(role, "AWS_ENDPOINT_URL"),
        region=_role_env(role, "AWS_DEFAULT_REGION") or _role_env(role, "AWS_REGION"),
        access_key=_role_env(role, "AWS_ACCESS_KEY_ID"),
        secret_key=_role_env(role, "AWS_SECRET_ACCESS_KEY"),
        ca_bundle=_role_env(role, "AWS_CA_BUNDLE"),
    )


def _local_path(uri: str) -> Path:
    """Map a local path or ``file://`` URI to a :class:`Path`."""
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def _s3_client(role: str = "sink"):
    """Build an S3 client for ``role`` from the resolved :class:`S3Profile`.

    Raises a clear, actionable error when the optional ``s3`` extra (boto3) is
    not installed. An endpoint override switches on path-style addressing so
    MinIO and other S3-compatible servers work without virtual-host DNS; a
    role-specific CA bundle (``verify``) supports a private-CA source.
    """
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "S3 access requires the 's3' extra; install with "
            "`uv sync --extra s3` (or `pip install cng-benchmark[s3]`)"
        ) from exc

    p = s3_profile(role)
    config = Config(s3={"addressing_style": "path"}) if p.endpoint else None
    return boto3.client(
        "s3",
        endpoint_url=p.endpoint,
        region_name=p.region,
        aws_access_key_id=p.access_key,
        aws_secret_access_key=p.secret_key,
        config=config,
        verify=p.ca_bundle or None,
    )


def list_object_sizes(uri: str, role: str = "sink") -> list[int]:
    """Return the byte sizes of the objects under ``uri``.

    For a local directory, every regular file beneath it (recursively) counts.
    For a local file, its single size. For ``s3://bucket/prefix``, every object
    whose key starts with the prefix. Raises :class:`FileNotFoundError` when a
    local path is missing and :class:`ValueError` when nothing is found at an
    otherwise valid location (an empty profile is never useful).
    """
    if is_s3(uri):
        loc = _parse_s3(uri)
        client = _s3_client(role)
        paginator = client.get_paginator("list_objects_v2")
        sizes: list[int] = []
        for page in paginator.paginate(Bucket=loc.bucket, Prefix=loc.key):
            for obj in page.get("Contents", []):
                # Skip "directory marker" zero-byte keys ending in "/".
                if obj["Key"].endswith("/"):
                    continue
                sizes.append(int(obj["Size"]))
        if not sizes:
            raise ValueError(f"no objects found at {uri}")
        return sizes

    path = _local_path(uri)
    if not path.exists():
        raise FileNotFoundError(f"no such path: {uri}")
    if path.is_file():
        return [path.stat().st_size]
    sizes = [p.stat().st_size for p in sorted(path.rglob("*")) if p.is_file()]
    if not sizes:
        raise ValueError(f"no objects found under {uri}")
    return sizes


def _require_object_key(loc: _S3Uri, uri: str) -> None:
    """Reject an S3 URI that names a bucket root or prefix rather than an object.

    Gives a clear, actionable error for the single-object operations
    (read/write) instead of a low-level botocore failure.
    """
    if not loc.key or loc.key.endswith("/"):
        raise ValueError(f"s3 URI {uri!r} must name an object key, not a prefix")


def write_bytes(uri: str, data: bytes, role: str = "sink") -> None:
    """Write ``data`` to ``uri`` (a local path/``file://`` or ``s3://`` key)."""
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        _s3_client(role).put_object(Bucket=loc.bucket, Key=loc.key, Body=data)
        return
    path = _local_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(uri: str, text: str, role: str = "sink") -> None:
    """Write ``text`` (UTF-8) to ``uri``."""
    write_bytes(uri, text.encode("utf-8"), role)


def download_to_path(uri: str, local_path: str, role: str = "source") -> None:
    """Stream the object at ``uri`` to ``local_path`` without buffering it all.

    Uses boto3's file transfer (multipart, streamed to disk) for S3 and a file
    copy for local sources, so a multi-gigabyte scene never has to fit in memory.
    """
    dest = Path(local_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        _s3_client(role).download_file(loc.bucket, loc.key, str(dest))
        return
    shutil.copyfile(_local_path(uri), dest)


def upload_from_path(local_path: str, uri: str, role: str = "sink") -> None:
    """Stream ``local_path`` to ``uri`` (S3 multipart or local copy)."""
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        _s3_client(role).upload_file(str(local_path), loc.bucket, loc.key)
        return
    dest = _local_path(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, dest)


def read_bytes(uri: str, role: str = "sink") -> bytes:
    """Read the bytes of the object/file at ``uri``."""
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        client = _s3_client(role)
        return client.get_object(Bucket=loc.bucket, Key=loc.key)["Body"].read()
    return _local_path(uri).read_bytes()


def object_size(uri: str, role: str = "sink") -> int | None:
    """Best-effort byte size of a single object, or ``None`` if unknown.

    Used for the write metric's ``bytes_in`` detail. Returns the size for a
    local file or an S3 object (via ``head_object``); returns ``None`` for a
    path GDAL can read but that has no cheap stat (e.g. a ``/vsizip`` member).
    """
    if is_s3(uri):
        loc = _parse_s3(uri)
        if not loc.key or loc.key.endswith("/"):
            return None
        try:
            return int(
                _s3_client(role).head_object(Bucket=loc.bucket, Key=loc.key)[
                    "ContentLength"
                ]
            )
        except Exception:  # noqa: BLE001 - best effort; size detail is optional
            return None
    path = _local_path(uri)
    return path.stat().st_size if path.is_file() else None


def join(base: str, name: str) -> str:
    """Join ``name`` onto a base location URI/path, S3-aware.

    Used to place named artifacts (``result.json``, ``summary.md``) under an
    output location given as a directory/prefix.
    """
    if is_s3(base):
        return base.rstrip("/") + "/" + name.lstrip("/")
    return str(_local_path(base) / name)
