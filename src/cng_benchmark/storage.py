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
"""

from __future__ import annotations

import os
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


def _local_path(uri: str) -> Path:
    """Map a local path or ``file://`` URI to a :class:`Path`."""
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def _s3_client():
    """Build an S3 client from the ``AWS_*`` environment.

    Raises a clear, actionable error when the optional ``s3`` extra (boto3) is
    not installed. An explicit endpoint override (``AWS_ENDPOINT_URL_S3`` or
    ``AWS_ENDPOINT_URL``) switches on path-style addressing so MinIO and other
    S3-compatible servers work without virtual-host DNS.
    """
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(
            "S3 access requires the 's3' extra; install with "
            "`uv sync --extra s3` (or `pip install cng-benchmark[s3]`)"
        ) from exc

    endpoint = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("AWS_ENDPOINT_URL")
    config = Config(s3={"addressing_style": "path"}) if endpoint else None
    return boto3.client("s3", endpoint_url=endpoint, config=config)


def list_object_sizes(uri: str) -> list[int]:
    """Return the byte sizes of the objects under ``uri``.

    For a local directory, every regular file beneath it (recursively) counts.
    For a local file, its single size. For ``s3://bucket/prefix``, every object
    whose key starts with the prefix. Raises :class:`FileNotFoundError` when a
    local path is missing and :class:`ValueError` when nothing is found at an
    otherwise valid location (an empty profile is never useful).
    """
    if is_s3(uri):
        loc = _parse_s3(uri)
        client = _s3_client()
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


def write_bytes(uri: str, data: bytes) -> None:
    """Write ``data`` to ``uri`` (a local path/``file://`` or ``s3://`` key)."""
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        _s3_client().put_object(Bucket=loc.bucket, Key=loc.key, Body=data)
        return
    path = _local_path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_text(uri: str, text: str) -> None:
    """Write ``text`` (UTF-8) to ``uri``."""
    write_bytes(uri, text.encode("utf-8"))


def read_bytes(uri: str) -> bytes:
    """Read the bytes of the object/file at ``uri``."""
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        return _s3_client().get_object(Bucket=loc.bucket, Key=loc.key)["Body"].read()
    return _local_path(uri).read_bytes()


def join(base: str, name: str) -> str:
    """Join ``name`` onto a base location URI/path, S3-aware.

    Used to place named artifacts (``result.json``, ``summary.md``) under an
    output location given as a directory/prefix.
    """
    if is_s3(base):
        return base.rstrip("/") + "/" + name.lstrip("/")
    return str(_local_path(base) / name)
