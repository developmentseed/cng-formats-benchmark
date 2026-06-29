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
import re
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


_ROLES = ("sink", "source")


def _role_env(role: str, name: str) -> str | None:
    """Read ``name`` for ``role``: ``source`` prefers ``SOURCE_`` then bare."""
    if role == "source":
        return os.getenv("SOURCE_" + name) or os.getenv(name)
    return os.getenv(name)


def s3_profile(role: str = "sink") -> S3Profile:
    """Resolve the S3 settings for ``role`` (``"sink"`` or ``"source"``).

    The role selects which credentials/CA are used, so an unknown value is a
    programming error and fails fast rather than silently using the sink's.
    """
    if role not in _ROLES:
        raise ValueError(f"unknown S3 role {role!r}; expected one of {_ROLES}")
    return S3Profile(
        endpoint=_role_env(role, "AWS_ENDPOINT_URL_S3")
        or _role_env(role, "AWS_ENDPOINT_URL"),
        region=_role_env(role, "AWS_DEFAULT_REGION") or _role_env(role, "AWS_REGION"),
        access_key=_role_env(role, "AWS_ACCESS_KEY_ID"),
        secret_key=_role_env(role, "AWS_SECRET_ACCESS_KEY"),
        ca_bundle=_role_env(role, "AWS_CA_BUNDLE"),
    )


def fsspec_storage_options(role: str = "sink") -> dict:
    """Build fsspec/s3fs ``storage_options`` for ``role`` from the environment.

    The GeoZarr read path opens a sharded store with zarr-python over fsspec
    (GDAL's Zarr driver cannot read the ``sharding_indexed`` codec), so it needs
    the same per-role endpoint/CA/credentials the boto3 and GDAL paths use,
    expressed in s3fs's shape (``key``/``secret`` plus ``client_kwargs``). Empty
    for a local store.
    """
    p = s3_profile(role)
    opts: dict = {}
    if p.access_key:
        opts["key"] = p.access_key
    if p.secret_key:
        opts["secret"] = p.secret_key
    client_kwargs: dict = {}
    if p.endpoint:
        client_kwargs["endpoint_url"] = p.endpoint
    if p.region:
        client_kwargs["region_name"] = p.region
    if p.ca_bundle:
        client_kwargs["verify"] = p.ca_bundle
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs
    return opts


def _local_path(uri: str) -> Path:
    """Map a local path or ``file://`` URI to a :class:`Path`."""
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def _s3_client(role: str = "sink", *, extra_config: dict | None = None):
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
    cfg: dict = {}
    if p.endpoint:
        cfg["s3"] = {"addressing_style": "path"}
    if extra_config:
        cfg.update(extra_config)
    config = Config(**cfg) if cfg else None
    return boto3.client(
        "s3",
        endpoint_url=p.endpoint,
        region_name=p.region,
        aws_access_key_id=p.access_key,
        aws_secret_access_key=p.secret_key,
        config=config,
        verify=p.ca_bundle or None,
    )


def download_s3_object(uri: str, dest_path, role: str = "source") -> None:
    """Download a single S3 object to ``dest_path`` via boto3's transfer manager.

    Used for granule sources that a library must read as a local file (the SWOT
    PIXC netCDF, read with h5netcdf). boto3's sync multipart transfer with
    generous timeouts and retries is robust where s3fs's async block reader trips
    socket read timeouts on a large object over a slow endpoint. The ``source``
    role resolves ``SOURCE_AWS_*`` (falling back to bare ``AWS_*``).
    """
    loc = _parse_s3(uri)
    client = _s3_client(
        role,
        extra_config={
            "read_timeout": 120,
            "connect_timeout": 30,
            "retries": {"max_attempts": 10, "mode": "standard"},
        },
    )
    client.download_file(loc.bucket, loc.key, str(dest_path))


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


def _alternation_prefixes(pattern: str) -> list[str] | None:
    """Extract literal branches from a leading anchored alternation group.

    Matches patterns of the form ``^(A|B|C)...`` where every branch is a
    literal path segment (word chars, hyphens, dots — no regex metacharacters).
    Returns the branch list so the S3 listing can issue one tight server-side
    prefix per branch instead of one broad scan that covers the whole bucket.

    Returns ``None`` when the pattern does not start with a literal alternation,
    leaving ``list_uris`` to fall back to the single-prefix approach.
    """
    m = re.match(r"^\^?\(([^()]+)\)", pattern)
    if m is None:
        return None
    alts = m.group(1).split("|")
    if not all(re.fullmatch(r"[A-Za-z0-9._\-]+", a) for a in alts):
        return None
    return alts


def list_uris(
    uri: str,
    role: str = "source",
    *,
    prefix: str | None = None,
    suffix: str | None = None,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """List the object URIs under ``uri`` (a local dir or ``s3://prefix``).

    Returns each object as a URI of the same scheme as ``uri`` (a plain local
    path for a local input, an ``s3://`` URI for an S3 prefix), in sorted order.
    Used to find the scenes/granules under a dataset root.

    ``prefix`` narrows enumeration to keys whose path *under* ``uri`` starts with
    it: for S3 it is folded into the ``list_objects_v2`` ``Prefix`` so the bound
    is applied server-side (a root-level list of a huge bucket never happens),
    and it is a path-prefix match, not a substring one. ``suffix`` keeps only
    keys ending in it. ``pattern`` is a regular expression matched (``re.search``)
    against the key *relative to* ``uri`` — a substring match by default, anchor
    it with ``^``/``$`` to bound — which is how a single run selects a set that
    is not a single path-prefix, e.g. one acquisition date across several
    adjacent MGRS tiles (``^T31T(CJ|DJ|CH|DH|DG)/2015/07/06/``). Narrow the
    listing server-side with ``prefix`` (the regex's literal head) so the regex
    only filters the candidates it returns. ``limit`` stops after that many
    matches — for S3 this relies on the API's lexical key order so the full
    listing is never materialised before the bound is applied.
    """
    if limit is not None and limit <= 0:
        return []
    rx = re.compile(pattern) if pattern is not None else None
    if is_s3(uri):
        loc = _parse_s3(uri)
        client = _s3_client(role)
        base = loc.key
        if base and not base.endswith("/"):
            base += "/"
        s3_prefix = base + (prefix or "")
        paginator = client.get_paginator("list_objects_v2")
        # When the pattern opens with a literal alternation group (e.g.
        # ``^(T30TWP|T31TCJ)/``), issue one tight listing per branch instead of
        # one broad scan under s3_prefix — avoids walking a multi-year archive
        # to find a sparse set of tiles in a large bucket (#58).
        alt_pfxs = _alternation_prefixes(pattern) if pattern is not None else None
        listing_prefixes = (
            [base + alt for alt in alt_pfxs] if alt_pfxs is not None else [s3_prefix]
        )
        uris: list[str] = []
        for lp in listing_prefixes:
            for page in paginator.paginate(Bucket=loc.bucket, Prefix=lp):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    if suffix is not None and not key.endswith(suffix):
                        continue
                    rel = key[len(base) :] if base else key
                    if rx is not None and not rx.search(rel):
                        continue
                    uris.append(f"s3://{loc.bucket}/{key}")
                    if limit is not None and len(uris) >= limit:
                        return uris
        return uris

    path = _local_path(uri)
    if not path.exists():
        raise FileNotFoundError(f"no such path: {uri}")
    if path.is_file():
        keep = (
            (suffix is None or path.name.endswith(suffix))
            and (prefix is None or path.name.startswith(prefix))
            and (rx is None or rx.search(path.name) is not None)
        )
        return [str(path)] if keep else []
    uris = []
    for p in sorted(path.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(path).as_posix()
        if prefix is not None and not rel.startswith(prefix):
            continue
        if suffix is not None and not p.name.endswith(suffix):
            continue
        if rx is not None and not rx.search(rel):
            continue
        uris.append(str(p))
        if limit is not None and len(uris) >= limit:
            break
    return uris


class _S3SeekableReader:
    """A minimal seekable, read-only file object over an S3 object.

    ``zipfile`` reads an archive's end-of-central-directory by seeking to the
    tail, so listing a remote zip's members needs only a seekable reader that
    fetches byte ranges on demand — never the whole (multi-hundred-MB) archive.
    Implements just enough of the IO protocol (``read``/``seek``/``tell``) for
    that.
    """

    def __init__(self, client, bucket: str, key: str, size: int) -> None:
        self._client = client
        self._bucket = bucket
        self._key = key
        self._size = size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self._size + offset
        else:  # pragma: no cover - defensive
            raise ValueError(f"invalid whence {whence!r}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size - 1 if size is None or size < 0 else self._pos + size - 1
        end = min(end, self._size - 1)
        rng = f"bytes={self._pos}-{end}"
        body = self._client.get_object(Bucket=self._bucket, Key=self._key, Range=rng)[
            "Body"
        ].read()
        self._pos += len(body)
        return body

    def close(self) -> None:
        """Release the reader. A no-op: the boto3 client is shared and each
        ``read`` is a self-contained ranged GET, so there is nothing to free.
        Present because callers (and ``zipfile``) close the file object, and a
        local file's object has ``close`` — the S3 reader must match the
        protocol or remote archives fail with ``AttributeError``.
        """

    def __enter__(self) -> _S3SeekableReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def zip_source_uri(vsi_path: str) -> str | None:
    """Extract the container URI from a ``/vsizip/`` GDAL path, or ``None``.

    Maps ``/vsizip//vsis3/bucket/key.zip/member`` → ``s3://bucket/key.zip``
    and ``/vsizip//local/path.zip/member`` → ``/local/path.zip``.
    Used by the runner to resolve a zip-delivered product's delivery object
    for ``bytes_in`` sizing — the zip is the honest storage footprint, and
    sizing it once (not per member) avoids N×zip_size double-counting.
    """
    if not vsi_path.startswith("/vsizip/"):
        return None
    inner = vsi_path[len("/vsizip/") :]
    dot_zip = inner.find(".zip/")
    if dot_zip == -1:
        if inner.endswith(".zip"):
            zip_gdal = inner
        else:
            return None
    else:
        zip_gdal = inner[: dot_zip + len(".zip")]
    if zip_gdal.startswith("/vsis3/"):
        return "s3://" + zip_gdal[len("/vsis3/") :]
    return zip_gdal


def open_seekable(uri: str, role: str = "source"):
    """Open ``uri`` as a seekable binary file object (local file or S3 range).

    The caller is responsible for closing the returned object. For S3 the reader
    fetches ranges lazily, so opening a large archive to read its directory is
    cheap.
    """
    if is_s3(uri):
        loc = _parse_s3(uri)
        _require_object_key(loc, uri)
        client = _s3_client(role)
        size = int(client.head_object(Bucket=loc.bucket, Key=loc.key)["ContentLength"])
        return _S3SeekableReader(client, loc.bucket, loc.key, size)
    return _local_path(uri).open("rb")


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


def upload_tree(local_dir: str, uri: str, role: str = "sink") -> None:
    """Upload every file under ``local_dir`` to ``uri`` (a prefix), preserving paths.

    A store format (GeoZarr) produces a *directory* of many objects rather than a
    single file, so the runner publishes it as a tree: each file is uploaded under
    ``uri`` at its path relative to ``local_dir`` (POSIX-joined for S3, a recursive
    copy locally). ``uri`` is treated as a prefix/directory, not an object key.

    The destination prefix is cleared first, so re-publishing to a reused store
    path (the runner's deterministic ``.../<format>/<basename>``) cannot leave
    stale shard objects from a previous run alongside the new ones — which would
    make the published store inconsistent with what was just produced.
    """
    src = Path(local_dir)
    if not src.exists():
        raise FileNotFoundError(f"no such directory: {local_dir}")
    if not src.is_dir():
        raise NotADirectoryError(f"not a directory: {local_dir}")
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"no files to upload under {local_dir}")
    if is_s3(uri):
        loc = _parse_s3(uri)
        base = loc.key.rstrip("/")
        client = _s3_client(role)
        _clear_s3_prefix(client, loc.bucket, base)
        for p in files:
            rel = p.relative_to(src).as_posix()
            key = f"{base}/{rel}" if base else rel
            client.upload_file(str(p), loc.bucket, key)
        return
    dest = _local_path(uri)
    if dest.exists():
        shutil.rmtree(dest)
    for p in files:
        rel = p.relative_to(src)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)


def _clear_s3_prefix(client, bucket: str, base: str) -> None:
    """Delete every object under ``base/`` in ``bucket`` (a store's own prefix)."""
    prefix = f"{base}/" if base else ""
    paginator = client.get_paginator("list_objects_v2")
    batch: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            batch.append({"Key": obj["Key"]})
            if len(batch) == 1000:  # S3 delete_objects caps at 1000 keys
                client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                batch = []
    if batch:
        client.delete_objects(Bucket=bucket, Delete={"Objects": batch})


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
    A point-cloud ``PIXC:<granule>::<group>?…`` component is sized by its
    underlying granule object, so the point-cloud arm reports ``bytes_in`` too.
    """
    if uri.startswith("PIXC:"):
        granule = uri[len("PIXC:") :].split("::", 1)[0]
        return object_size(granule, role)
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
