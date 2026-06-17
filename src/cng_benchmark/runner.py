"""Benchmark orchestration — the harness core.

Two entry points, both resolving the format adapter by name from the registry
(the plug-in seam) and stamping the result with run context:

* :func:`run_benchmark` — profile a *given* list of object sizes (no conversion,
  no live IO). The object-size-only path the CLI uses for an object listing.
* :func:`run_conversion_benchmark` — the COG end-to-end path: convert a baseline
  raster to the target format, then run the requested collectors (write, object
  size, read, display) against the produced object. This one does live IO and
  talks to object storage and TiTiler, so it lives behind the same seam but is
  exercised by the deployed runner rather than in unit tests.

Which metrics run is config-driven (``config.metrics``); adding a format is a
new registered adapter, never a change here.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime

import cng_benchmark.formats  # noqa: F401  (registers the built-in adapters)
from cng_benchmark import __version__, storage
from cng_benchmark.config import BenchmarkConfig, tier_policy_from_config
from cng_benchmark.metrics.display import measure_display
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.metrics.read import measure_read
from cng_benchmark.metrics.write import measure_write
from cng_benchmark.models import BenchmarkRun, MetricResult
from cng_benchmark.registry import FORMATS


def run_benchmark(
    config: BenchmarkConfig,
    sizes: list[int],
    *,
    format_id: str | None = None,
) -> BenchmarkRun:
    """Profile ``sizes`` for one configured format and return a BenchmarkRun.

    ``format_id`` selects which of the config's formats to attribute the result
    to; it defaults to the first listed format. Resolving it through
    :data:`FORMATS` raises ``KeyError`` for an unknown format, which is the
    registry seam in action.
    """
    if not config.formats:
        raise ValueError(f"benchmark {config.id!r} lists no formats")
    chosen = format_id or config.formats[0]

    adapter = FORMATS.get(chosen)()
    policy = tier_policy_from_config(config.tiers)
    profile = profile_object_sizes(sizes, policy)

    params = {**config.params, "grouping_lever": adapter.describe_grouping_lever()}
    return BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        metrics=[
            MetricResult(name="object_count", value=profile.count),
            MetricResult(name="total_bytes", value=profile.total_bytes, unit="bytes"),
        ],
    )


def run_conversion_benchmark(
    config: BenchmarkConfig,
    source_uri: str,
    output_uri: str,
    *,
    titiler_endpoint: str | None = None,
    format_id: str | None = None,
) -> BenchmarkRun:
    """Convert ``source_uri`` to the target format and run the configured metrics.

    Pipeline: download the baseline raster, convert it (timing the *write*), and
    — for the metrics named in ``config.metrics`` — profile the produced object's
    size, read windows back over range requests, and time TiTiler tiles. The
    produced object is uploaded under ``output_uri`` so the read/display metrics
    (and a real consumer) can address it on the store.

    ``read`` reads from the uploaded object (S3 ``/vsis3`` range requests when
    ``output_uri`` is S3); ``display`` requires both a ``titiler_endpoint`` and
    an S3 ``output_uri`` (TiTiler reads the object from the store).
    """
    if not config.formats:
        raise ValueError(f"benchmark {config.id!r} lists no formats")
    chosen = format_id or config.formats[0]
    adapter = FORMATS.get(chosen)()
    requested = set(config.metrics)

    with tempfile.TemporaryDirectory() as workdir:
        src_name = os.path.basename(source_uri.rstrip("/")) or "source.tif"
        local_source = os.path.join(workdir, src_name)
        storage.write_bytes(local_source, storage.read_bytes(source_uri))

        local_target = os.path.join(workdir, f"{chosen}.tif")
        write_metrics = measure_write(
            adapter, local_source, local_target, config.params
        )

        # Publish the produced object so read/display can address it on the store.
        object_uri = storage.join(output_uri, f"{chosen}/{chosen}.tif")
        needs_published = bool({"read", "display"} & requested)
        if needs_published:
            storage.write_bytes(object_uri, storage.read_bytes(local_target))

        policy = tier_policy_from_config(config.tiers)
        profile = profile_object_sizes(adapter.enumerate_objects(local_target), policy)

        metrics: list[MetricResult] = []
        if "write" in requested:
            metrics += write_metrics
        if "object_size" in requested:
            metrics += [
                MetricResult(name="object_count", value=profile.count),
                MetricResult(
                    name="total_bytes", value=profile.total_bytes, unit="bytes"
                ),
            ]
        if "read" in requested:
            read_uri = object_uri if needs_published else local_target
            metrics += measure_read(read_uri)
        if "display" in requested:
            if not titiler_endpoint:
                raise ValueError("the display metric requires a TiTiler endpoint")
            if not storage.is_s3(object_uri):
                raise ValueError("the display metric requires an S3 output location")
            metrics += measure_display(titiler_endpoint, object_uri)

    params = {**config.params, "grouping_lever": adapter.describe_grouping_lever()}
    return BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        metrics=metrics,
    )
