"""Benchmark orchestration — the harness core.

``run_benchmark`` is the in-process runner that the container entry point and
the CLI both delegate to. It turns a validated config plus a set of object sizes
into a :class:`~cng_benchmark.models.BenchmarkRun`: it resolves the format
adapter by name from the registry (the plug-in seam), builds the tier policy,
runs the object-size metric, and stamps the result with run context.

It is deliberately free of services and live IO. In M1 the format adapters are
stubs, so the sizes are supplied by the caller (a local object listing) rather
than produced by a real conversion; M2 swaps the stub ``enumerate_objects`` for
real format output behind the same seam.
"""

from __future__ import annotations

from datetime import UTC, datetime

import cng_benchmark.formats  # noqa: F401  (registers the built-in adapters)
from cng_benchmark import __version__
from cng_benchmark.config import BenchmarkConfig, tier_policy_from_config
from cng_benchmark.metrics.objects import profile_object_sizes
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
