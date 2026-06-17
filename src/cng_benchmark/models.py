"""Result schema for a benchmark run.

These models are the system's serialisable output: a benchmark run produces a
:class:`BenchmarkRun` capturing *what was measured, against what, and when*, so
results from different datasets, formats, and tool versions remain comparable
over time. They are deliberately free of any service or IO dependency — the
harness assembles them, and the deployment (M2) persists them.

The headline payload is the :class:`ObjectSizeProfile`. Object size is a hard
constraint on tiered object stores (see :mod:`cng_benchmark.tiers`), so the
profile is a first-class result rather than an incidental statistic.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HistogramBin(BaseModel):
    """A single half-open ``[lower, upper)`` object-size histogram bucket."""

    lower: int
    upper: int
    count: int


class ObjectSizeProfile(BaseModel):
    """Summary of the sizes of the objects a format layout produces.

    Percentiles are reported in bytes. ``p50`` is identical to ``median`` and
    kept alongside the other percentiles for convenience. Tier fitness is
    derived from the *mean* object size against a configured policy: ``tier_fit``
    lists every tier the layout satisfies and ``highest_tier`` is the coldest of
    those (or ``None`` if the objects are too small for any tier).
    """

    count: int
    total_bytes: int
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    min_bytes: int
    max_bytes: int
    histogram: list[HistogramBin]
    tier_fit: list[str]
    highest_tier: str | None


class MetricResult(BaseModel):
    """A single named measurement produced by a metric collector."""

    name: str
    value: float
    unit: str | None = None
    detail: dict = Field(default_factory=dict)


class BenchmarkRun(BaseModel):
    """The full, serialisable record of one benchmark run.

    Captures the run context needed to interpret and compare results: when it
    ran, the versions of the tools involved, and which dataset/format/params
    were exercised. ``object_profile`` carries the object-size differentiator and
    ``metrics`` holds any additional scalar measurements.
    """

    timestamp: datetime
    tool_versions: dict[str, str] = Field(default_factory=dict)
    dataset_id: str
    format_id: str
    params: dict = Field(default_factory=dict)
    object_profile: ObjectSizeProfile | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
