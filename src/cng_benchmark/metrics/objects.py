"""Object-size profiler — the system's differentiating metric.

Cloud-native formats trade off how they group bytes into addressable objects
(COG internal tiling, Zarr v3 sharding, COPC octree nodes, GeoParquet row
groups). That grouping decides whether a layout is viable on a tiered object
store, because tiers impose a minimum recommended *mean object size*. This
collector reduces a list of produced object sizes to an
:class:`~cng_benchmark.models.ObjectSizeProfile` — distribution summary plus
tier fitness — without touching any storage backend, so it is fully unit
testable on its own.

Percentiles use linear interpolation between closest ranks (the same "type 7"
method as ``numpy.percentile``), implemented here in pure stdlib to keep the
harness dependency-light.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from statistics import median as _median

from cng_benchmark.models import HistogramBin, ObjectSizeProfile
from cng_benchmark.tiers import TierPolicy


def _percentile(sorted_sizes: list[int], q: float) -> float:
    """Return the ``q``-th percentile (0..100) of an already-sorted list."""
    if not sorted_sizes:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_sizes) == 1:
        return float(sorted_sizes[0])
    rank = (q / 100.0) * (len(sorted_sizes) - 1)
    lower = math.floor(rank)
    frac = rank - lower
    if frac == 0:
        return float(sorted_sizes[lower])
    return sorted_sizes[lower] + frac * (sorted_sizes[lower + 1] - sorted_sizes[lower])


def _power_of_two_edges(min_size: int, max_size: int) -> list[int]:
    """Default log-scale (power-of-two) bin edges spanning ``min..max``.

    The lowest edge is at or below ``min_size`` and the highest is strictly
    above ``max_size``, so every observed size falls inside a bin.
    """
    e_min = int(math.floor(math.log2(max(min_size, 1))))
    e_max = int(math.floor(math.log2(max(max_size, 1))))
    return [2**e for e in range(e_min, e_max + 2)]


def _histogram(sorted_sizes: list[int], edges: list[int]) -> list[HistogramBin]:
    """Bucket sizes into half-open ``[lower, upper)`` bins from ``edges``.

    Each size is placed by its position among the edges and clamped to the
    valid bin range, so the bin counts always sum to ``len(sorted_sizes)`` even
    if explicit edges do not fully span the data.
    """
    n_bins = len(edges) - 1
    counts = [0] * n_bins
    for size in sorted_sizes:
        idx = bisect_right(edges, size) - 1
        idx = max(0, min(idx, n_bins - 1))
        counts[idx] += 1
    return [
        HistogramBin(lower=edges[i], upper=edges[i + 1], count=counts[i])
        for i in range(n_bins)
    ]


def profile_object_sizes(
    sizes: list[int],
    policy: TierPolicy,
    *,
    bins: list[int] | None = None,
) -> ObjectSizeProfile:
    """Reduce object sizes (in bytes) to an :class:`ObjectSizeProfile`.

    ``policy`` evaluates tier fitness against the mean object size. ``bins``
    optionally overrides the default power-of-two histogram edges (a sorted list
    of at least two edge values); otherwise log-scale edges spanning the data
    are used. Raises :class:`ValueError` on an empty input.
    """
    if not sizes:
        raise ValueError("cannot profile an empty list of object sizes")
    if bins is not None and len(bins) < 2:
        raise ValueError("explicit bin edges need at least two values")

    sorted_sizes = sorted(sizes)
    count = len(sorted_sizes)
    total = sum(sorted_sizes)
    mean = total / count
    edges = (
        bins
        if bins is not None
        else _power_of_two_edges(sorted_sizes[0], sorted_sizes[-1])
    )

    mean_int = round(mean)
    return ObjectSizeProfile(
        count=count,
        total_bytes=total,
        mean=mean,
        median=_median(sorted_sizes),
        p50=_percentile(sorted_sizes, 50),
        p90=_percentile(sorted_sizes, 90),
        p95=_percentile(sorted_sizes, 95),
        p99=_percentile(sorted_sizes, 99),
        min_bytes=sorted_sizes[0],
        max_bytes=sorted_sizes[-1],
        histogram=_histogram(sorted_sizes, edges),
        tier_fit=policy.fit(mean_int),
        highest_tier=policy.highest_fit(mean_int),
    )
