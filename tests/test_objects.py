"""Tests for the object-size profiler — the differentiating metric."""

import pytest

from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.tiers import Tier, TierPolicy

MB = 1024 * 1024

POLICY = TierPolicy(
    tiers=(
        Tier(name="warm", min_object_bytes=32 * MB),
        Tier(name="cold", min_object_bytes=100 * MB),
    )
)


def test_summary_statistics_on_known_sizes():
    profile = profile_object_sizes([10, 20, 30, 40], POLICY)
    assert profile.count == 4
    assert profile.total_bytes == 100
    assert profile.mean == 25.0
    assert profile.median == 25.0
    assert profile.p50 == 25.0
    # Linear interpolation (type-7), rank = q/100 * (n-1).
    assert profile.p90 == pytest.approx(37.0)
    assert profile.p95 == pytest.approx(38.5)
    assert profile.p99 == pytest.approx(39.7)
    assert profile.min_bytes == 10
    assert profile.max_bytes == 40


def test_histogram_counts_sum_to_count():
    sizes = [10, 20, 30, 40, 50, 60]
    profile = profile_object_sizes(sizes, POLICY)
    assert sum(b.count for b in profile.histogram) == len(sizes)
    # Default edges are powers of two and span the data.
    assert profile.histogram[0].lower <= min(sizes)
    assert profile.histogram[-1].upper > max(sizes)


def test_explicit_bins_clamp_and_sum():
    profile = profile_object_sizes([5, 15, 25], POLICY, bins=[0, 10, 20, 100])
    counts = [b.count for b in profile.histogram]
    assert counts == [1, 1, 1]
    assert sum(counts) == 3


def test_tiny_objects_fit_no_tier():
    profile = profile_object_sizes([1 * MB, 2 * MB], POLICY)
    assert profile.tier_fit == []
    assert profile.highest_tier is None


def test_large_objects_fit_cold_tier():
    profile = profile_object_sizes([200 * MB, 200 * MB, 200 * MB], POLICY)
    assert profile.tier_fit == ["warm", "cold"]
    assert profile.highest_tier == "cold"


def test_single_object():
    profile = profile_object_sizes([5], POLICY)
    assert profile.count == 1
    assert profile.mean == 5.0
    assert profile.p50 == profile.p99 == 5.0
    assert sum(b.count for b in profile.histogram) == 1


def test_all_equal_sizes():
    profile = profile_object_sizes([7, 7, 7, 7], POLICY)
    assert profile.mean == 7.0
    assert profile.min_bytes == profile.max_bytes == 7
    assert sum(b.count for b in profile.histogram) == 4


def test_mean_below_threshold_does_not_round_up_into_tier():
    # Mean is 32 MiB - 1 byte: must NOT be reported as fitting the warm tier
    # even though it rounds to 32 MiB.
    just_under = 32 * MB - 1
    profile = profile_object_sizes([just_under], POLICY)
    assert profile.tier_fit == []
    assert profile.highest_tier is None


def test_empty_input_raises():
    with pytest.raises(ValueError):
        profile_object_sizes([], POLICY)


def test_negative_size_raises():
    with pytest.raises(ValueError):
        profile_object_sizes([10, -1, 20], POLICY)


def test_too_few_explicit_edges_raises():
    with pytest.raises(ValueError):
        profile_object_sizes([1, 2, 3], POLICY, bins=[10])


def test_unsorted_explicit_edges_raise():
    with pytest.raises(ValueError):
        profile_object_sizes([1, 2, 3], POLICY, bins=[0, 20, 10])
