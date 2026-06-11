"""Tests for the storage-tier fitness model."""

from cng_benchmark.tiers import Tier, TierPolicy

MB = 1024 * 1024

# An example two-tier policy (a warm disk tier and a colder archive tier).
POLICY = TierPolicy(
    tiers=(
        Tier(name="warm", min_object_bytes=32 * MB),
        Tier(name="cold", min_object_bytes=100 * MB),
    )
)


def test_fits_no_tier_when_objects_too_small():
    assert POLICY.fit(1 * MB) == []
    assert POLICY.highest_fit(1 * MB) is None


def test_fits_warm_tier_only():
    assert POLICY.fit(50 * MB) == ["warm"]
    assert POLICY.highest_fit(50 * MB) == "warm"


def test_fits_both_tiers():
    assert POLICY.fit(150 * MB) == ["warm", "cold"]
    assert POLICY.highest_fit(150 * MB) == "cold"


def test_threshold_is_inclusive():
    assert "warm" in POLICY.fit(32 * MB)
    assert "cold" in POLICY.fit(100 * MB)
