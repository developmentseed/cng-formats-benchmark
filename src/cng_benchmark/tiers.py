"""Storage-tier policy for object-size fitness.

Tiered object stores often recommend a minimum *mean object size* per
storage class (for example, a warm disk tier and a colder archive tier),
because many tiny objects degrade throughput and metadata performance.
Object size is therefore a first-class benchmark output: a format layout
is only viable on a tier if its mean object size clears that tier's
threshold while still allowing partial access via HTTP range requests.

Thresholds are policy, not physics, so they are configuration. This module
provides the generic model; concrete thresholds come from a benchmark's
config file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """A storage tier with a minimum recommended mean object size."""

    name: str
    min_object_bytes: int


@dataclass(frozen=True)
class TierPolicy:
    """An ordered set of storage tiers.

    Tiers are ordered from most granular (smallest minimum object size) to
    least. A layout "fits" a tier when its mean object size meets or exceeds
    that tier's minimum.
    """

    tiers: tuple[Tier, ...]

    def fit(self, mean_object_bytes: int) -> list[str]:
        """Return the names of the tiers a given mean object size satisfies."""
        return [
            tier.name
            for tier in self.tiers
            if mean_object_bytes >= tier.min_object_bytes
        ]

    def highest_fit(self, mean_object_bytes: int) -> str | None:
        """Return the coldest tier a layout fits, or ``None`` if it fits none.

        The coldest fitting tier is the one with the largest minimum object
        size that the layout still satisfies.
        """
        satisfied = [
            tier for tier in self.tiers if mean_object_bytes >= tier.min_object_bytes
        ]
        if not satisfied:
            return None
        return max(satisfied, key=lambda tier: tier.min_object_bytes).name
