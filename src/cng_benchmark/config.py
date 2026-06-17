"""Configuration schema and loaders.

Datasets and benchmark runs are described as data, not code: a dataset config
names its source, baseline format, candidate target formats, and the grouping
lever to sweep; a benchmark config names which dataset and formats to exercise,
which metrics to collect, and the storage-tier policy to judge object-size
fitness against. Validation is pydantic; configs are loaded from YAML.

Keeping these as config (rather than hard-coding any particular dataset) is what
makes the system reusable: adding a dataset or format is a new file here plus a
registration, never a change to the harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from cng_benchmark.tiers import Tier, TierPolicy


class DatasetConfig(BaseModel):
    """Descriptor for one dataset."""

    id: str
    source: str
    baseline_format: str
    target_formats: list[str]
    grouping_lever: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class TierConfig(BaseModel):
    """A storage tier and its minimum recommended mean object size (bytes)."""

    name: str
    min_object_bytes: int


class BenchmarkConfig(BaseModel):
    """Descriptor for one benchmark run.

    ``object_source`` and ``output`` are optional location URIs (a local path,
    ``file://`` or ``s3://…``): where the deployed runner reads the objects to
    profile and where it writes its result artifacts. They are kept generic so
    the same config is portable across targets — the deployment supplies the
    concrete URIs (via CLI flags or a ConfigMap), and the CLI flags override
    whatever the config carries.
    """

    id: str
    dataset: str
    formats: list[str]
    metrics: list[str]
    tiers: list[TierConfig]
    params: dict[str, Any] = Field(default_factory=dict)
    object_source: str | None = None
    output: str | None = None


def tier_policy_from_config(tiers: list[TierConfig]) -> TierPolicy:
    """Build a :class:`TierPolicy` from configured tiers (ordered by minimum)."""
    ordered = sorted(tiers, key=lambda t: t.min_object_bytes)
    return TierPolicy(
        tiers=tuple(
            Tier(name=t.name, min_object_bytes=t.min_object_bytes) for t in ordered
        )
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        kind = type(data).__name__
        raise ValueError(f"config {path} must be a YAML mapping, got {kind}")
    return data


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load and validate a dataset config from a YAML file."""
    return DatasetConfig.model_validate(_load_yaml(path))


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark config from a YAML file."""
    return BenchmarkConfig.model_validate(_load_yaml(path))
