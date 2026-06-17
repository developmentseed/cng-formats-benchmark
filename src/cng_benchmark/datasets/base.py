"""Dataset contract.

A :class:`Dataset` is the harness's handle on the source data named by a
:class:`~cng_benchmark.config.DatasetConfig`. It exposes the config-derived
identity (id, baseline and target formats) generically and defines the hook the
runner uses to make the baseline data locally available for conversion.

Materialisation touches real storage, so it is deferred to the deployable stack
(M2); M1 fixes the contract and the config-to-object construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cng_benchmark.config import DatasetConfig


class Dataset(ABC):
    """A source dataset constructed from its config."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def baseline_format(self) -> str:
        return self.config.baseline_format

    @property
    def target_formats(self) -> list[str]:
        return self.config.target_formats

    @property
    def source_uri(self) -> str:
        return self.config.source

    @abstractmethod
    def materialize(self) -> str:
        """Make the baseline data locally available and return its path."""
