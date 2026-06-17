"""Datasets.

Dataset implementations are constructed from config and registered into
:data:`cng_benchmark.registry.DATASETS`. M1 ships only the contract
(:class:`cng_benchmark.datasets.base.Dataset`); concrete, storage-backed
datasets arrive with the deployable stack.
"""

from __future__ import annotations

from cng_benchmark.datasets.base import Dataset

__all__ = ["Dataset"]
