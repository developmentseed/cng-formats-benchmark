"""Datasets — layout-aware enumeration of products and components.

Dataset implementations are constructed from config and registered into
:data:`cng_benchmark.registry.DATASETS` by ``reader`` name. Importing this
package registers the built-in readers (mirroring
:mod:`cng_benchmark.formats`); :func:`build_dataset` resolves a
:class:`~cng_benchmark.config.DatasetConfig` to a constructed, options-validated
:class:`Dataset`.

Adding a layout = a new subclass + its typed ``Options`` + one ``@DATASETS.register``
line in its module, imported here — no change to the core config or runner.
"""

from __future__ import annotations

from cng_benchmark.config import DatasetConfig

# Import the built-in readers for their registration side effects.
from cng_benchmark.datasets import (
    sentinel1,  # noqa: F401,E402
    sentinel2,  # noqa: F401,E402
    single_object,  # noqa: F401,E402
    zip_delivery,  # noqa: F401,E402
)
from cng_benchmark.datasets.base import Dataset, DatasetOptions, Product, SourceObject
from cng_benchmark.registry import DATASETS


def build_dataset(config: DatasetConfig) -> Dataset:
    """Resolve and construct the :class:`Dataset` for ``config``.

    Looks the reader up in :data:`DATASETS` (``KeyError`` lists the registered
    readers for an unknown one) and constructs it, which validates
    ``config.options`` against that reader's typed ``Options`` model.
    """
    cls = DATASETS.get(config.reader)
    return cls(config)


__all__ = [
    "Dataset",
    "DatasetOptions",
    "Product",
    "SourceObject",
    "build_dataset",
]
