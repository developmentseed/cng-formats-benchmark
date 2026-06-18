"""Single-object dataset — the default reader.

The fallback layout: one product whose single component *is* ``source``. This
keeps every pre-existing single-file descriptor working unchanged (one product,
one component) and is what ``reader: single-object`` (the config default)
selects. It takes no options.
"""

from __future__ import annotations

from cng_benchmark.datasets.base import Dataset, Product, SourceObject
from cng_benchmark.registry import DATASETS


@DATASETS.register("single-object")
class SingleObjectDataset(Dataset):
    """One product, one component = ``source``."""

    def products(
        self, *, prefix: str | None = None, limit: int | None = None
    ) -> list[Product]:
        component = SourceObject(name=self.id, uri=self.source_uri)
        return [Product(id=self.id, components=[component])]
