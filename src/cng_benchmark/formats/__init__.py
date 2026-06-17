"""Format adapters.

Importing this package registers the built-in adapters into
:data:`cng_benchmark.registry.FORMATS` as a side effect, so the runner can
resolve a format by name from a config file.
"""

from __future__ import annotations

from cng_benchmark.formats import (  # noqa: F401
    cog,  # noqa: F401
    copc,
    geoparquet,
    geozarr,
)
from cng_benchmark.formats.base import FormatAdapter

__all__ = ["FormatAdapter"]
