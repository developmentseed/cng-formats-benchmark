"""Write metric — conversion throughput.

Times the baseline → target conversion performed by a format adapter and reports
elapsed time and throughput (bytes of output per second). The conversion itself
is the measured work, so the runner always calls this to produce the target
object; whether the resulting metrics are reported is a config choice.

When the source is read over the network (a GDAL ``/vsis3`` path, including a
``/vsizip`` archive member), the elapsed time deliberately includes that read —
the extraction/transfer cost is part of what a real migration pays, so it is
measured rather than hidden behind a pre-download.
"""

from __future__ import annotations

import os
import time

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.models import MetricResult


def _output_size(path: str) -> int:
    """Total bytes written at ``path`` — a single file or a store directory.

    A raster adapter writes one file; a store adapter (GeoZarr) writes a directory
    tree, so the output size is the sum of every file beneath it.
    """
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(path)
            for f in files
        )
    return os.path.getsize(path)


def measure_write(
    adapter: FormatAdapter,
    source_path: str,
    target_path: str,
    params: dict,
    *,
    source_size: int | None = None,
) -> list[MetricResult]:
    """Convert ``source_path`` to ``target_path`` and return write metrics.

    ``source_path`` is GDAL-readable (a local path or a ``/vsis3`` URI).
    ``source_size`` is the input byte size for the ``bytes_in`` detail when it is
    known (a remote source may not have a cheap size); throughput is computed
    from the output size regardless. Side effect: writes the target object.
    """
    if source_size is None and os.path.isfile(source_path):
        source_size = os.path.getsize(source_path)

    start = time.perf_counter()
    adapter.convert(source_path, target_path, params)
    elapsed = time.perf_counter() - start

    size_out = _output_size(target_path)
    throughput = size_out / elapsed if elapsed > 0 else float("inf")
    detail: dict = {"bytes_out": size_out}
    if source_size is not None:
        detail["bytes_in"] = source_size
    return [
        MetricResult(name="write_elapsed", value=elapsed, unit="s"),
        MetricResult(
            name="write_throughput", value=throughput, unit="bytes/s", detail=detail
        ),
    ]
