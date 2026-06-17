"""Write metric — conversion throughput.

Times the baseline → target conversion performed by a format adapter and reports
elapsed time and throughput (bytes of output per second). The conversion itself
is the measured work, so the runner always calls this to produce the target
object; whether the resulting metrics are reported is a config choice.
"""

from __future__ import annotations

import os
import time

from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.models import MetricResult


def measure_write(
    adapter: FormatAdapter,
    source_path: str,
    target_path: str,
    params: dict,
) -> list[MetricResult]:
    """Convert ``source_path`` to ``target_path`` and return write metrics.

    Side effect: writes the target object (the runner relies on this for the
    object/read/display metrics even when write metrics are not reported).
    """
    size_in = os.path.getsize(source_path)
    start = time.perf_counter()
    adapter.convert(source_path, target_path, params)
    elapsed = time.perf_counter() - start
    size_out = os.path.getsize(target_path)
    throughput = size_out / elapsed if elapsed > 0 else float("inf")
    return [
        MetricResult(name="write_elapsed", value=elapsed, unit="s"),
        MetricResult(
            name="write_throughput",
            value=throughput,
            unit="bytes/s",
            detail={"bytes_in": size_in, "bytes_out": size_out},
        ),
    ]
