"""Metric collectors.

Each collector turns observations about a converted dataset into a result that
slots into :class:`cng_benchmark.models.BenchmarkRun`. The object-size profiler
(:mod:`cng_benchmark.metrics.objects`) is the only collector that needs no
service or live IO, so it is the one M1 implements; read/write/display metrics
need TiTiler and real object storage and arrive with the deployable stack (M2).
"""

from __future__ import annotations
