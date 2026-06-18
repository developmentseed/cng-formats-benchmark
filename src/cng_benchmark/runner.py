"""Benchmark orchestration — the harness core.

Two entry points, both resolving the format adapter by name from the registry
(the plug-in seam) and stamping the result with run context:

* :func:`run_benchmark` — profile a *given* list of object sizes (no conversion,
  no live IO). The object-size-only path the CLI uses for an object listing.
* :func:`run_conversion_benchmark` — the COG end-to-end path: convert a baseline
  raster to the target format, then run the requested collectors (write, object
  size, read, display) against the produced object. This one does live IO and
  talks to object storage and TiTiler, so it lives behind the same seam but is
  exercised by the deployed runner rather than in unit tests.

Which metrics run is config-driven (``config.metrics``); adding a format is a
new registered adapter, never a change here.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime

import cng_benchmark.datasets  # noqa: F401  (registers the built-in readers)
import cng_benchmark.formats  # noqa: F401  (registers the built-in adapters)
from cng_benchmark import __version__, storage
from cng_benchmark.config import BenchmarkConfig, DatasetConfig, tier_policy_from_config
from cng_benchmark.datasets import Product, build_dataset
from cng_benchmark.formats.base import FormatAdapter
from cng_benchmark.gdal_env import gdal_session
from cng_benchmark.metrics.display import measure_display
from cng_benchmark.metrics.layout import describe_object_layout
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.metrics.read import measure_read
from cng_benchmark.metrics.write import measure_write
from cng_benchmark.models import BenchmarkRun, MetricResult, ObjectLayout
from cng_benchmark.registry import FORMATS


def _safe_object_layout(name: str, path: str) -> ObjectLayout | None:
    """Describe the produced object's tiling layout, or ``None`` if unavailable.

    The layout is a best-effort structural extra: a non-raster output (or a
    missing geo stack) yields ``None`` rather than failing the run, the same way
    the display layout image is best-effort.
    """
    try:
        return describe_object_layout(name, path, os.path.getsize(path))
    except Exception:  # noqa: BLE001 - structural extra; never fail the run for it
        return None


def run_benchmark(
    config: BenchmarkConfig,
    sizes: list[int],
    *,
    format_id: str | None = None,
) -> BenchmarkRun:
    """Profile ``sizes`` for one configured format and return a BenchmarkRun.

    ``format_id`` selects which of the config's formats to attribute the result
    to; it defaults to the first listed format. Resolving it through
    :data:`FORMATS` raises ``KeyError`` for an unknown format, which is the
    registry seam in action.
    """
    if not config.formats:
        raise ValueError(f"benchmark {config.id!r} lists no formats")
    chosen = format_id or config.formats[0]

    adapter = FORMATS.get(chosen)()
    policy = tier_policy_from_config(config.tiers)
    profile = profile_object_sizes(sizes, policy)

    params = {**config.params, "grouping_lever": adapter.describe_grouping_lever()}
    return BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        metrics=[
            MetricResult(name="object_count", value=profile.count),
            MetricResult(name="total_bytes", value=profile.total_bytes, unit="bytes"),
        ],
    )


def run_conversion_benchmark(
    config: BenchmarkConfig,
    source_uri: str,
    output_uri: str,
    *,
    titiler_endpoint: str | None = None,
    format_id: str | None = None,
) -> BenchmarkRun:
    """Convert ``source_uri`` to the target format and run the configured metrics.

    Pipeline: download the baseline raster, convert it (timing the *write*), and
    — for the metrics named in ``config.metrics`` — profile the produced object's
    size, read windows back over range requests, and time TiTiler tiles. The
    produced object is uploaded under ``output_uri`` so the read/display metrics
    (and a real consumer) can address it on the store.

    ``read`` reads from the uploaded object (S3 ``/vsis3`` range requests when
    ``output_uri`` is S3); ``display`` requires both a ``titiler_endpoint`` and
    an S3 ``output_uri`` (TiTiler reads the object from the store).

    The source is read in place via GDAL (the ``source`` role's endpoint/CA),
    so the conversion's source-read cost is measured, not laundered by a
    pre-download; the produced object and the read metric use the ``sink`` role.
    """
    if not config.formats:
        raise ValueError(f"benchmark {config.id!r} lists no formats")
    chosen = format_id or config.formats[0]
    adapter = FORMATS.get(chosen)()
    requested = set(config.metrics)

    with tempfile.TemporaryDirectory() as workdir:
        local_target = os.path.join(workdir, f"{chosen}.tif")
        # Read the source in place (network reads counted in the conversion).
        source_path = storage.to_gdal_path(source_uri)
        with gdal_session("source"):
            write_metrics = measure_write(
                adapter,
                source_path,
                local_target,
                config.params,
                source_size=storage.object_size(source_uri, "source"),
            )

        # Always publish the produced object under the output location: it is a
        # first-class run artifact, and read/display address it on the store.
        object_uri = storage.join(output_uri, f"{chosen}/{chosen}.tif")
        storage.upload_from_path(local_target, object_uri, role="sink")

        policy = tier_policy_from_config(config.tiers)
        profile = profile_object_sizes(adapter.enumerate_objects(local_target), policy)
        layout = _safe_object_layout(chosen, local_target)
        object_layouts = [layout] if layout is not None else []

        metrics: list[MetricResult] = []
        if "write" in requested:
            metrics += write_metrics
        if "object_size" in requested:
            metrics += [
                MetricResult(name="object_count", value=profile.count),
                MetricResult(
                    name="total_bytes", value=profile.total_bytes, unit="bytes"
                ),
            ]
        if "read" in requested:
            with gdal_session("sink"):
                metrics += measure_read(object_uri)
        if "display" in requested:
            if not titiler_endpoint:
                raise ValueError("the display metric requires a TiTiler endpoint")
            if not storage.is_s3(object_uri):
                raise ValueError("the display metric requires an S3 output location")
            from cng_benchmark.metrics.display_tiles import (
                DEFAULT_TARGETS,
                render_chunk_layout,
                select_chunk_tiles,
            )

            targets = tuple(config.params.get("display_chunk_targets", DEFAULT_TARGETS))
            # Select against the *local* COG (cheap, no network); time the tiles
            # against the uploaded S3 object via TiTiler.
            tiles = select_chunk_tiles(local_target, targets=targets)
            metrics += measure_display(titiler_endpoint, object_uri, tiles)

            # Publish a chunk-grid + tile-footprint layout image alongside the
            # object (best-effort: a missing matplotlib must not fail the run).
            try:
                local_layout = os.path.join(workdir, "display_chunk_layout.png")
                render_chunk_layout(local_target, tiles, local_layout)
                layout_uri = storage.join(
                    output_uri, f"{chosen}/display_chunk_layout.png"
                )
                storage.upload_from_path(local_layout, layout_uri, role="sink")
                for m in metrics:
                    if m.name == "display_scenarios":
                        m.detail["layout_uri"] = layout_uri
            except RuntimeError as exc:
                metrics.append(
                    MetricResult(
                        name="display_layout_skipped",
                        value=0,
                        detail={"reason": str(exc)},
                    )
                )

    params = {**config.params, "grouping_lever": adapter.describe_grouping_lever()}
    return BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        object_layouts=object_layouts,
        metrics=metrics,
    )


@dataclass(frozen=True)
class ProductSetResult:
    """The fan-out result: one run per product plus a pooled roll-up.

    ``per_product`` carries one :class:`BenchmarkRun` per scene (its profile is
    the object-size distribution over that scene's components); ``rollup`` pools
    every object across the set into one honest distribution. When the scope is a
    single product the set is that one product and ``rollup`` mirrors it.
    """

    per_product: list[BenchmarkRun]
    rollup: BenchmarkRun


def _aggregate_write_metrics(
    per_component: list[list[MetricResult]],
) -> list[MetricResult]:
    """Pool per-component write metrics into one product-level write result.

    Elapsed times sum (the product's total conversion wall time) and throughput
    is recomputed from the pooled output bytes over that total, so a product's
    write metric is comparable to a single object's.
    """
    total_elapsed = 0.0
    bytes_out = 0
    bytes_in = 0
    have_bytes_in = False
    for metrics in per_component:
        for m in metrics:
            if m.name == "write_elapsed":
                total_elapsed += m.value
            elif m.name == "write_throughput":
                bytes_out += int(m.detail.get("bytes_out", 0))
                if "bytes_in" in m.detail:
                    bytes_in += int(m.detail["bytes_in"])
                    have_bytes_in = True
    throughput = bytes_out / total_elapsed if total_elapsed > 0 else float("inf")
    detail: dict = {"bytes_out": bytes_out, "components": len(per_component)}
    if have_bytes_in:
        detail["bytes_in"] = bytes_in
    return [
        MetricResult(name="write_elapsed", value=total_elapsed, unit="s"),
        MetricResult(
            name="write_throughput", value=throughput, unit="bytes/s", detail=detail
        ),
    ]


def _run_product(
    adapter: FormatAdapter,
    product: Product,
    config: BenchmarkConfig,
    output_uri: str,
    *,
    titiler_endpoint: str | None,
    requested: set[str],
    samples: dict,
) -> tuple[BenchmarkRun, list[int]]:
    """Convert every component of ``product`` and assemble its BenchmarkRun.

    ``object_size`` + ``write`` cover **all** components; ``read`` and
    ``display`` run only on the first ``samples[...]`` components (a
    representative sample, default 1). Each produced object is uploaded and its
    local copy freed before the next component is converted, so local disk is
    bounded by one component at a time regardless of product size. Returns the
    run and the per-object sizes (for the roll-up to pool).
    """
    chosen = adapter.name
    read_samples = int(samples.get("read", 1))
    display_samples = int(samples.get("display", 1))

    sizes: list[int] = []
    layouts: list[ObjectLayout] = []
    write_per_component: list[list[MetricResult]] = []
    extra_metrics: list[MetricResult] = []

    with tempfile.TemporaryDirectory() as workdir:
        for i, component in enumerate(product.components):
            local_target = os.path.join(workdir, f"{component.name}.tif")
            source_path = storage.to_gdal_path(component.uri)
            with gdal_session("source"):
                write_per_component.append(
                    measure_write(
                        adapter,
                        source_path,
                        local_target,
                        config.params,
                        source_size=storage.object_size(component.uri, "source"),
                    )
                )

            object_uri = storage.join(
                output_uri, f"objects/{product.id}/{component.name}/{chosen}.tif"
            )
            storage.upload_from_path(local_target, object_uri, role="sink")
            sizes += adapter.enumerate_objects(local_target)
            # Capture the produced object's tiling layout (structural, per object).
            layout = _safe_object_layout(component.name, local_target)
            if layout is not None:
                layouts.append(layout)

            if "read" in requested and i < read_samples:
                with gdal_session("sink"):
                    extra_metrics += measure_read(object_uri)
            if "display" in requested and i < display_samples:
                component_dir = storage.join(
                    output_uri, f"objects/{product.id}/{component.name}"
                )
                extra_metrics += _measure_display_component(
                    config, local_target, object_uri, component_dir, titiler_endpoint
                )

            os.remove(local_target)

    policy = tier_policy_from_config(config.tiers)
    profile = profile_object_sizes(sizes, policy)

    metrics: list[MetricResult] = []
    if "write" in requested:
        metrics += _aggregate_write_metrics(write_per_component)
    if "object_size" in requested:
        metrics += [
            MetricResult(name="object_count", value=profile.count),
            MetricResult(name="total_bytes", value=profile.total_bytes, unit="bytes"),
        ]
    metrics += extra_metrics

    params = {
        **config.params,
        "grouping_lever": adapter.describe_grouping_lever(),
        "product_id": product.id,
        "scope": "product",
    }
    run = BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        object_layouts=layouts,
        metrics=metrics,
    )
    return run, sizes


def _measure_display_component(
    config: BenchmarkConfig,
    local_target: str,
    object_uri: str,
    component_dir: str,
    titiler_endpoint: str | None,
) -> list[MetricResult]:
    """Run the display metric for one component and publish its chunk layout.

    Selects chunk-crossing tiles against the local COG, times them via TiTiler
    against the uploaded object, and (best-effort, like the single-source path)
    renders the block-grid + tile-footprint ``display_chunk_layout.png`` next to
    the object so the produced object's tiling is visible alongside its metrics.
    """
    if not titiler_endpoint:
        raise ValueError("the display metric requires a TiTiler endpoint")
    if not storage.is_s3(object_uri):
        raise ValueError("the display metric requires an S3 output location")
    from cng_benchmark.metrics.display_tiles import (
        DEFAULT_TARGETS,
        render_chunk_layout,
        select_chunk_tiles,
    )

    targets = tuple(config.params.get("display_chunk_targets", DEFAULT_TARGETS))
    tiles = select_chunk_tiles(local_target, targets=targets)
    metrics = measure_display(titiler_endpoint, object_uri, tiles)

    try:
        import os as _os

        local_layout = _os.path.join(_os.path.dirname(local_target), "_layout.png")
        render_chunk_layout(local_target, tiles, local_layout)
        layout_uri = storage.join(component_dir, "display_chunk_layout.png")
        storage.upload_from_path(local_layout, layout_uri, role="sink")
        for m in metrics:
            if m.name == "display_scenarios":
                m.detail["layout_uri"] = layout_uri
    except RuntimeError as exc:
        metrics.append(
            MetricResult(
                name="display_layout_skipped", value=0, detail={"reason": str(exc)}
            )
        )
    return metrics


def run_dataset_benchmark(
    config: BenchmarkConfig,
    dataset_config: DatasetConfig,
    output_uri: str,
    *,
    titiler_endpoint: str | None = None,
    format_id: str | None = None,
) -> ProductSetResult:
    """Fan out a benchmark over a dataset's product(s) and pool a roll-up.

    The dataset's reader enumerates its products (``scope: product`` takes one,
    ``scope: product-set`` takes the set bounded by ``params.products``'s prefix
    + limit). Each product is converted component-by-component into one
    :class:`BenchmarkRun` (its object-size distribution); the roll-up pools every
    object across the set into one honest distribution. Reuses the result model
    throughout — ``params`` carries ``product_id`` / ``scope`` to tell the runs
    apart.
    """
    if not config.formats:
        raise ValueError(f"benchmark {config.id!r} lists no formats")
    chosen = format_id or config.formats[0]
    adapter = FORMATS.get(chosen)()
    requested = set(config.metrics)
    samples = dict(config.params.get("samples", {}))

    dataset = build_dataset(dataset_config)
    scope = config.params.get("scope", "product")
    bound = dict(config.params.get("products", {}))
    prefix = bound.get("prefix")
    limit = bound.get("limit")
    if scope == "product" and limit is None:
        limit = 1
    products = dataset.products(prefix=prefix, limit=limit)
    if not products:
        raise ValueError(
            f"dataset {dataset_config.id!r} enumerated no products"
            + (f" under prefix {prefix!r}" if prefix else "")
        )

    per_product: list[BenchmarkRun] = []
    pooled_sizes: list[int] = []
    for product in products:
        run, sizes = _run_product(
            adapter,
            product,
            config,
            output_uri,
            titiler_endpoint=titiler_endpoint,
            requested=requested,
            samples=samples,
        )
        per_product.append(run)
        pooled_sizes += sizes

    policy = tier_policy_from_config(config.tiers)
    rollup_profile = profile_object_sizes(pooled_sizes, policy)
    rollup = BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions={"cng_benchmark": __version__},
        dataset_id=config.dataset,
        format_id=chosen,
        params={
            **config.params,
            "grouping_lever": adapter.describe_grouping_lever(),
            "scope": "rollup",
            "product_count": len(per_product),
            "product_ids": [p.id for p in products],
        },
        object_profile=rollup_profile,
        object_layouts=[ly for run in per_product for ly in run.object_layouts],
        metrics=[
            MetricResult(name="object_count", value=rollup_profile.count),
            MetricResult(
                name="total_bytes", value=rollup_profile.total_bytes, unit="bytes"
            ),
            MetricResult(name="product_count", value=len(per_product)),
        ],
    )
    return ProductSetResult(per_product=per_product, rollup=rollup)
