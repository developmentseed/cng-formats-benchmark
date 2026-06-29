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
from cng_benchmark.datasets.base import Dataset
from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.gdal_env import gdal_session
from cng_benchmark.metrics.display import measure_display
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.metrics.read import (
    measure_copc_read,
    measure_read,
    measure_vector_read,
    measure_zarr_read,
)
from cng_benchmark.metrics.write import measure_write
from cng_benchmark.models import Artifact, BenchmarkRun, MetricResult, ObjectLayout
from cng_benchmark.registry import FORMATS


def _safe_object_layouts(
    adapter: FormatAdapter, name: str, path: str
) -> list[ObjectLayout]:
    """Describe the produced object(s)' layout, or ``[]`` if unavailable.

    Delegates to the adapter's per-format describer (a ``CogLayout`` per COG, a
    ``GeoZarrLayout`` per array). Best-effort structural extra: a missing geo stack
    or an unreadable output yields ``[]`` rather than failing the run, the same way
    the display layout image is best-effort.
    """
    try:
        return list(adapter.describe_layout(path, name=name))
    except Exception:  # noqa: BLE001 - structural extra; never fail the run for it
        return []


def _publish_object(adapter: FormatAdapter, local_target: str, object_uri: str) -> None:
    """Upload the produced object to ``object_uri`` — a file, or a store tree."""
    if adapter.object_kind is ObjectKind.ZARR_STORE:
        storage.upload_tree(local_target, object_uri, role="sink")
    else:
        storage.upload_from_path(local_target, object_uri, role="sink")


def _remove_target(local_target: str) -> None:
    """Free a produced object's local copy — a single file or a store directory."""
    import shutil

    if os.path.isdir(local_target):
        shutil.rmtree(local_target, ignore_errors=True)
    else:
        os.remove(local_target)


def _measure_object_read(adapter: FormatAdapter, object_uri: str) -> list[MetricResult]:
    """Read part of the produced object back, per its object kind.

    A zarr store is read zarr-natively over fsspec (GDAL cannot read the
    ``sharding_indexed`` codec); a GeoParquet file is read with a bbox/row-group
    spatial query over fsspec; a COPC file is read with an octree-node spatial
    query over fsspec; a raster file is read window-by-window with rasterio under
    the sink role's ``/vsis3`` session.
    """
    if adapter.object_kind is ObjectKind.ZARR_STORE:
        return measure_zarr_read(object_uri, role="sink")
    if adapter.object_kind is ObjectKind.VECTOR_FILE:
        return measure_vector_read(object_uri, role="sink")
    if adapter.object_kind is ObjectKind.POINT_CLOUD_FILE:
        return measure_copc_read(object_uri, role="sink")
    with gdal_session("sink"):
        return measure_read(object_uri)


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
        local_target = os.path.join(workdir, adapter.target_basename())
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
        # first-class run artifact, and read/display address it on the store. A
        # store format publishes a tree; a raster, a single file.
        artifact_dir = storage.join(output_uri, chosen)
        object_uri = storage.join(artifact_dir, adapter.target_basename())
        _publish_object(adapter, local_target, object_uri)

        policy = tier_policy_from_config(config.tiers)
        profile = profile_object_sizes(adapter.enumerate_objects(local_target), policy)
        object_layouts = _safe_object_layouts(adapter, chosen, local_target)

        metrics: list[MetricResult] = []
        artifacts: list[Artifact] = []
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
            metrics += _measure_object_read(adapter, object_uri)
        if "display" in requested:
            display_metrics, display_artifacts = _measure_display_object(
                config,
                adapter,
                local_target,
                object_uri,
                artifact_dir,
                titiler_endpoint,
            )
            metrics += display_metrics
            artifacts += display_artifacts

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
        artifacts=artifacts,
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
    extra_artifacts: list[Artifact] = []

    # For zip-delivered products all components live in one .zip object.  Size
    # that container once and charge it to component 0 only — summing the zip
    # size N times in _aggregate_write_metrics would give N×zip_size.
    _first_uri = product.components[0].uri if product.components else ""
    _zip_uri = storage.zip_source_uri(_first_uri)
    _zip_source_size = (
        storage.object_size(_zip_uri, "source") if _zip_uri is not None else None
    )

    with tempfile.TemporaryDirectory() as workdir:
        for i, component in enumerate(product.components):
            local_target = os.path.join(
                workdir, f"{component.name}-{adapter.target_basename()}"
            )
            source_path = storage.to_gdal_path(component.uri)
            if _zip_uri is not None:
                source_size = _zip_source_size if i == 0 else None
            else:
                source_size = storage.object_size(component.uri, "source")
            with gdal_session("source"):
                write_per_component.append(
                    measure_write(
                        adapter,
                        source_path,
                        local_target,
                        config.params,
                        source_size=source_size,
                    )
                )

            component_dir = storage.join(
                output_uri, f"objects/{product.id}/{component.name}"
            )
            object_uri = storage.join(component_dir, adapter.target_basename())
            _publish_object(adapter, local_target, object_uri)
            sizes += adapter.enumerate_objects(local_target)
            # Capture the produced object's layout (structural, per object).
            layouts += _safe_object_layouts(adapter, component.name, local_target)

            # A point cloud has no display tiles; its structural artifact is the
            # octree level-of-detail figure (the COPC analogue of the COG chunk
            # layout). Render it once per product, best-effort.
            if adapter.object_kind is ObjectKind.POINT_CLOUD_FILE and i == 0:
                extra_artifacts += _publish_copc_lod(local_target, component_dir)

            if "read" in requested and i < read_samples:
                extra_metrics += _measure_object_read(adapter, object_uri)
            if "display" in requested and i < display_samples:
                display_metrics, display_artifacts = _measure_display_object(
                    config,
                    adapter,
                    local_target,
                    object_uri,
                    component_dir,
                    titiler_endpoint,
                )
                extra_metrics += display_metrics
                extra_artifacts += display_artifacts

            _remove_target(local_target)

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
        artifacts=extra_artifacts,
    )
    return run, sizes


def _publish_copc_lod(local_target: str, artifact_dir: str) -> list[Artifact]:
    """Render + publish the COPC octree level-of-detail PNG next to the object.

    The point-cloud structural artifact, mirroring how the display path publishes
    ``display_chunk_layout.png`` for a raster. Best-effort: a missing matplotlib
    (the ``cog`` extra) is reported as a skipped artifact, not a failure.
    """
    from cng_benchmark.formats.copc import render_copc_lod

    try:
        local_lod = os.path.join(os.path.dirname(local_target) or ".", "_lod.png")
        render_copc_lod(local_target, local_lod)
        lod_uri = storage.join(artifact_dir, "copc_octree_lod.png")
        storage.upload_from_path(local_lod, lod_uri, role="sink")
        return [
            Artifact(
                kind="octree_lod",
                name="copc_octree_lod",
                uri=lod_uri,
                media_type="image/png",
            )
        ]
    except RuntimeError as exc:
        return [
            Artifact(
                kind="octree_lod",
                name="copc_octree_lod",
                detail={"skipped_reason": str(exc)},
            )
        ]


def _publish_rgb_vrts(
    dataset: Dataset,
    products: list[Product],
    adapter: FormatAdapter,
    output_uri: str,
) -> list[Artifact]:
    """Stack the produced per-band COGs into run-level RGB composite VRT(s).

    A viewer convenience: for each composite the dataset exposes
    (:meth:`~cng_benchmark.datasets.base.Dataset.rgb_composites`), every RGB band
    is mosaicked across the products that carry it, and the result is written as
    ``run-<name>.vrt`` at the run root so its ``s3://`` path opens directly in
    TiTiler's viewer. Only single-file rasters published to S3 qualify; a
    composite whose bands are absent or unreadable is recorded as a skipped
    artifact rather than failing the run. The per-band COG URIs are reconstructed
    from the deterministic layout :func:`_run_product` uploads them to.
    """
    if adapter.object_kind is not ObjectKind.RASTER_FILE:
        return []
    if not storage.is_s3(output_uri):
        return []
    composites = dataset.rgb_composites()
    if not composites:
        return []

    from cng_benchmark import vrt

    basename = adapter.target_basename()
    artifacts: list[Artifact] = []
    with gdal_session("sink"):
        for composite in composites:
            try:
                band_grids: list[list[vrt.GridMeta]] = []
                for band in composite.bands:
                    grids: list[vrt.GridMeta] = []
                    for product in products:
                        if band not in {c.name for c in product.components}:
                            continue
                        comp_dir = storage.join(
                            output_uri, f"objects/{product.id}/{band}"
                        )
                        object_uri = storage.join(comp_dir, basename)
                        grids.append(vrt.read_grid(storage.to_gdal_path(object_uri)))
                    band_grids.append(grids)
                if any(not grids for grids in band_grids):
                    raise ValueError("no source COGs for one or more RGB bands")

                xml = vrt.build_rgb_vrt_xml(band_grids)
                vrt_uri = storage.join(output_uri, f"run-{composite.name}.vrt")
                storage.write_text(vrt_uri, xml, role="sink")

                detail: dict = {}
                if composite.rescale is not None:
                    lo, hi = composite.rescale
                    detail["rescale"] = [lo, hi]
                    detail["titiler_url"] = (
                        f"/cog/viewer?url={vrt_uri}&rescale={lo:g},{hi:g}"
                    )
                artifacts.append(
                    Artifact(
                        kind="viewer_vrt",
                        name=composite.name,
                        uri=vrt_uri,
                        media_type="application/xml",
                        detail=detail,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - best-effort viewer extra
                artifacts.append(
                    Artifact(
                        kind="viewer_vrt",
                        name=composite.name,
                        detail={"skipped_reason": str(exc)},
                    )
                )
    return artifacts


def _measure_display_object(
    config: BenchmarkConfig,
    adapter: FormatAdapter,
    local_target: str,
    object_uri: str,
    artifact_dir: str,
    titiler_endpoint: str | None,
) -> tuple[list[MetricResult], list[Artifact]]:
    """Run the display metric for one produced object and publish its chunk layout.

    Selects chunk-crossing tiles against the *local* produced object (cheap, no
    network), times them via TiTiler against the uploaded object, and (best-effort)
    renders the block/chunk-grid + tile-footprint ``display_chunk_layout.png`` next
    to it. Branches on the object kind: a raster file uses TiTiler's ``/cog``
    endpoints + the rasterio grid; a zarr store uses the multidim/xarray router
    (``display_titiler_path``, default ``""`` — the in-stack ``titiler-xarray``
    serves at the root) + the zarr chunk grid + the ``variable`` query.
    """
    if not titiler_endpoint:
        raise ValueError("the display metric requires a TiTiler endpoint")
    if not storage.is_s3(object_uri):
        raise ValueError("the display metric requires an S3 output location")
    from cng_benchmark.metrics.display_tiles import (
        DEFAULT_TARGETS,
        render_chunk_layout,
        render_zarr_chunk_layout,
        select_chunk_tiles,
        select_zarr_chunk_tiles,
    )

    targets = tuple(config.params.get("display_chunk_targets", DEFAULT_TARGETS))
    if adapter.object_kind is ObjectKind.ZARR_STORE:
        from cng_benchmark.formats.geozarr import DATA_VAR

        tiles = select_zarr_chunk_tiles(local_target, targets=targets)
        prefix = str(config.params.get("display_titiler_path", ""))
        metrics = measure_display(
            titiler_endpoint,
            object_uri,
            tiles,
            path_prefix=prefix,
            extra_query={"variable": DATA_VAR},
        )
        render = render_zarr_chunk_layout
    else:
        tiles = select_chunk_tiles(local_target, targets=targets)
        metrics = measure_display(titiler_endpoint, object_uri, tiles)
        render = render_chunk_layout

    try:
        local_layout = os.path.join(os.path.dirname(local_target) or ".", "_layout.png")
        render(local_target, tiles, local_layout)
        layout_uri = storage.join(artifact_dir, "display_chunk_layout.png")
        storage.upload_from_path(local_layout, layout_uri, role="sink")
        artifact = Artifact(
            kind="chunk_layout",
            name="display_chunk_layout",
            uri=layout_uri,
            media_type="image/png",
        )
    except RuntimeError as exc:
        artifact = Artifact(
            kind="chunk_layout",
            name="display_chunk_layout",
            detail={"skipped_reason": str(exc)},
        )
    return metrics, [artifact]


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

    vrt_artifacts = _publish_rgb_vrts(dataset, products, adapter, output_uri)

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
        artifacts=vrt_artifacts,
    )
    return ProductSetResult(per_product=per_product, rollup=rollup)
