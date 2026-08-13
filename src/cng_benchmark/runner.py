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

import functools
import logging
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from statistics import mean, median, pstdev

import cng_benchmark.datasets  # noqa: F401  (registers the built-in readers)
import cng_benchmark.formats  # noqa: F401  (registers the built-in adapters)
from cng_benchmark import __version__, storage
from cng_benchmark.config import BenchmarkConfig, DatasetConfig, tier_policy_from_config
from cng_benchmark.datasets import Product, build_dataset
from cng_benchmark.datasets.base import Dataset, SourceObject
from cng_benchmark.formats.base import FormatAdapter, ObjectKind
from cng_benchmark.gdal_env import gdal_session
from cng_benchmark.metrics.display import fetch_titiler_versions, measure_display
from cng_benchmark.metrics.objects import profile_object_sizes
from cng_benchmark.metrics.read import (
    measure_copc_read,
    measure_read,
    measure_vector_read,
    measure_zarr_read,
)
from cng_benchmark.metrics.write import measure_write, measure_write_batch
from cng_benchmark.models import (
    Artifact,
    BenchmarkRun,
    ConditionReplicate,
    ConditionResult,
    MetricResult,
    ObjectLayout,
)
from cng_benchmark.registry import FORMATS

logger = logging.getLogger(__name__)


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


def _log_write_done(
    metrics: list[MetricResult],
    prefix: str = "",
) -> None:
    """Emit an INFO line with elapsed time and throughput from write metrics."""
    elapsed = next((m.value for m in metrics if m.name == "write_elapsed"), None)
    if elapsed is None:
        return
    tput = next((m.value for m in metrics if m.name == "write_throughput"), None)
    tput_str = f", {tput / 1e6:.1f} MB/s" if tput else ""
    logger.info("%swrite done (%.1fs%s)", prefix, elapsed, tput_str)


def _tool_versions(titiler_endpoint: str | None) -> dict[str, str]:
    """``cng_benchmark``'s own version, plus the tiler's if reachable.

    Best-effort: a display metric already records which router (``path_prefix``)
    served each tile, but not which build of that router — this makes the run's
    ``tool_versions`` the place a report resolves "which reader" to "which code".
    """
    versions = {"cng_benchmark": __version__}
    if titiler_endpoint:
        versions.update(fetch_titiler_versions(titiler_endpoint))
    return versions


def _measure_object_read(
    adapter: FormatAdapter,
    object_uri: str,
    *,
    role: str = "sink",
    seed: int = 0,
    locator: str | None = None,
    name: str | None = None,
) -> list[MetricResult]:
    """Read part of the produced object back, per its object kind.

    A zarr store is read zarr-natively over fsspec (GDAL cannot read the
    ``sharding_indexed`` codec); a vector file is read with a bbox spatial query —
    pushed down to the row groups of a GeoParquet over fsspec, or to the packed
    R-tree of a FlatGeobuf through OGR; a COPC file is read with an octree-node
    spatial query over fsspec; a raster file is read window-by-window with rasterio
    under the ``role``'s ``/vsis3`` session.

    ``role`` and ``seed`` are threaded through so a run-protocol replicate (a
    fresh subprocess, per issue #87) can address the object under the right
    credentials and sample a different window/query set than its siblings.

    ``locator``/``name`` address one component of a batched adapter's bundled
    object (#102) — ``locator`` is that adapter's own ``component_locator``
    (a GeoZarr grid id, or a COG 1-based band index as a string) and ``name``
    the component's own name (a bundled GeoZarr array is always named for its
    component). Both ``None`` (the defaults) address a non-batched object,
    unchanged.
    """
    if adapter.object_kind is ObjectKind.ZARR_STORE:
        return measure_zarr_read(
            object_uri, role=role, seed=seed, group=locator, var_name=name
        )
    if adapter.object_kind is ObjectKind.VECTOR_FILE:
        return measure_vector_read(object_uri, role=role, seed=seed)
    if adapter.object_kind is ObjectKind.POINT_CLOUD_FILE:
        return measure_copc_read(object_uri, role=role, seed=seed)
    with gdal_session(role):
        band = int(locator) if locator is not None else 1
        return measure_read(object_uri, seed=seed, band=band)


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
        tool_versions=_tool_versions(None),
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

    logger.info("convert %s → %s", source_uri, chosen)
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
        _log_write_done(write_metrics)

        # Always publish the produced object under the output location: it is a
        # first-class run artifact, and read/display address it on the store. A
        # store format publishes a tree; a raster, a single file.
        artifact_dir = storage.join(output_uri, chosen)
        object_uri = storage.join(artifact_dir, adapter.target_basename())
        _publish_object(adapter, local_target, object_uri)
        logger.info("uploaded to %s", object_uri)

        policy = tier_policy_from_config(config.tiers)
        profile = profile_object_sizes(adapter.enumerate_objects(local_target), policy)
        object_layouts = _safe_object_layouts(adapter, chosen, local_target)

        metrics: list[MetricResult] = []
        artifacts: list[Artifact] = []
        conditions: list[ConditionResult] = []
        run_protocol = "run_protocol" in config.params
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
            logger.info("read metric")
            metrics += _safe_read_metrics(adapter, object_uri)
            if run_protocol:
                logger.info("read metric: run protocol")
                conditions += _run_read_conditions(config, chosen, object_uri)
        if "display" in requested:
            logger.info("display metric")
            display_metrics, display_artifacts = _safe_display_metrics(
                config,
                adapter,
                local_target,
                object_uri,
                artifact_dir,
                titiler_endpoint,
            )
            metrics += display_metrics
            artifacts += display_artifacts
            if run_protocol:
                logger.info("display metric: run protocol")
                conditions += _run_display_conditions(
                    config,
                    adapter,
                    local_target,
                    object_uri,
                    artifact_dir,
                    titiler_endpoint,
                )

    params = {**config.params, "grouping_lever": adapter.describe_grouping_lever()}
    return BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions=_tool_versions(titiler_endpoint),
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        object_layouts=object_layouts,
        metrics=metrics,
        artifacts=artifacts,
        conditions=conditions,
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
    *,
    component_count: int | None = None,
) -> list[MetricResult]:
    """Pool per-component write metrics into one product-level write result.

    Elapsed times sum (the product's total conversion wall time) and throughput
    is recomputed from the pooled output bytes over that total, so a product's
    write metric is comparable to a single object's. ``component_count``
    defaults to ``len(per_component)`` (one write call per component, true
    before #102) — a bundled write call covers several components in one
    call, so a batched caller passes the real component count explicitly
    rather than undercounting it as "one write call = one component".
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
    detail: dict = {
        "bytes_out": bytes_out,
        "components": component_count
        if component_count is not None
        else len(per_component),
    }
    if have_bytes_in:
        detail["bytes_in"] = bytes_in
    return [
        MetricResult(name="write_elapsed", value=total_elapsed, unit="s"),
        MetricResult(
            name="write_throughput", value=throughput, unit="bytes/s", detail=detail
        ),
    ]


def _parse_bundle_groups(
    params: dict, components: list[SourceObject]
) -> tuple[list[list[SourceObject]], list[SourceObject]]:
    """Partition a product's components by ``params.bundle_components`` (#102).

    Bundling is opt-in and off by default — it must never auto-trigger just
    because a format could support it, or every component of an arm like S2
    MAJA (bands *and* masks) would get silently merged into one object with
    no way to opt out:

    * unset / falsy (the default) — every component stays a "single"; the
      existing per-component path, byte-for-byte unchanged.
    * ``true`` — every component forms one implicit group (the format
      adapter's own grid-equality detection splits it further if they don't
      all share one grid).
    * a list of lists of component names — only those named components
      bundle, grouped exactly as listed; every component named nowhere stays
      a single. A named group of one component is just that component,
      unbundled (bundling one thing achieves nothing).

    Raises ``ValueError`` for a name the product doesn't actually have — a
    config/product mismatch should fail loudly, not silently drop the group.
    """
    spec = params.get("bundle_components")
    if not spec:
        return [], list(components)

    by_name = {c.name: c for c in components}
    if spec is True:
        if len(components) <= 1:
            return [], list(components)
        return [list(components)], []

    groups: list[list[SourceObject]] = []
    bundled_names: set[str] = set()
    for raw_group in spec:
        missing = [n for n in raw_group if n not in by_name]
        if missing:
            raise ValueError(
                f"params.bundle_components names {missing!r}, which "
                f"{'is' if len(missing) == 1 else 'are'} not among this "
                f"product's components ({sorted(by_name)})"
            )
        members = [by_name[n] for n in raw_group]
        if len(members) > 1:
            groups.append(members)
            bundled_names.update(raw_group)
    singles = [c for c in components if c.name not in bundled_names]
    return groups, singles


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
    ``display`` run only on the first ``samples[...]`` components overall (a
    representative sample, default 1 — counted across singles *and* bundled
    components together, so the sample is "the first N of the product," not
    "the first N of each path"). Each produced object is uploaded and its
    local copy freed before the next one is converted, so local disk is
    bounded by one component (or one bundle group) at a time regardless of
    product size. Returns the run and the per-object sizes (for the roll-up
    to pool).

    ``params.bundle_components`` (#102) partitions the product's components
    into bundle groups (converted together, one call, via
    ``adapter.convert_batch``) and singles (the path above, unchanged) — see
    :func:`_parse_bundle_groups`. Unset, every component is a single: this
    function behaves exactly as it did before bundling existed.
    """
    chosen = adapter.name
    read_samples = int(samples.get("read", 1))
    display_samples = int(samples.get("display", 1))

    sizes: list[int] = []
    layouts: list[ObjectLayout] = []
    write_calls: list[list[MetricResult]] = []
    extra_metrics: list[MetricResult] = []
    extra_artifacts: list[Artifact] = []
    conditions: list[ConditionResult] = []
    run_protocol = "run_protocol" in config.params

    bundle_groups, singles = _parse_bundle_groups(config.params, product.components)
    if bundle_groups and not getattr(adapter, "supports_batch", False):
        raise ValueError(
            f"benchmark {config.id!r} sets params.bundle_components, but "
            f"format {chosen!r} does not support batched conversion"
        )

    # A zip-delivered product's components all live in one .zip object; a
    # multi-variable netCDF granule's components (SWOT Raster100m's
    # ``variables: all``) all live in one .nc object. Either way, size that
    # container once and charge it to whichever write call processes first —
    # summing the container size N times in _aggregate_write_metrics would
    # give N×container_size.
    _first_uri = product.components[0].uri if product.components else ""
    _container_uri = storage.zip_source_uri(_first_uri) or storage.netcdf_source_uri(
        _first_uri
    )
    _container_source_size = (
        storage.object_size(_container_uri, "source")
        if _container_uri is not None
        else None
    )
    container_charged = False
    n_comp = len(product.components)
    sample_index = 0  # "first N components of the product", singles + bundled alike

    with tempfile.TemporaryDirectory() as workdir:
        for i, component in enumerate(singles):
            logger.info(
                "  [%d/%d] %s: convert → %s", i + 1, n_comp, component.name, chosen
            )
            local_target = os.path.join(
                workdir, f"{component.name}-{adapter.target_basename()}"
            )
            source_path = storage.to_gdal_path(component.uri)
            if _container_uri is not None:
                source_size = _container_source_size if not container_charged else None
                container_charged = True
            else:
                source_size = storage.object_size(component.uri, "source")
            # Merge the reader's per-component pixel-interpretation metadata into
            # params so the adapters can write it onto the produced object. The
            # source rasters don't always carry it (MAJA keeps nodata and the
            # reflectance quantification in side-metadata), and what the writer
            # is not told, the output cannot declare (#70). Always a fresh copy,
            # so a component's values can never leak into the next one or into
            # the run's own params; `setdefault` keeps an explicit config param
            # authoritative.
            convert_params = dict(config.params)
            for key, value in (
                ("nodata", component.nodata),
                ("scale_factor", component.scale_factor),
                ("standard_name", component.standard_name),
            ):
                if value is not None:
                    convert_params.setdefault(key, value)
            with gdal_session("source"):
                wm = measure_write(
                    adapter,
                    source_path,
                    local_target,
                    convert_params,
                    source_size=source_size,
                )
            write_calls.append(wm)
            _log_write_done(wm, prefix=f"  [{i + 1}/{n_comp}] {component.name}: ")

            component_dir = storage.join(
                output_uri, f"objects/{product.id}/{component.name}"
            )
            object_uri = storage.join(component_dir, adapter.target_basename())
            _publish_object(adapter, local_target, object_uri)
            logger.info("  [%d/%d] %s: uploaded", i + 1, n_comp, component.name)
            sizes += adapter.enumerate_objects(local_target)
            # Capture the produced object's layout (structural, per object).
            layouts += _safe_object_layouts(adapter, component.name, local_target)

            # A point cloud has no display tiles; its structural artifact is the
            # octree level-of-detail figure (the COPC analogue of the COG chunk
            # layout). Render it once per product, best-effort. Point clouds
            # never support batching, so `singles` is always the whole product
            # for this kind and `i == 0` is still the product's first component.
            if adapter.object_kind is ObjectKind.POINT_CLOUD_FILE and i == 0:
                extra_artifacts += _publish_copc_lod(local_target, component_dir)

            if "read" in requested and sample_index < read_samples:
                logger.info("  [%d/%d] %s: read metric", i + 1, n_comp, component.name)
                extra_metrics += _safe_read_metrics(adapter, object_uri)
                if run_protocol:
                    logger.info(
                        "  [%d/%d] %s: read metric run protocol",
                        i + 1,
                        n_comp,
                        component.name,
                    )
                    conditions += _run_read_conditions(config, chosen, object_uri)
            if "display" in requested and sample_index < display_samples:
                logger.info(
                    "  [%d/%d] %s: display metric", i + 1, n_comp, component.name
                )
                display_metrics, display_artifacts = _safe_display_metrics(
                    config,
                    adapter,
                    local_target,
                    object_uri,
                    component_dir,
                    titiler_endpoint,
                )
                extra_metrics += display_metrics
                extra_artifacts += display_artifacts
                if run_protocol:
                    logger.info(
                        "  [%d/%d] %s: display metric run protocol",
                        i + 1,
                        n_comp,
                        component.name,
                    )
                    conditions += _run_display_conditions(
                        config,
                        adapter,
                        local_target,
                        object_uri,
                        component_dir,
                        titiler_endpoint,
                    )
            sample_index += 1

            _remove_target(local_target)

        for g, group in enumerate(bundle_groups):
            # A label is only needed to disambiguate multiple bundles (or a mix
            # of bundles and singles) sharing one product; the common case —
            # `bundle_components: true`, the whole product one group — writes
            # straight to the product's own object path.
            label = f"bundle-{g}" if (len(bundle_groups) > 1 or singles) else None
            names = [c.name for c in group]
            logger.info(
                "  [bundle %d/%d] %s: convert %d components → %s",
                g + 1,
                len(bundle_groups),
                label or product.id,
                len(group),
                chosen,
            )
            local_target = os.path.join(
                workdir, f"{label or 'bundle'}-{adapter.target_basename()}"
            )
            if _container_uri is not None:
                source_size = _container_source_size if not container_charged else None
                container_charged = True
            else:
                source_size = sum(storage.object_size(c.uri, "source") for c in group)
            sources = [replace(c, uri=storage.to_gdal_path(c.uri)) for c in group]
            with gdal_session("source"):
                wm = measure_write_batch(
                    adapter,
                    sources,
                    local_target,
                    dict(config.params),
                    source_size=source_size,
                )
            write_calls.append(wm)
            _log_write_done(
                wm, prefix=f"  [bundle {g + 1}/{len(bundle_groups)}] {names}: "
            )

            bundle_dir = storage.join(
                output_uri,
                f"objects/{product.id}/{label}" if label else f"objects/{product.id}",
            )
            object_uri = storage.join(bundle_dir, adapter.target_basename())
            _publish_object(adapter, local_target, object_uri)
            logger.info(
                "  [bundle %d/%d] uploaded (%s)", g + 1, len(bundle_groups), names
            )
            sizes += adapter.enumerate_objects(local_target)
            # One call, per bundle: the adapter itself returns 1..N layouts
            # (one per component for GeoZarr's sibling arrays; one for the
            # whole file for a COG multi-band write — see FormatAdapter.
            # describe_layout).
            layouts += _safe_object_layouts(adapter, product.id, local_target)

            for component in group:
                locator = adapter.component_locator(local_target, component.name)
                if "read" in requested and sample_index < read_samples:
                    logger.info(
                        "  [bundle %d/%d] %s: read metric",
                        g + 1,
                        len(bundle_groups),
                        component.name,
                    )
                    extra_metrics += _safe_read_metrics(
                        adapter, object_uri, locator=locator, name=component.name
                    )
                    if run_protocol:
                        conditions += _run_read_conditions(
                            config,
                            chosen,
                            object_uri,
                            locator=locator,
                            name=component.name,
                        )
                if "display" in requested and sample_index < display_samples:
                    logger.info(
                        "  [bundle %d/%d] %s: display metric",
                        g + 1,
                        len(bundle_groups),
                        component.name,
                    )
                    display_metrics, display_artifacts = _safe_display_metrics(
                        config,
                        adapter,
                        local_target,
                        object_uri,
                        bundle_dir,
                        titiler_endpoint,
                        locator=locator,
                        name=component.name,
                    )
                    extra_metrics += display_metrics
                    extra_artifacts += display_artifacts
                    if run_protocol:
                        conditions += _run_display_conditions(
                            config,
                            adapter,
                            local_target,
                            object_uri,
                            bundle_dir,
                            titiler_endpoint,
                            locator=locator,
                            name=component.name,
                        )
                sample_index += 1

            _remove_target(local_target)

    policy = tier_policy_from_config(config.tiers)
    profile = profile_object_sizes(sizes, policy)

    metrics: list[MetricResult] = []
    if "write" in requested:
        metrics += _aggregate_write_metrics(write_calls, component_count=n_comp)
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
        tool_versions=_tool_versions(titiler_endpoint),
        dataset_id=config.dataset,
        format_id=chosen,
        params=params,
        object_profile=profile,
        object_layouts=layouts,
        metrics=metrics,
        artifacts=extra_artifacts,
        conditions=conditions,
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
    """Stack the produced per-band COGs into run-level viewer VRT(s).

    Two paths:
    * **RGB**: for each composite the dataset exposes via
      :meth:`~cng_benchmark.datasets.base.Dataset.rgb_composites`, every band is
      mosaicked across the products that carry it into a 3-band ``run-<name>.vrt``.
    * **Single-band**: for each band the dataset exposes via
      :meth:`~cng_benchmark.datasets.base.Dataset.viewer_bands`, a 1-band Gray
      mosaic is built — one VRT per CRS group when the products span multiple
      UTM zones (e.g. a France-wide SWOT run).  VRT names are
      ``run-<band>.vrt`` for a single-zone run and
      ``run-<band>-epsg<N>.vrt`` for multi-zone runs.

    Only single-file rasters published to S3 qualify; a composite whose bands are
    absent or unreadable is recorded as a skipped artifact rather than failing the
    run. The per-band COG URIs are reconstructed from the deterministic layout
    :func:`_run_product` uploads them to.
    """
    if adapter.object_kind is not ObjectKind.RASTER_FILE:
        return []
    if not storage.is_s3(output_uri):
        return []
    composites = dataset.rgb_composites()
    single_bands = dataset.viewer_bands()
    if not composites and not single_bands:
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

        for sb in single_bands:
            try:
                grids: list[vrt.GridMeta] = []
                for product in products:
                    if sb.band not in {c.name for c in product.components}:
                        continue
                    comp_dir = storage.join(
                        output_uri, f"objects/{product.id}/{sb.band}"
                    )
                    object_uri = storage.join(comp_dir, basename)
                    grids.append(vrt.read_grid(storage.to_gdal_path(object_uri)))
                if not grids:
                    raise ValueError(f"no source COGs for band {sb.band!r}")

                # Group by CRS — products spanning multiple UTM zones need one VRT
                # per zone (GDAL VRT requires a single SRS per dataset).
                by_crs: dict[str, list[vrt.GridMeta]] = {}
                for g in grids:
                    by_crs.setdefault(g.crs_wkt, []).append(g)
                multi_zone = len(by_crs) > 1

                for crs_wkt, zone_grids in by_crs.items():
                    xml = vrt.build_single_band_vrt_xml(zone_grids)
                    if multi_zone:
                        epsg = vrt.crs_epsg(crs_wkt)
                        zone_idx = list(by_crs).index(crs_wkt)
                        suffix = f"-epsg{epsg}" if epsg else f"-zone{zone_idx}"
                        vrt_name = f"{sb.name}{suffix}"
                    else:
                        vrt_name = sb.name
                    vrt_uri = storage.join(output_uri, f"run-{vrt_name}.vrt")
                    storage.write_text(vrt_uri, xml, role="sink")

                    detail = {}
                    if sb.rescale is not None:
                        lo, hi = sb.rescale
                        detail["rescale"] = [lo, hi]
                        detail["titiler_url"] = (
                            f"/cog/viewer?url={vrt_uri}&rescale={lo:g},{hi:g}"
                        )
                    artifacts.append(
                        Artifact(
                            kind="viewer_vrt",
                            name=vrt_name,
                            uri=vrt_uri,
                            media_type="application/xml",
                            detail=detail,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - best-effort viewer extra
                artifacts.append(
                    Artifact(
                        kind="viewer_vrt",
                        name=sb.name,
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
    *,
    locator: str | None = None,
    name: str | None = None,
) -> tuple[list[MetricResult], list[Artifact]]:
    """Run the display metric for one produced object and publish its chunk layout.

    Selects chunk-crossing *and* resolution-coverage tiles against the *local*
    produced object (cheap, no network), times them via TiTiler against the
    uploaded object, and (best-effort) renders the block/chunk-grid +
    tile-footprint ``display_chunk_layout.png`` next to it. Branches on the
    object kind: a raster file uses TiTiler's ``/cog`` endpoints + the
    rasterio grid; a zarr store is served by the router named by
    ``display_titiler_path`` (default ``"zarr"``, the bench tiler's stock-xarray
    router; ``"geozarr"`` selects its ``GeoZarrReader``-backed one instead) + the
    zarr chunk grid. The two routers address the array differently -- the stock
    router has no multiscale awareness of its own and is queried with a fixed
    ``variable=``/``group=`` pair naming the store's own native/canonical
    group, the same *every* tile gets regardless of which resolution is being
    timed (the same fixed location a real client's STAC asset href would
    already carry, never a per-zoom-level path it resolves itself); whatever
    it reads gets resampled on the fly by TiTiler, so it pays the honest cost
    of not knowing the store's pyramid, however far a target resolution is
    from native (#121 removed the #116/#117 per-tile override that swapped in
    a *different*, resolution-appropriate group per tile, crediting this
    router with multiscale awareness it doesn't have) -- while
    ``GeoZarrReader`` resolves the pyramid itself from the requested zoom and
    addresses the array as ``{group}:{name}``.

    ``display_chunk_targets``/``display_target_resolutions``/
    ``display_tile_samples`` (``config.params``, #116) control the scenario
    set: chunk-crossing buckets (unchanged, #102's original question — the
    cost of straddling chunks within one level) plus one tile per target
    ground resolution (new — the cost of reading a precomputed overview vs.
    downsampling from native, at a zoom a panning/zooming client would
    actually generate). ``display_target_resolutions`` should be set to the
    *same* list on a COG arm and its matched GeoZarr arm so the resulting
    ``res_*`` labels are directly comparable across formats — see
    :func:`~cng_benchmark.metrics.display_tiles.select_resolution_tiles`.
    Unset, resolutions are derived from the object's own decimations
    (today's implicit per-format behaviour, unaffected).

    ``locator``/``name`` address one component of a batched adapter's bundled
    object (#102) the same way :func:`_measure_object_read` does: a COG's 1-based
    band index goes out as ``bidx``, a GeoZarr grid group as ``{group}:{name}``
    (GeoZarrReader) or an explicit ``group=``/``variable=`` pair (the stock
    router). Run live against the docker-compose bench titiler for all three
    routers and confirmed each component resolves to its own exact pixel values, not a
    neighbour's — that pass also caught and fixed a real bug where a bundled
    *flat* (``multiscale_levels: 0``) GeoZarr store's grid attrs got wiped by
    the second component's write (see
    :func:`~cng_benchmark.formats.geozarr._write_sharded`).
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
        select_resolution_tiles,
        select_zarr_chunk_tiles,
        select_zarr_resolution_tiles,
    )

    targets = tuple(config.params.get("display_chunk_targets", DEFAULT_TARGETS))
    target_resolutions = config.params.get("display_target_resolutions")
    samples = int(config.params.get("display_tile_samples", 8))
    if adapter.object_kind is ObjectKind.ZARR_STORE:
        from cng_benchmark.formats.geozarr import (
            DATA_VAR,
            finest_level_group,
            is_unified_pyramid,
        )

        var_name = name or DATA_VAR
        tiles = select_zarr_chunk_tiles(
            local_target, targets=targets, group=locator, var_name=name
        ) + select_zarr_resolution_tiles(
            local_target,
            target_resolutions=target_resolutions,
            group=locator,
            var_name=name,
        )
        prefix = str(config.params.get("display_titiler_path", "zarr"))
        group_query_key: str | None = None
        if prefix == "geozarr":
            # GeoZarrReader addresses a variable as "{group}:{name}", where
            # `group` is the group that declares the zarr_conventions
            # multiscales entry -- and resolves the multiscale level from
            # the requested zoom itself, but *only* when addressed via that
            # exact group; any other group is read as-is, with no
            # zoom-awareness at all (#119). That's this writer's store root
            # ("/") for a non-batched store, or one grid's own subtree
            # ("/grid0") for a #102 single-resolution bundle -- both of
            # which `locator` already names. A #114 cross-tier unified
            # pyramid is the one case `locator` gets wrong: it names a
            # component's own native *level* ("/0", "/1", ...), never where
            # that store's one shared multiscales doc actually lives -- the
            # store root, covering every component regardless of which
            # level its own data starts at.
            group_path = f"/{locator}" if locator else "/"
            if locator and is_unified_pyramid(local_target):
                group_path = "/"
            extra_query = {"variables": f"{group_path}:{var_name}"}
        else:
            # The stock xarray router still needs *some* group to locate a
            # nested variable -- titiler.xarray opens exactly the group it's
            # given (no group -> the store's true root), and this writer
            # nests even the native level under its own integer group
            # ("0/data") whenever the store carries any pyramid at all, so
            # omitting `group` outright would 404 every non-flat store. What
            # a real client (its STAC asset href) actually points at is one
            # *fixed* location -- the store's own native/canonical group --
            # never a per-zoom-level path it would have to resolve itself.
            # So the base query below stays fixed at the native level, same
            # as every tile got before #116; what #121 removes is only the
            # #116/#117 per-tile override that swapped in a *different*,
            # resolution-appropriate group for each res_*m scenario, which
            # credited this router with multiscale awareness it doesn't
            # have -- no `group_query_key` is set here anymore, so every
            # tile, whatever resolution it targets, reads this same fixed
            # group and TiTiler resamples on the fly.
            extra_query = {"variable": var_name}
            level = finest_level_group(local_target, group=locator, var_name=name)
            group_path = locator
            if level is not None:
                group_path = f"{locator}/{level}" if locator else level
            if group_path is not None:
                extra_query["group"] = group_path
        metrics = measure_display(
            titiler_endpoint,
            object_uri,
            tiles,
            samples=samples,
            path_prefix=prefix,
            extra_query=extra_query,
            group_query_key=group_query_key,
        )
        render = functools.partial(
            render_zarr_chunk_layout, group=locator, var_name=name
        )
    else:
        tiles = select_chunk_tiles(
            local_target, targets=targets
        ) + select_resolution_tiles(local_target, target_resolutions=target_resolutions)
        extra_query = {"bidx": locator} if locator else None
        metrics = measure_display(
            titiler_endpoint,
            object_uri,
            tiles,
            samples=samples,
            extra_query=extra_query,
        )
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


def _safe_read_metrics(
    adapter: FormatAdapter,
    object_uri: str,
    *,
    locator: str | None = None,
    name: str | None = None,
) -> list[MetricResult]:
    """Run the read metric, returning a skipped marker on any exception.

    Network-dependent: a range-request timeout or transient S3 error should not
    abort the enclosing product or dataset run, so failures are caught, logged,
    and surfaced as a ``read_skipped`` result with the error string in ``detail``.
    ``locator``/``name`` address one component of a bundled object (#102).
    """
    try:
        return _measure_object_read(adapter, object_uri, locator=locator, name=name)
    except Exception as exc:  # noqa: BLE001 - best-effort, same stance as layout image
        logger.warning("read metric skipped: %s", exc)
        detail = {"error": str(exc)}
        return [MetricResult(name="read_skipped", value=0.0, detail=detail)]


def _safe_display_metrics(
    config: BenchmarkConfig,
    adapter: FormatAdapter,
    local_target: str,
    object_uri: str,
    artifact_dir: str,
    titiler_endpoint: str | None,
    *,
    locator: str | None = None,
    name: str | None = None,
) -> tuple[list[MetricResult], list[Artifact]]:
    """Run the display metric, returning a skipped marker on any exception.

    Network-dependent (TiTiler tile fetches with a 30 s timeout): a transient
    timeout on one product must not abort a 20- or 100-product dataset run, so
    failures are caught, logged, and surfaced as a ``display_skipped`` result.
    ``locator``/``name`` address one component of a bundled object (#102).
    """
    try:
        return _measure_display_object(
            config,
            adapter,
            local_target,
            object_uri,
            artifact_dir,
            titiler_endpoint,
            locator=locator,
            name=name,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort, same stance as layout image
        logger.warning("display metric skipped: %s", exc)
        detail = {"error": str(exc)}
        return [MetricResult(name="display_skipped", value=0.0, detail=detail)], []


# --- Run protocol (issue #87): replicates, cold/warm cache, isolated/concurrent ---
#
# Opt-in via config.params["run_protocol"]; purely additive to run.conditions —
# run.metrics keeps coming from the single-shot _safe_read_metrics /
# _safe_display_metrics calls above, unchanged. Applies fully (cache ×
# concurrency × replicates) to the read phase, since its caches (GDAL block
# cache, vsicurl, fsspec) are entirely client-side and harness-controlled. The
# display phase only varies replicates: TiTiler's cache is a server-side
# property of the deployed tiler pod, so a cold or concurrent display
# condition would need bouncing that deployment between passes — the
# cluster-level `deploy/` work the issue lists as a touch point, not something
# this process can do to itself.


@dataclass(frozen=True)
class RunCondition:
    """One point in the run-protocol condition matrix: a cache state + worker count."""

    cache: str  # "cold" | "warm"
    concurrency: int  # worker count; 1 == isolated


def _parse_run_protocol(params: dict) -> tuple[int, list[RunCondition]]:
    """Parse ``params["run_protocol"]`` into ``(replicates, conditions)``.

    Defaults to a single warm, isolated condition (today's behaviour) when a
    key is omitted. Raises ``ValueError`` for a malformed ``cache`` or a
    non-positive ``replicates``/``concurrency`` — a misconfigured protocol
    should fail loudly, not silently measure the wrong thing.
    """
    raw = dict(params.get("run_protocol", {}))
    replicates = int(raw.get("replicates", 1))
    if replicates < 1:
        raise ValueError(f"run_protocol.replicates must be >= 1, got {replicates}")

    raw_conditions = raw.get("conditions") or [{"cache": "warm", "concurrency": 1}]
    conditions: list[RunCondition] = []
    for c in raw_conditions:
        cache = c.get("cache", "warm")
        if cache not in ("cold", "warm"):
            raise ValueError(
                f"run_protocol condition cache must be 'cold' or 'warm', got {cache!r}"
            )
        concurrency = int(c.get("concurrency", 1))
        if concurrency < 1:
            raise ValueError(
                f"run_protocol condition concurrency must be >= 1, got {concurrency}"
            )
        conditions.append(RunCondition(cache=cache, concurrency=concurrency))
    return replicates, conditions


def _read_phase_worker(
    format_id: str,
    object_uri: str,
    role: str,
    seed: int,
    locator: str | None = None,
    component_name: str | None = None,
) -> list[MetricResult]:
    """One worker's full read pass — the ``ProcessPoolExecutor`` task.

    Module-level and picklable so a ``spawn``-context pool can dispatch it into
    a fresh interpreter: the adapter is re-resolved by name (never pickled),
    and the read runs exactly as :func:`_measure_object_read` does outside a
    run protocol.

    ``locator``/``component_name`` address one component of a bundled object
    (#102) and are passed explicitly rather than read from the freshly
    resolved adapter's own bookkeeping — a spawned worker's adapter instance
    never saw the ``convert_batch`` call that produced ``object_uri``, so it
    has no bookkeeping to read.
    """
    adapter = FORMATS.get(format_id)()
    return _measure_object_read(
        adapter, object_uri, role=role, seed=seed, locator=locator, name=component_name
    )


def _merge_worker_metrics(
    worker_metrics: list[list[MetricResult]],
) -> list[MetricResult]:
    """Pool ``concurrency`` workers' read results into one replicate's metrics.

    A single worker's result passes through unchanged (the common, isolated
    case, and identical to a run without a run protocol at all). For more than
    one, every metric keeps its original name — so a concurrent replicate stays
    comparable to an isolated one in the report — but its value becomes the
    mean across workers, and ``detail`` gains ``workers``/``worker_values``:
    the per-worker breakdown the issue asks concurrency to report.
    """
    if len(worker_metrics) == 1:
        return worker_metrics[0]

    by_name: dict[str, list[MetricResult]] = {}
    order: list[str] = []
    for wm in worker_metrics:
        for m in wm:
            if m.name not in by_name:
                by_name[m.name] = []
                order.append(m.name)
            by_name[m.name].append(m)

    merged: list[MetricResult] = []
    for name in order:
        results = by_name[name]
        values = [r.value for r in results]
        merged.append(
            MetricResult(
                name=name,
                value=mean(values),
                unit=results[0].unit,
                detail={"workers": len(results), "worker_values": values},
            )
        )
    return merged


def _run_condition_replicates(
    format_id: str,
    object_uri: str,
    role: str,
    condition: RunCondition,
    replicates: int,
    *,
    base_seed: int = 0,
    locator: str | None = None,
    name: str | None = None,
) -> list[list[MetricResult]]:
    """Run ``replicates`` read passes under ``condition``, one metrics list each.

    Unifies cold and warm through worker-pool *lifetime*, both via a
    ``spawn``-context :class:`ProcessPoolExecutor` — never ``fork``, which
    would silently copy the parent's already-warm GDAL/vsicurl/fsspec state
    into the "fresh" child and defeat a cold measurement:

    * ``cold`` — a brand-new pool per replicate. A new pool means brand-new
      interpreters, so no cache from one replicate can leak into the next.
    * ``warm`` — one persistent pool for the whole condition, given one
      untimed warm-up round before the first *timed* replicate. Each worker's
      GDAL/fsspec state, once warm, carries across every following replicate —
      "the tile server under traffic."

    Workers within one replicate get distinct, reproducible seeds
    (``base_seed + replicate_index * concurrency + worker_index``) so
    concurrent workers sample different windows, as real concurrent clients
    would. ``locator``/``name`` address one component of a bundled object
    (#102), passed through to every worker explicitly.
    """
    ctx = multiprocessing.get_context("spawn")
    k = condition.concurrency

    def _seeds(replicate_index: int) -> list[int]:
        start = base_seed + replicate_index * k
        return list(range(start, start + k))

    def _round(
        executor: ProcessPoolExecutor, replicate_index: int
    ) -> list[MetricResult]:
        futures = [
            executor.submit(
                _read_phase_worker, format_id, object_uri, role, s, locator, name
            )
            for s in _seeds(replicate_index)
        ]
        return _merge_worker_metrics([f.result() for f in futures])

    if condition.cache == "cold":
        results: list[list[MetricResult]] = []
        for r in range(replicates):
            with ProcessPoolExecutor(max_workers=k, mp_context=ctx) as ex:
                results.append(_round(ex, r))
        return results

    # warm: one persistent pool, an untimed warm-up round, then the replicates.
    with ProcessPoolExecutor(max_workers=k, mp_context=ctx) as ex:
        warmup = [
            ex.submit(_read_phase_worker, format_id, object_uri, role, s, locator, name)
            for s in _seeds(-1)
        ]
        for f in warmup:
            f.result()
        return [_round(ex, r) for r in range(replicates)]


def _aggregate_condition_metrics(
    replicate_metrics: list[list[MetricResult]],
) -> list[MetricResult]:
    """Cross-replicate mean/median/stdev per metric name — a condition's ``aggregate``.

    Every replicate is expected to report the same metric names (it's the same
    collector run repeatedly); a name that doesn't appear in every replicate is
    aggregated over however many carried it.
    """
    by_name: dict[str, list[MetricResult]] = {}
    order: list[str] = []
    for rm in replicate_metrics:
        for m in rm:
            if m.name not in by_name:
                by_name[m.name] = []
                order.append(m.name)
            by_name[m.name].append(m)

    aggregate: list[MetricResult] = []
    for name in order:
        results = by_name[name]
        values = [r.value for r in results]
        aggregate.append(
            MetricResult(
                name=name,
                value=mean(values),
                unit=results[0].unit,
                detail={
                    "replicate_values": values,
                    "median": median(values),
                    "stdev": pstdev(values) if len(values) > 1 else 0.0,
                },
            )
        )
    return aggregate


def _build_condition_result(
    phase: str, condition: RunCondition, replicate_metrics: list[list[MetricResult]]
) -> ConditionResult:
    """Assemble a :class:`ConditionResult` from one condition's replicate runs."""
    return ConditionResult(
        phase=phase,
        cache=condition.cache,
        concurrency=condition.concurrency,
        replicates=[
            ConditionReplicate(index=i, metrics=m)
            for i, m in enumerate(replicate_metrics)
        ],
        aggregate=_aggregate_condition_metrics(replicate_metrics),
    )


def _skipped_condition(
    phase: str, condition: RunCondition, exc: Exception
) -> ConditionResult:
    """A best-effort placeholder for a condition that raised instead of measuring."""
    return ConditionResult(
        phase=phase,
        cache=condition.cache,
        concurrency=condition.concurrency,
        replicates=[],
        aggregate=[
            MetricResult(
                name=f"{phase}_condition_skipped", value=0.0, detail={"error": str(exc)}
            )
        ],
    )


def _run_read_conditions(
    config: BenchmarkConfig,
    format_id: str,
    object_uri: str,
    role: str = "sink",
    *,
    locator: str | None = None,
    name: str | None = None,
) -> list[ConditionResult]:
    """Run the read phase under every configured condition (issue #87).

    Best-effort per condition, mirroring :func:`_safe_read_metrics`: a
    transient S3 error under one condition must not drop the others or abort
    the run. ``locator``/``name`` address one component of a bundled object
    (#102).
    """
    replicates, conditions = _parse_run_protocol(config.params)
    results: list[ConditionResult] = []
    for condition in conditions:
        try:
            replicate_metrics = _run_condition_replicates(
                format_id,
                object_uri,
                role,
                condition,
                replicates,
                locator=locator,
                name=name,
            )
            results.append(
                _build_condition_result("read", condition, replicate_metrics)
            )
        except Exception as exc:  # noqa: BLE001 - best-effort, same stance as _safe_read_metrics
            logger.warning(
                "read condition skipped (cache=%s, concurrency=%d): %s",
                condition.cache,
                condition.concurrency,
                exc,
            )
            results.append(_skipped_condition("read", condition, exc))
    return results


def _run_display_conditions(
    config: BenchmarkConfig,
    adapter: FormatAdapter,
    local_target: str,
    object_uri: str,
    artifact_dir: str,
    titiler_endpoint: str | None,
    *,
    locator: str | None = None,
    name: str | None = None,
) -> list[ConditionResult]:
    """Repeat the display phase ``run_protocol.replicates`` times.

    Only the replicate count applies to display — TiTiler's cache is the
    deployed tiler pod's own, out of this process's control, so the resulting
    condition is always reported as ``cache="warm"``, ``concurrency=1``
    regardless of what ``run_protocol.conditions`` asks for (see the module
    note above). Best-effort: a failure yields a skipped condition rather than
    aborting the run. ``locator``/``name`` address one component of a bundled
    object (#102).
    """
    replicates, _conditions = _parse_run_protocol(config.params)
    condition = RunCondition(cache="warm", concurrency=1)
    try:
        replicate_metrics = []
        for _ in range(replicates):
            metrics, _artifacts = _measure_display_object(
                config,
                adapter,
                local_target,
                object_uri,
                artifact_dir,
                titiler_endpoint,
                locator=locator,
                name=name,
            )
            replicate_metrics.append(metrics)
        return [_build_condition_result("display", condition, replicate_metrics)]
    except Exception as exc:  # noqa: BLE001 - best-effort, same stance as _safe_display_metrics
        logger.warning("display run-protocol skipped: %s", exc)
        return [_skipped_condition("display", condition, exc)]


def _pool_conditions(per_product: list[list[ConditionResult]]) -> list[ConditionResult]:
    """Pool per-product condition results into the roll-up's condition spread.

    Groups by ``(phase, cache, concurrency)`` across every sampled product,
    concatenates their replicates (re-indexed), and recomputes ``aggregate``
    over the pooled values — the set-level spread the roll-up carries, the
    same way :func:`_aggregate_write_metrics` pools per-component write
    metrics into a product-level one.
    """
    grouped: dict[tuple[str, str, int], list[ConditionReplicate]] = {}
    order: list[tuple[str, str, int]] = []
    for conditions in per_product:
        for c in conditions:
            key = (c.phase, c.cache, c.concurrency)
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key] += c.replicates

    pooled: list[ConditionResult] = []
    for phase, cache, concurrency in order:
        replicates = grouped[(phase, cache, concurrency)]
        replicate_metrics = [r.metrics for r in replicates]
        pooled.append(
            ConditionResult(
                phase=phase,
                cache=cache,
                concurrency=concurrency,
                replicates=[
                    ConditionReplicate(index=i, metrics=m)
                    for i, m in enumerate(replicate_metrics)
                ],
                aggregate=_aggregate_condition_metrics(replicate_metrics)
                if replicate_metrics
                else [],
            )
        )
    return pooled


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
    pattern = bound.get("pattern")
    limit = bound.get("limit")
    if scope == "product" and limit is None:
        limit = 1
    products = dataset.products(prefix=prefix, pattern=pattern, limit=limit)
    if not products:
        raise ValueError(
            f"dataset {dataset_config.id!r} enumerated no products"
            + (f" under prefix {prefix!r}" if prefix else "")
            + (f" matching pattern {pattern!r}" if pattern else "")
        )

    n_products = len(products)
    logger.info(
        "dataset %s: %d product(s) [format: %s, metrics: %s]",
        dataset_config.id,
        n_products,
        chosen,
        sorted(requested),
    )

    per_product: list[BenchmarkRun] = []
    pooled_sizes: list[int] = []
    for i, product in enumerate(products):
        logger.info(
            "[%d/%d] %s — %d component(s)",
            i + 1,
            n_products,
            product.id,
            len(product.components),
        )
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
        logger.info("[%d/%d] %s done", i + 1, n_products, product.id)

    logger.info(
        "roll-up: pooling %d objects across %d product(s)",
        len(pooled_sizes),
        len(per_product),
    )
    vrt_artifacts = _publish_rgb_vrts(dataset, products, adapter, output_uri)

    policy = tier_policy_from_config(config.tiers)
    rollup_profile = profile_object_sizes(pooled_sizes, policy)
    rollup = BenchmarkRun(
        timestamp=datetime.now(UTC),
        tool_versions=_tool_versions(titiler_endpoint),
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
        conditions=_pool_conditions([run.conditions for run in per_product]),
    )
    return ProductSetResult(per_product=per_product, rollup=rollup)
