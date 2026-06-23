# Configuration

Datasets and benchmark runs are described as **data, not code**, under
`configs/`. The schema is pydantic-validated in `cng_benchmark.config` and
loaded with `load_dataset_config` / `load_benchmark_config`. Worked examples
live in `configs/datasets/` and `configs/benchmarks/` and are validated by the
test suite.

Adding a dataset or a target format must not require touching CI or the
manifests — it is a new file here plus (for a new format) a registered adapter.

## Dataset descriptor

`configs/datasets/<id>.yaml` — names where the baseline lives, its format, the
candidate target formats, and the object-grouping lever to sweep.

```yaml
id: example-raster
description: Generic multi-band raster scene used to exercise the harness.
source: s3://example-bucket/rasters/scene.tif
baseline_format: geotiff
target_formats:
  - cog
  - geozarr
grouping_lever:
  # Candidate object-grouping settings to sweep (format-specific keys).
  cog_block_size: [256, 512, 1024]
  zarr_shard_shape: [[1, 1024, 1024], [1, 2048, 2048]]
```

| Field | Meaning |
| --- | --- |
| `id` | stable dataset identifier |
| `source` | dataset root: a single object (`single-object`) or the prefix scenes/granules live under |
| `baseline_format` | e.g. `geotiff` |
| `target_formats` | cloud-native targets to evaluate |
| `reader` | layout-aware reader that enumerates the dataset's products/components (default `single-object`) |
| `options` | reader-specific picks, validated against the reader's typed `Options` model |
| `grouping_lever` | format-specific knob(s) that control how bytes group into objects |
| `description` | optional human note |

### Readers and layout-specific options

A real delivery is rarely one object. The `reader` selects a layout-aware
[`Dataset`](https://github.com/developmentseed/cng-formats-benchmark/blob/main/src/cng_benchmark/datasets/base.py)
subclass that enumerates the dataset's **products** (scenes/granules) and the
**components** within each (bands, masks, …). Component selection is
layout-specific, so it lives in a typed `options` block owned by the reader — not
in generic benchmark params. Adding a layout is a new subclass + its `Options` +
one registry line; the core config and runner are untouched.

| `reader` | Layout | `options` |
| --- | --- | --- |
| `single-object` (default) | one product, one component = `source` | none |
| `sentinel2-maja` | a `.zip`-per-scene MAJA L2A delivery under `source` | `reflectance` (FRE/SRE), `bands`, `masks` (CLM/EDG/SAT/MG2) |

```yaml
id: sentinel2-l2a-maja
reader: sentinel2-maja          # selects the Sentinel2MajaDataset subclass
source: s3://sentinel2-l2a-sprid/T31TCJ/   # tile root; scenes (zips) underneath
baseline_format: geotiff
target_formats: [cog]
options:                        # validated by Sentinel2MajaOptions
  reflectance: [FRE]
  bands: [B2, B3, B4, B8]
  masks: [CLM, EDG, SAT, MG2]
```

MAJA members are read **on the fly** through GDAL's `/vsizip//vsis3` chain — no
pre-extraction — so the write metric pays the real archive read cost. The
member-name patterns (`…_FRE_B3.tif`, `MASKS/…_CLM_R1.tif`) live in the reader,
never in config.

## Benchmark descriptor

`configs/benchmarks/<id>.yaml` — names which dataset and formats to exercise,
which metrics to collect, and the storage-tier policy that object-size fitness
is judged against.

```yaml
id: synthetic-cog-end-to-end
dataset: synthetic-cog
formats:
  - cog
metrics:
  - write
  - object_size
  - read
  - display
tiers:
  - name: warm
    min_object_bytes: 33554432    # 32 MiB
  - name: cold
    min_object_bytes: 104857600   # 100 MiB
params:
  block_size: 256                 # COG internal tiling — the grouping lever
# Location URIs (optional in the file; usually supplied per-deployment):
# source: s3://bucket/scene.tif   # baseline to convert (COG end-to-end path)
# object_source: s3://bucket/objs/ # existing objects to list (object-only path)
# output: s3://bucket/results/     # where artifacts + the produced object go
```

| Field | Meaning |
| --- | --- |
| `dataset` | the dataset `id` this run targets |
| `formats` | target format(s); the first is used unless overridden |
| `metrics` | any of `write`, `object_size`, `read`, `display` |
| `tiers` | tier policy: a name + minimum recommended mean object size (bytes) |
| `params` | format params: the grouping lever + run shape (see below) |
| `source` / `object_source` / `output` | location URIs; CLI flags override them |

The grouping-lever params are format-specific — the runner resolves the adapter by
name and reads what it needs, so adding a format never changes the schema:

| Format | `params` levers |
| --- | --- |
| `cog` | `block_size` (internal tiling), `compress` |
| `geozarr` | `chunk_shape` (addressable unit), `shard_shape` (stored object), `codec` (`zstd`/`gzip`/`blosc`/`none`), `multiscale_levels`; `display_titiler_path` selects the multidim/xarray TiTiler router for display |

GeoZarr is a **per-component, 2D** adapter: each source raster becomes one sharded
2D store (a directory of shard objects), the per-component analogue of the COG arm,
so it flows through the same `--source` and `--dataset` paths. `chunk_shape` /
`shard_shape` accept a 2D `[y, x]` or a 3D `[t, y, x]` shape (the trailing two,
spatial, dims are used) and tolerate a swept list of shapes (the first is taken).
Time-stacking the scenes into a 3D cube, and reading a set of objects as a cube,
are deferred follow-ups.

!!! note "Why URIs are usually omitted from the file"
    Keeping `source` / `output` out of the committed config makes it portable
    across targets. The deployment supplies the concrete URIs via CLI flags
    (`--source`, `--output`) or Helm values, so the same benchmark file runs
    against the synthetic stack, a kind cluster, or a real bucket unchanged.

### Running over a dataset's products (fan-out)

Pass a dataset descriptor with `--dataset <dataset.yaml>` (or `runner.datasetFile`
in the chart) and the run fans out over the dataset's product(s) instead of a
single `--source` raster. The benchmark carries only **run-shape** params — the
component picks live in the dataset `options`:

```yaml
params:
  block_size: 512
  scope: product-set            # product (one scene) | product-set (many)
  products: {prefix: "2015/", limit: 3}    # bounds a product-set enumeration
  samples: {read: 1, display: 1}           # object_size + write cover ALL objects
```

| Param | Meaning |
| --- | --- |
| `scope` | `product` (one product) or `product-set` (the bounded set) |
| `products.prefix` / `products.limit` | bound which/how many products a set covers — `prefix` is a path prefix **under** `source` (applied server-side for S3), `limit` caps the count |
| `samples.read` / `samples.display` | how many components per product to sample for read/display (default 1) |

`object_size` and `write` cover **every** component; `read` and `display` run on
the first `samples.{read,display}` components. The run writes a product-set tree:

```text
<output>/
  product/<scene-id>/result.json   # ObjectSizeProfile over that scene's components
  product/<scene-id>/summary.md
  rollup/result.json               # profile pooled over ALL products' objects
  rollup/summary.md
  summary.md                       # per-product table + roll-up
```

Each run reuses the `BenchmarkRun` model; `params` carries `product_id` and
`scope` (`product` / `rollup`) to tell the per-product runs apart from the pooled
roll-up.

## Tier policy

Object size is a hard constraint on a tiered object store, so it is first-class.
Each `tiers` entry is a name and the minimum recommended **mean** object size to
qualify for that tier. The result reports every tier the layout satisfies and
the coldest (highest) one — or none, if the objects are too small for any tier.

## Metrics

| Name | Collector | Reports |
| --- | --- | --- |
| `object_size` | `metrics/objects.py` | `object_count`, `total_bytes` + the `object_profile` |
| `write` | `metrics/write.py` | `write_elapsed`, `write_throughput` (output bytes/s, source read included) |
| `read` | `metrics/read.py` | `read_window_count`, `read_latency_mean/p50`, `read_decoded_throughput` |
| `display` | `metrics/display.py` (+ `display_tiles.py`) | per chunk-bucket `display_{1,2,4,9}chunk_latency_mean/p50`, `display_scenarios`, plus a `display_chunk_layout.png` artifact |

`read` and `display` adapt to the produced object kind: a COG is read with
rasterio over `/vsis3` and served by TiTiler's `/cog` endpoints; a GeoZarr store
is read zarr-natively over fsspec (GDAL cannot read the `sharding_indexed` codec)
and served by a multidim/xarray TiTiler surface (`params.display_titiler_path`).
Both emit the same `read_*` / `display_*` metric names.

`read` throughput is **decoded** bytes/s (a fair relative cross-format number),
not bytes over the wire; latency reflects the full range-request round-trip.

`display` does not time a single fixed tile. It inspects the produced object's
block/chunk grid and overview/multiscale levels to pick WebMercator tiles that
each touch a target number of internal blocks/chunks — 1, 2, 4 and 9+ — and times
each, so latency can be read against chunk-crossing. Unreachable buckets (e.g. on
a tiny raster) are skipped; the targets default to `(1, 2, 4, 9)` and can be
overridden via `params.display_chunk_targets`. A `display_chunk_layout.png`
overlaying each served tile on the block/chunk grid is written alongside the
object.

## The result

A run produces a `BenchmarkRun` (`cng_benchmark.models`):

- run context — `timestamp`, `tool_versions`, `dataset_id`, `format_id`, `params`
- `object_profile` — `count`, `total_bytes`, `mean`/`median`/`p50`/`p90`/`p95`/`p99`,
  `min_bytes`/`max_bytes`, a `histogram`, and `tier_fit` / `highest_tier`
- `object_layouts` — per produced object, its **partial-access layout**, typed per
  format (discriminated by `kind`). Every format answers the same "can a client
  fetch part without the whole" question through its own structure:
  - `cog` → a `CogLayout`: `is_tiled` (range-read friendly vs striped),
    `block_width`/`block_height`, `overview_decimations`, `internal_tiles`;
    `summary.md` renders a "Tiling layout" table + a tiled/striped count.
  - `geozarr` → a `GeoZarrLayout`: `chunk_shape` (addressable unit),
    `shard_shape` (stored object), `chunks_per_shard`, `codec`,
    `multiscale_levels`, `shard_count`; `summary.md` renders a "Chunk/shard
    layout" table + a shard-object count.

  Captured for every object (no tile server needed). The chunk-aware `display`
  metric also publishes a `display_chunk_layout.png` next to the sampled object
  (the block/chunk grid with each served tile's footprint).
- `metrics` — a list of `{name, value, unit, detail}` scalars

It is written as `result.json` and rendered to `summary.md`
([report.py](https://github.com/developmentseed/cng-formats-benchmark/blob/main/src/cng_benchmark/report.py)).

## Adding a format

Register a `FormatAdapter` subclass (see `formats/cog.py`) under a name in
`FORMATS`; the runner resolves it by the name used in a config's `formats`. No
CI or manifest change is needed — see
[Architecture › Plug-in seams](architecture.md#plug-in-seams).
