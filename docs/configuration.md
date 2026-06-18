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
| `source` | baseline location URI |
| `baseline_format` | e.g. `geotiff` |
| `target_formats` | cloud-native targets to evaluate |
| `grouping_lever` | format-specific knob(s) that control how bytes group into objects |
| `description` | optional human note |

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
| `params` | format params, e.g. `block_size` (COG grouping lever) |
| `source` / `object_source` / `output` | location URIs; CLI flags override them |

!!! note "Why URIs are usually omitted from the file"
    Keeping `source` / `output` out of the committed config makes it portable
    across targets. The deployment supplies the concrete URIs via CLI flags
    (`--source`, `--output`) or Helm values, so the same benchmark file runs
    against the synthetic stack, a kind cluster, or a real bucket unchanged.

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

`read` throughput is **decoded** bytes/s (a fair relative cross-format number),
not bytes over the wire; latency reflects the full range-request round-trip.

`display` does not time a single fixed tile. It inspects the produced COG's block
size and overviews to pick WebMercator tiles that each touch a target number of
internal blocks ("chunks") — 1, 2, 4 and 9+ — and times each, so latency can be
read against chunk-crossing. Unreachable buckets (e.g. on a tiny raster) are
skipped; the targets default to `(1, 2, 4, 9)` and can be overridden via
`params.display_chunk_targets`. A `display_chunk_layout.png` overlaying each
served tile on the block grid is written alongside the object.

## The result

A run produces a `BenchmarkRun` (`cng_benchmark.models`):

- run context — `timestamp`, `tool_versions`, `dataset_id`, `format_id`, `params`
- `object_profile` — `count`, `total_bytes`, `mean`/`median`/`p50`/`p90`/`p95`/`p99`,
  `min_bytes`/`max_bytes`, a `histogram`, and `tier_fit` / `highest_tier`
- `metrics` — a list of `{name, value, unit, detail}` scalars

It is written as `result.json` and rendered to `summary.md`
([report.py](https://github.com/developmentseed/cng-formats-benchmark/blob/main/src/cng_benchmark/report.py)).

## Adding a format

Register a `FormatAdapter` subclass (see `formats/cog.py`) under a name in
`FORMATS`; the runner resolves it by the name used in a config's `formats`. No
CI or manifest change is needed — see
[Architecture › Plug-in seams](architecture.md#plug-in-seams).
