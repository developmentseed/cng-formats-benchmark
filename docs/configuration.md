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
| `sentinel1-otb-rtc` | a `.zip`-per-scene S1Tiling (OTB) RTC gamma0 delivery under `source` | `polarizations` (VV/VH) |
| `swot-raster100m` | one netCDF granule per file under `source` (SWOT L2 HR Raster 100m) | `variables` (CF variables; default `wse`) |
| `swot-lakesp-prior` | a `.zip`-per-pass shapefile delivery under `source` (SWOT L2 HR LakeSP Prior) | none (one `.shp` member = one component) |
| `swot-pixc` | one netCDF point-cloud granule per file under `source` (SWOT L2 HR PIXC) | `groups` (default `pixel_cloud`); `point_variables` (allow-list of carried point vars; default all) / `exclude_variables` (deny-list) |
| `sentinel2-l2b-snow-lis` | one loose GeoTIFF per date, flat under a tile `source` (Sentinel-2 L2B snow / Let-it-Snow) | none (one file = one component) |

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

Not every delivery is a zip. The `swot-raster100m` reader is the *granule*
layout — one netCDF file per granule, flat under `source` — where each selected
CF `variable` becomes one component, read in place via GDAL's CF subdataset
syntax (`NETCDF:"<granule>":<variable>`) and converted to a sharded GeoZarr store
by the existing GeoZarr adapter. `variables` defaults to the primary
water-surface-elevation variable (`wse`).

The `swot-lakesp-prior` reader is the *vector* arm — a cross-mission proof that a
non-raster delivery wires through config alone. It is the same `.zip`-per-scene
shape as the S1/S2 readers, but each pass's zip holds an ESRI Shapefile: every
`.shp` member becomes one component (one pass = one layer), read on the fly via
`/vsizip//vsis3` (the OGR driver finds the `.shx`/`.dbf`/`.prj` sidecars inside
the archive) and converted to a GeoParquet file by the GeoParquet adapter.

The `swot-pixc` reader is the *point-cloud* arm — the same granule layout as
`swot-raster100m`, but a component is a netCDF **group** read as points rather
than a CF raster variable. Each selected `group` (default `pixel_cloud`) becomes
one component, converted to a COPC file by the COPC adapter, whose point loader
reads the group with xarray in place. The COPC is **content-complete**: the
geometry (lon/lat/height → x/y/z) *and* every other per-point variable (`sig0`,
`water_frac`, the quality flags, …) are carried — the geometry as the LAS point and
the rest as LAS **extra dimensions**, each keeping its source dtype where LAS
allows it (so the produced size is a like-for-like basis for comparison, not a
geometry-only fraction). `point_variables` (allow-list) and `exclude_variables`
(deny-list) choose the carried set; the default carries every point-dimensioned
variable. A variable whose name collides with a reserved LAS dimension (e.g.
`classification`) is carried under a suffixed name. The COPC adapter and this
point-cloud path are reused by the CO3D CARS arm (tiled LAZ → COPC).

The `sentinel2-l2b-snow-lis` reader is the **small-file** arm — the clearest
anti-pattern in the study, not the biggest object. Let-it-Snow output is
delivered as loose GeoTIFFs, one per date, flat under a tile prefix (not a
zip), so it reuses the *granule* base's prefix-listing but skips both the
`/vsizip` member step and the CF subdataset step: one file **is** one product
**is** one component, read directly. The component is named for its
acquisition date (parsed from the filename) rather than a fixed variable name,
so the per-date fan-out — the object count and size distribution, which is the
headline here, not the byte saving — stays visible per product. Converting
each date independently is the **per-file baseline to beat**; grouping many
dates into one object is the separate aggregation arm.

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
| `cog` | `block_size` (internal tiling), `codec` (`deflate`/`zstd`/`lzw`/`packbits`/`lzma`/`webp`/`lerc`/`raw`; `deflate` by default) |
| `geozarr` | `chunk_shape` (addressable unit), `shard_shape` (stored object), `codec` (`zstd`/`gzip`/`blosc`/`none`), `multiscale_levels` (overview-pyramid depth, `auto` by default), `scale_offset` (apply a packed source's scale in the array's codec pipeline), `standard_name` (the CF quantity, normally supplied by the dataset reader); `display_titiler_path` selects the multidim/xarray TiTiler router for display |
| `geoparquet` | `row_group_rows` (rows per row group — the addressable unit a bbox query fetches), `spatial_partitioning` (spatially order features so each group's covering bbox is tight), `compression` (`zstd`/`snappy`/`gzip`/`brotli`/`none`; `zstd` by default — not geopandas'/pyarrow's own `snappy` default), `compression_level` (codec effort; `null` uses the codec's default), `data_page_size` (parquet page size in bytes, within a row group; `null` uses pyarrow's default) |
| `copc` | `span` (per-node voxel-grid edge — the per-node point budget ≈ `span**3`), `max_depth` (octree depth; `null` derives it from point density), `scale` (`null` derives LAS quantisation from the extent) |

GeoZarr is a **per-component, 2D** adapter: each source raster becomes one sharded
2D store (a directory of shard objects), the per-component analogue of the COG arm,
so it flows through the same `--source` and `--dataset` paths. `chunk_shape` /
`shard_shape` accept a 2D `[y, x]` or a 3D `[t, y, x]` shape (the trailing two,
spatial, dims are used) and tolerate a swept list of shapes (the first is taken).
Time-stacking the scenes into a 3D cube, and reading a set of objects as a cube,
are deferred follow-ups.

Each store carries an **overview pyramid** — the GeoZarr analogue of COG
overviews, and the thing that makes the display comparison between the two arms
like-for-like. `multiscale_levels: auto` (the default) coarsens by `/2` while the
shorter side stays at or above 256 px, the same rule GDAL follows for COG
overviews; an integer sets the depth explicitly, and `0` writes a single
full-resolution level. A store without a pyramid makes the tile server read
full-resolution chunks and downsample them for every zoomed-out tile, so its
display latency measures the missing pyramid rather than the format — use `0`
only to quantify exactly that.

Every level is a child group (`0` = native resolution, `1` = /2, …) holding the
array, its cell-centre coordinates and its **own** affine transform, so an
overview georeferences on its own. The pyramid is described with the published
Zarr conventions, declared in `zarr_conventions`:

| Convention | What it carries |
| --- | --- |
| [`multiscales`](https://github.com/zarr-conventions/multiscales/blob/v0.1/README.md) | the `layout`: which group holds each level, which level it was derived from, the relative `scale`/`translation` and the `resampling_method` (`average`) |
| [`spatial`](https://github.com/zarr-conventions/spatial/blob/v0.1/README.md) | `spatial:transform` (rasterio/affine coefficient order), `spatial:shape`, `spatial:bbox`, `spatial:dimensions`, `spatial:registration` — per level and on the group |
| [`geo-proj`](https://github.com/zarr-conventions/proj/blob/v0.1/README.md) | the **native** CRS as `proj:code` (or `proj:wkt2`) — nothing is reprojected to web mercator |

GeoZarr 0.4 is deprecated and GeoZarr v1 — which will assemble these conventions
— has not landed, so there is no GeoZarr version to claim conformance to; the
store is written against the published conventions directly, through
[`zarr-cm`](https://github.com/zarr-conventions/zarr-cm), which owns each
convention's uuid, its pinned schema URLs and its validation.

The conventions are the **only** place the georeferencing lives. There is no CF
`spatial_ref` grid-mapping variable duplicating the CRS and transform, no
`grid_mapping` attribute pointing at one, and no `_ARRAY_DIMENSIONS` — Zarr v3
names an array's axes in its own `dimension_names` field, which is what
`spatial:dimensions` refers to. A reader that wants the grid reads the
convention: [rioxarray 0.22+](https://github.com/corteva/rioxarray/releases)
does, so `.rio.crs` / `.rio.transform()` work on any level, and so does this
harness's own display metric, which selects tiles from `proj:code` +
`spatial:transform`.

The grid is written on the level **group and the array inside it**. The
convention lets a group's description stand in for its arrays, but xarray drops
a dataset's attributes when a variable is selected out of it, so a client
holding only the array — which is what a tile server addresses — would otherwise
have nothing to georeference with.

Overviews cost bytes, the same way a COG's do. `summary.md` reports how much of
the stored size the pyramid accounts for, so the size and display numbers can be
read together rather than separately.

`scale_offset` decides how a **packed** source — one whose values are counts to be
multiplied by a scale, like Sentinel-2 MAJA reflectance (DN = reflectance × 10000)
— is encoded:

| | Off (default) | On |
| --- | --- | --- |
| Where the scale lives | CF `scale_factor` / `add_offset` attributes | the array's codec pipeline (`scale_offset` + `cast_value`) |
| Array's declared dtype | the stored integer | float |
| What a plain Zarr reader gets | the raw count — it must know to unscale | physical units |
| What reaches disk | the integer | the integer |

Both write the same bytes, so object sizes and compression ratios stay comparable
with the COG arm — where the equivalent scale is always out-of-band metadata
(GeoTIFF tags, STAC `raster:scale`, GDAL `unscale=True`) that a client has to find
and apply. That contrast is the point of the lever, so `summary.md` reports the
active encoding per array in the "Chunk/shard layout" table. Turning it on needs
the `geozarr` extra's `cast-value-rs`, and it is a no-op for a source with no
scale (a mask, say).

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
| `read` | `metrics/read.py` | `read_window_count` (vector: `read_query_count`), `read_latency_mean/p50/spread`, `read_decoded_throughput` |
| `display` | `metrics/display.py` (+ `display_tiles.py`) | per chunk-bucket `display_{1,2,4,9}chunk_latency_mean/p50`, `display_scenarios`, plus a `display_chunk_layout.png` artifact |

`read` and `display` adapt to the produced object kind: a COG is read with
rasterio over `/vsis3` and served by TiTiler's `/cog` endpoints; a GeoZarr store
is read zarr-natively over fsspec (GDAL cannot read the `sharding_indexed` codec)
and served by a multidim/xarray TiTiler surface (`params.display_titiler_path`); a
GeoParquet file is read with a bbox/row-group spatial query over fsspec (only the
row groups whose covering bbox overlaps are fetched); a COPC file is read with an
octree-node spatial query over fsspec (only the octree nodes that overlap the bbox
are fetched). The vector and point-cloud arms have **no `display` metric** — a
table or point cloud is not a TiTiler raster tile. All raster paths emit the same
`read_*` / `display_*` names; the vector and point-cloud `read` swap
`read_window_count` for `read_query_count` and count returned features / points
rather than pixels.

`read` measures **subsetting reads** — a bbox / sub-zone query, not a
full/sequential scan — the access pattern CNES prioritises (issue #74). Window
and query positions are drawn at random (seeded, so a run is reproducible) over
the object's extent rather than laid out on a fixed grid; the COG path
alternates tile-aligned and tile-unaligned origins so both access patterns are
exercised. `read_latency_spread` (population stdev across the sampled
windows/queries) is reported alongside the mean and p50, with the raw
per-window/query latencies attached in that metric's `detail`.

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
    `multiscale_levels` (the realised pyramid depth), `overview_bytes` (how much
    of the size the pyramid accounts for), `shard_count`; `summary.md` renders a
    "Chunk/shard layout" table + a shard-object count.
  - `geoparquet` → a `GeoParquetLayout`: `geometry_column`, `num_rows`,
    `num_row_groups`, `row_group_rows` (the addressable unit a bbox query fetches),
    and `has_bbox_covering` (whether spatial pushdown to row groups is possible).
  - `copc` → a `CopcLayout`: `num_nodes` (octree nodes — the addressable units),
    `max_depth`, `point_count`, `points_per_node` (the largest node, i.e. the
    realised per-node point budget), and `extra_dimensions` (the carried point
    variables, recovered from the LAS ExtraBytes schema — so the run is
    self-describing about its content); `summary.md` renders an "Octree layout"
    table plus the carried-variable list.

  Captured for every object (no tile server needed). The chunk-aware `display`
  metric also publishes a `display_chunk_layout.png` next to the sampled object
  (the block/chunk grid with each served tile's footprint). A **COPC** run, which
  has no display tiles, instead publishes a `copc_octree_lod.png` — the
  clustered-octree level-of-detail (coarse overview → full detail), the
  point-cloud structural artifact; its sink URI is in the `octree_lod` metric
  detail.
- `metrics` — a list of `{name, value, unit, detail}` scalars

It is written as `result.json` and rendered to `summary.md`
([report.py](https://github.com/developmentseed/cng-formats-benchmark/blob/main/src/cng_benchmark/report.py)).

## Adding a format

Register a `FormatAdapter` subclass (see `formats/cog.py`) under a name in
`FORMATS`; the runner resolves it by the name used in a config's `formats`. No
CI or manifest change is needed — see
[Architecture › Plug-in seams](architecture.md#plug-in-seams).
