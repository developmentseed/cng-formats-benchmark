# Configs

Datasets and benchmark runs are described here as data, not code.

- `datasets/` — one descriptor per dataset: its source location, baseline
  format, candidate target formats, and the object-grouping lever to sweep
  (COG internal tiling, Zarr v3 sharding, COPC octree, GeoParquet row groups /
  partitioning).
- `benchmarks/` — one descriptor per run: which dataset, which formats, which
  metrics, and the storage-tier policy to evaluate object-size fitness against.

The config schema is pydantic-validated in `cng_benchmark.config`; load configs
with `load_dataset_config` / `load_benchmark_config`. Worked examples live in
`datasets/example_cog.yaml` and `benchmarks/example_cog.yaml` and are validated
by the test suite.
