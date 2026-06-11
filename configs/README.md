# Configs

Datasets and benchmark runs are described here as data, not code.

- `datasets/` — one descriptor per dataset: its source location, baseline
  format, candidate target formats, and the object-grouping lever to sweep
  (COG internal tiling, Zarr v3 sharding, COPC octree, GeoParquet row groups /
  partitioning).
- `benchmarks/` — one descriptor per run: which dataset, which formats, which
  metrics, and the storage-tier policy to evaluate object-size fitness against.

The config schema (pydantic-validated) and worked examples land in M1. Until
then this directory documents the intended layout.
