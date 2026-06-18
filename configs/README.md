# Configs

Datasets and benchmark runs are described here as **data, not code**:

- `datasets/` — one descriptor per dataset (source, baseline format, target
  formats, object-grouping lever to sweep).
- `benchmarks/` — one descriptor per run (dataset, formats, metrics, tier
  policy).

A dataset's `reader` selects a layout-aware enumerator: `single-object` (the
default — one product, one component) or a multi-component layout such as
`sentinel2-maja` (a `.zip`-per-scene delivery whose `options` pick the
reflectance bands and masks). A benchmark over such a dataset is invoked with
`--dataset` and fans out into a per-product + roll-up tree.

The schema is pydantic-validated in `cng_benchmark.config`. Worked examples:
`datasets/example_cog.yaml` + `benchmarks/example_cog.yaml` (single-object),
`datasets/example_sentinel2_maja.yaml` + `benchmarks/example_sentinel2_maja_cog.yaml`
(multi-component fan-out), and the `synthetic_cog.yaml` pair used by the
deployable stack.

📖 **Full reference:** [Configuration](../docs/configuration.md).
