# Configs

Datasets and benchmark runs are described here as **data, not code**:

- `datasets/` — one descriptor per dataset (source, baseline format, target
  formats, object-grouping lever to sweep).
- `benchmarks/` — one descriptor per run (dataset, formats, metrics, tier
  policy).

The schema is pydantic-validated in `cng_benchmark.config`. Worked examples:
`datasets/example_cog.yaml`, `benchmarks/example_cog.yaml`, and the
`synthetic_cog.yaml` pair used by the deployable stack.

📖 **Full reference:** [Configuration](../docs/configuration.md).
