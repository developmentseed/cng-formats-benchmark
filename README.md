# cng-formats-benchmark

A reusable, **deployable** benchmarking system for cloud-native geospatial
formats (COG, GeoZarr v3, COPC, GeoParquet, and their baselines). It measures
read, write, and display performance plus the **object-size distribution** that
decides whether a layout fits a tiered object store.

The methodology is opinionated and reproducible: describe your datasets in
config files, deploy the stack, and run.

## What it is (and is not)

It is a **deployable component**, not a CI job. The harness runs on real
infrastructure (a workstation, a notebook environment, any Kubernetes cluster)
against real data. Continuous integration only builds the images, unit-tests
the harness, and proves the stack deploys — **it never runs a benchmark**.

Two layers:

- **Harness** — the Python logic that runs the metrics against a dataset,
  packaged as a container image (a batch runner).
- **Deployment** — a stack bundling the runner and its service dependencies
  (notably [TiTiler][titiler] for the display metric), deployable via
  **docker-compose** (local) and a **Helm chart** (Kubernetes). Datasets and
  benchmark runs are described by **config files**, not code.

## Metrics

Per format, per dataset:

- read latency / throughput (HTTP range-request aware)
- write / conversion throughput
- display: TiTiler-served tile latency
- file count, mean / median object size
- **object-size distribution with tier flags** — first-class, because a tiered
  object store makes object size a hard constraint, not a footnote

Processing benchmarks are out of scope, but the harness keeps an extension
seam for them.

## Datasets

Datasets are declared in `configs/datasets/`. Each descriptor names the
source location, the baseline format, the candidate target formats, and the
object-grouping lever to sweep. Bring your own; the descriptors are data, not
code.

## Status

Bootstrapping. M0 lands the harness skeleton, the runner image, and the CI
spine; later milestones add the metric collectors, the config schema, and the
deployable stack.

## Licence

MIT. See [LICENSE](LICENSE).

[titiler]: https://developmentseed.org/titiler/
