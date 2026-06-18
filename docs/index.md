# cng-formats-benchmark

A reusable, **deployable** benchmarking system for cloud-native geospatial
formats (COG, GeoZarr v3, COPC, GeoParquet, and their baselines). It measures
read, write, and display performance plus the **object-size distribution** that
decides whether a layout fits a tiered object store.

The methodology is opinionated and reproducible: **describe your datasets in
config files, deploy the stack, and run.** Nothing about a particular dataset or
provider is baked into the code.

## What it is (and is not)

It is a **deployable component**, not a CI job. The harness runs on real
infrastructure (a workstation, a notebook environment, any Kubernetes cluster)
against real data. Continuous integration only builds the image, unit-tests the
harness, and **proves the stack deploys — it never runs a benchmark.**

Two layers:

- **Harness** — the Python logic that runs the metrics against a dataset,
  packaged as a container image (a batch runner). Importable and unit-testable
  in isolation; no live services required.
- **Deployment** — a stack bundling the runner and its service dependencies
  (notably [TiTiler](https://developmentseed.org/titiler/) for the display
  metric, and S3-compatible object storage), deployable via **docker-compose**
  (local) and a **Helm chart** (Kubernetes).

## Metrics

Per format, per dataset:

| Metric | What it captures |
| --- | --- |
| **object size** | distribution + tier fitness — *first-class*, because a tiered object store makes object size a hard constraint, not a footnote |
| **write** | conversion throughput (baseline → target), including the source read |
| **read** | range-request-aware read latency / throughput (windowed `/vsis3` reads) |
| **display** | TiTiler tile latency per chunk-crossing scenario (tiles touching 1 / 2 / 4 / 9+ internal blocks), with a block-grid layout image |

Processing benchmarks are out of scope, but the harness keeps an extension seam
for them.

## Where to go next

- **[Architecture](architecture.md)** — the design and how the pieces fit
  together, with diagrams.
- **[Getting started](getting-started.md)** — run the synthetic stack
  end-to-end on your machine in a few minutes.
- **[Configuration](configuration.md)** — describe datasets, benchmarks, and
  tier policies as data.
- **[Deployment](deployment.md)** — docker-compose and Helm, local and lab.

## Status

The deployable stack and the COG end-to-end path (convert → object size → read →
display) are implemented and proven in CI against a synthetic fixture, including
the two-provider (source ≠ sink) storage model for real runs. Bringing a real
mission online is a deployment activity; extending to a second dataset/format is
configuration. See [Architecture › Status & roadmap](architecture.md#status-roadmap).

## Licence

MIT. See [LICENSE](https://github.com/developmentseed/cng-formats-benchmark/blob/main/LICENSE).
