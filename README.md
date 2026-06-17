# cng-formats-benchmark

A reusable, **deployable** benchmarking system for cloud-native geospatial
formats (COG, GeoZarr v3, COPC, GeoParquet, and their baselines). It measures
read, write, and display performance plus the **object-size distribution** that
decides whether a layout fits a tiered object store.

The methodology is opinionated and reproducible: **describe your datasets in
config files, deploy the stack, and run.** Nothing about a particular dataset or
provider is baked into the code.

📖 **Full documentation:** <https://developmentseed.github.io/cng-formats-benchmark/>
(source in [docs/](docs/index.md), built with MkDocs — see below).

## What it is (and is not)

It is a **deployable component**, not a CI job. The harness runs on real
infrastructure (a workstation, a notebook environment, any Kubernetes cluster)
against real data. CI builds the image, unit-tests the harness, and **proves the
stack deploys — it never runs a benchmark.**

- **Harness** — the Python logic that runs the metrics against a dataset,
  packaged as a container image. Importable and unit-testable in isolation.
- **Deployment** — a stack bundling the runner and its service dependencies
  ([TiTiler][titiler] for the display metric, S3-compatible storage), deployable
  via **docker-compose** (local) and a **Helm chart** (Kubernetes).

## Metrics

Per format, per dataset: **object size** (distribution + tier fitness, the
first-class differentiator), **write** (conversion throughput), **read**
(range-request-aware latency/throughput), and **display** (TiTiler tile
latency). Processing benchmarks are out of scope, with an extension seam kept
for them.

## Quick start

```bash
uv sync --extra dev          # install harness + dev tools
uv run pytest                # unit tests (service-free)

# Run the full stack end-to-end on a synthetic fixture (no external data):
docker build -f docker/Dockerfile.runner -t cng-benchmark-runner:dev .
cd deploy && RUNNER_IMAGE=cng-benchmark-runner:dev docker compose up --wait
```

See [Getting started](docs/getting-started.md) for the walkthrough,
[Configuration](docs/configuration.md) to describe your own datasets, and
[Deployment](docs/deployment.md) for Kubernetes.

## Documentation

The docs live in [`docs/`](docs/) and build with [MkDocs Material][mkdocs]:

```bash
uv sync --extra docs
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict # build the static site
```

- [Architecture](docs/architecture.md) — design and diagrams
- [Getting started](docs/getting-started.md) — run it in a few minutes
- [Configuration](docs/configuration.md) — datasets, benchmarks, tiers, metrics
- [Deployment](docs/deployment.md) — docker-compose and Helm

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

MIT. See [LICENSE](LICENSE).

[titiler]: https://developmentseed.org/titiler/
[mkdocs]: https://squidfunk.github.io/mkdocs-material/
