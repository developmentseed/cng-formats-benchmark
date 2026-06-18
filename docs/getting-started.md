# Getting started

This walks a benchmarker from a clone to a full benchmark run on a synthetic
fixture — no external data or credentials required.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12+ toolchain)
- Docker with the Compose plugin (for the deployable stack)

## 1. Install and test the harness

```bash
uv sync --extra dev          # create .venv, install the project + dev tools
uv run pytest                # run the unit tests
uv run ruff check .          # lint
```

The harness is service-free, so the unit tests run with no Docker, no S3, and
no TiTiler. The geo collectors (`cog` extra: rasterio / rio-cogeo) are exercised
when installed and skipped otherwise.

## 2. Try the CLI

```bash
uv run cng-benchmark version
# validate a config without running anything (no IO):
uv run cng-benchmark run configs/benchmarks/synthetic_cog.yaml
```

## 3. Run the full stack end-to-end (local)

The docker-compose stack stands up **MinIO + TiTiler + the runner**. A seed step
writes a small synthetic fixture raster into MinIO; the runner converts it to a
COG and collects every metric — write, object size, read (`/vsis3` range
requests), and display (tiles from TiTiler) — then writes the produced COG and a
result artifact back to MinIO.

```bash
# Build the runner image the stack uses
docker build -f docker/Dockerfile.runner -t cng-benchmark-runner:dev .

cd deploy
RUNNER_IMAGE=cng-benchmark-runner:dev docker compose up --wait
```

`up --wait` blocks until the whole pipeline has succeeded (MinIO + TiTiler
healthy, the runner exited 0). Host ports `9000/9001/8000` are for inspection
only and can be overridden if taken:

```bash
MINIO_PORT=19000 MINIO_CONSOLE_PORT=19001 TITILER_PORT=18000 \
  docker compose up --wait
```

### Inspect the result

```bash
docker compose exec minio sh -c \
  'mc alias set local http://localhost:9000 minioadmin minioadmin >/dev/null && \
   mc ls --recursive local/bench && \
   mc cat local/bench/results/summary.md'
```

You should see the produced `results/cog/cog.tif`, a machine-readable
`results/result.json`, and a human `results/summary.md` with the object-size
profile and the write/read/display metrics.

Tear down when done:

```bash
docker compose down -v
```

## 4. Read the result

`result.json` is a serialised `BenchmarkRun`:

- `object_profile` — the headline: count, total bytes, percentiles, a histogram,
  and **tier fitness** (which storage tiers the layout satisfies by mean object
  size).
- `metrics` — the scalar measurements: `write_elapsed` / `write_throughput`,
  `read_latency_*` / `read_decoded_throughput`, the per-chunk-bucket
  `display_{1,2,4,9}chunk_latency_*`, etc.

See [Configuration](configuration.md#the-result) for the full shape.

## Next steps

- **Benchmark your own data** — write a dataset + benchmark config
  ([Configuration](configuration.md)) and run it on a deployed stack
  ([Deployment](deployment.md)).
- **Add a format** — register a new `FormatAdapter`
  ([Architecture › Plug-in seams](architecture.md#plug-in-seams)).
- **Deploy to Kubernetes** — the Helm chart, local (kind) or a real cluster
  ([Deployment](deployment.md)).
