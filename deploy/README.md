# Deploy

The benchmark system is a deployable stack: the runner image plus its service
dependencies (TiTiler for the display metric, and S3-compatible object storage).
The same stack targets a workstation, an ephemeral CI cluster, and the real lab
Kubernetes cluster — datasets and runs are configuration, not code.

**MinIO is a disposable, S3-compatible stand-in for local and CI only.** Real
runs target a real S3 provider (the lab bucket); the runner and TiTiler use the
same S3 code path against both — only the endpoint and credentials differ. CI
never runs a benchmark on real data; it only proves the stack deploys.

## docker-compose (local)

`docker-compose.yml` stands up MinIO + TiTiler + the runner. A seed step writes
a small synthetic fixture raster into MinIO; the runner converts it to a COG and
collects the full metric set end-to-end — write (conversion throughput),
object-size profile, read (windowed `/vsis3` range requests), and display (tile
latency against TiTiler) — then writes the produced COG and a result artifact
(`result.json` + `summary.md`) back to MinIO.

```bash
docker build -f docker/Dockerfile.runner -t cng-benchmark-runner:dev .
cd deploy
RUNNER_IMAGE=cng-benchmark-runner:dev docker compose up --wait
# host ports (9000/9001/8000) are for inspection only and can be overridden:
#   MINIO_PORT=19000 TITILER_PORT=18000 docker compose up --wait
docker compose down -v
```

## Helm (Kubernetes)

`helm/cng-benchmark/` deploys the runner as a `Job` (with seed/bucket
initContainers), TiTiler as a `Deployment` + `Service` (probed on `/healthz`),
and the benchmark configs via a `ConfigMap`. MinIO is an optional in-cluster
`Deployment` gated by `minio.enabled`.

- `values-local.yaml` — kind + in-cluster MinIO + synthetic fixture (used by CI).
- `values-lab.yaml` — the real Scaleway cluster: MinIO off, credentials from a
  pre-created `Secret`, results to an external S3 bucket.

```bash
helm lint deploy/helm/cng-benchmark -f deploy/helm/cng-benchmark/values-local.yaml
helm template t deploy/helm/cng-benchmark \
  -f deploy/helm/cng-benchmark/values-local.yaml | kubeconform -strict

kind create cluster --name cngbench
kind load docker-image cng-benchmark-runner:ci --name cngbench
helm install bench deploy/helm/cng-benchmark -f deploy/helm/cng-benchmark/values-local.yaml
kubectl wait --for=condition=complete --timeout=240s job/bench-cng-benchmark-runner
```

### Adding a dataset / benchmark

Add the benchmark YAML to the chart's `configs` map (and point `runner.configFile`
/ `runner.objectSource` / `runner.output` at it) — no template or CI change. For
docker-compose, drop the YAML under `configs/benchmarks/` and reference it on the
runner command.

## CI

`deploy-compose` runs `docker compose up --wait` and asserts the artifact lands
in MinIO. `deploy-k8s` spins up an ephemeral kind cluster, loads the built image,
`helm install`s the local overlay, and asserts TiTiler readiness and the runner
`Job` reaching `Complete`. Both use only the synthetic fixture.
