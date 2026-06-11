# Deploy

The benchmark system is a deployable stack: the runner image plus its service
dependencies (notably TiTiler, for the display metric). The same stack targets
a workstation and a Kubernetes cluster.

Landing in M2:

- `docker-compose.yml` — local stack (runner + TiTiler + MinIO).
- `helm/cng-benchmark/` — Kubernetes chart (runner as a `Job`, TiTiler as a
  `Deployment` + `Service`), with per-target values files
  (`values-local.yaml`, and others as needed).

CI proves deployability by standing the stack up (docker-compose `up --wait`
and an ephemeral kind cluster `helm install`) and asserting health — it never
runs a benchmark.
