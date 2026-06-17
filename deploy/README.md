# Deploy

The deployable stack: the runner image plus its service dependencies (TiTiler,
S3-compatible storage), via **docker-compose** (local) and a **Helm chart**
(`helm/cng-benchmark/`).

📖 **Full guide:** [Deployment](../docs/deployment.md) — compose and Helm,
local and lab, the source ≠ sink two-provider model, and the deployability CI.

Quick local run:

```bash
docker build -f docker/Dockerfile.runner -t cng-benchmark-runner:dev .
cd deploy && RUNNER_IMAGE=cng-benchmark-runner:dev docker compose up --wait
docker compose down -v
```
