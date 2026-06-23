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

The default smoke runs the COG arm. The **GeoZarr** display metric needs a tile
server that speaks Zarr v3 (GDAL's COG TiTiler cannot read the sharding codec), so
a `titiler-xarray` service ships under the optional `geozarr` compose profile and a
matching `titilerXarray` toggle (off by default) in the Helm chart:

```bash
docker compose --profile geozarr up titiler-xarray   # bring up the xarray surface
# helm: --set titilerXarray.enabled=true
```
