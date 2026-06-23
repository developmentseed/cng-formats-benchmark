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

## Definition of done for a benchmark arm

A new arm (reader + adapter) is not done when its unit tests pass — it is done
when it can actually **run on the cluster**. The code PR is necessary but not
sufficient; an arm has historically merged green yet been unable to run (the
GeoParquet arm's pyogrio source read did not inherit the S3 endpoint config, fixed
in #32). So an arm PR is complete only when all three hold:

1. **Code + unit tests.** The reader, the format adapter, the read metric routing,
   and their tests (the harness is exercised on synthetic inputs in CI).
2. **Runner-image dependencies are declared *and* verified.** Any new library or
   GDAL/OGR driver the arm needs at run time is:
   - added to the right extra and to the `uv sync` line in
     [`docker/Dockerfile.runner`](../docker/Dockerfile.runner), **and**
   - listed in the per-arm capability contract
     [`cng_benchmark/drivers.py`](../src/cng_benchmark/drivers.py) (`REQUIRED`).

   `cng-benchmark check-drivers` runs at image-build time and in CI's `build`
   job, so a missing driver fails the **build**, not a cluster run. Beware the
   *implicit* ones: the SWOT-A netCDF read relies on the GDAL netCDF driver
   bundled in rasterio's wheel, and the vector arm on the ESRI Shapefile OGR
   driver in pyogrio's *own* bundled GDAL — both are now asserted, not assumed.
3. **On-cluster verification.** A run on the lab cluster against real staged data
   (or an explicit note that the integration run is tracked in the study repo —
   `cnes-cng-study`, `deliverables/D3-benchmarks/`), linked from the PR. CI's
   `deploy-compose` / `deploy-k8s` only prove the stack *stands up* against a tiny
   synthetic fixture; they do not run the arm against its real source.
