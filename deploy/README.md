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

The default smoke runs the COG arm. The **GeoZarr** display metric is served by
TiTiler's built-in `/zarr` endpoint (available since TiTiler 2.0) — the same
TiTiler pod that serves `/cog` for raster tiles. No separate service is needed.

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

## Running a SWOT arm on the lab cluster

Each SWOT arm ships a self-contained lab values overlay — a single `helm install
-f` against the samples staged on Scaleway (`s3://cnes-cng-study/…`). Source and
sink are the same provider there, so `s3Source` stays off (the source role falls
back to the Scaleway sink credentials); flip it on (as `values-lab.yaml` does) to
read the source from the CNES Datalake instead.

| Arm | Overlay | Target | Display |
| --- | --- | --- | --- |
| SWOT-A | `values-lab-swot-raster100m.yaml` | Raster100m → GeoZarr | yes (TiTiler /zarr) |
| SWOT-B | `values-lab-swot-lakesp.yaml` | LakeSP Prior → GeoParquet | no |
| SWOT-C | `values-lab-swot-pixc.yaml` | PIXC pixel cloud → COPC | no |

Each overlay bundles its benchmark + dataset config into `configs` and points
`runner.configFile` / `runner.datasetFile` at them, so the only prerequisite is
the Scaleway sink-credentials secret:

```bash
kubectl create secret generic cng-benchmark-s3 \
  --from-literal=AWS_ACCESS_KEY_ID=<key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret>

helm install swot-pixc helm/cng-benchmark \
  -f helm/cng-benchmark/values-lab-swot-pixc.yaml
```

Adjust `runner.output` (the sink results prefix) and the dataset `source` in the
overlay if your staged paths differ.
