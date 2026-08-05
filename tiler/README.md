# cng-benchmark-tiler

The bench's own TiTiler application: one FastAPI process, three tile
routers, each backed by a different reader, so a display measurement's only
variable is which reader served it.

| Prefix      | Factory                                          | Reader          | What it measures                    |
| ----------- | ------------------------------------------------- | --------------- | ------------------------------------ |
| `/cog`      | `titiler.core.factory.TilerFactory`                | rasterio / GDAL | the COG baseline                     |
| `/zarr`     | `titiler.xarray.factory.TilerFactory`              | stock xarray    | TiTiler out of the box               |
| `/geozarr`  | `titiler.eopf.factory.TilerFactory` (url-bound)    | `GeoZarrReader` | a tuned, index-free GeoZarr reader   |

See `cng_benchmark.metrics.display`'s `path_prefix` for how a benchmark arm
selects a router, and `configs/benchmarks/*.yaml`'s `display_titiler_path`
for how an arm config sets it.

Built and published as its own image (`docker/Dockerfile.tiler`), separate
from the `cng-benchmark` runner package, so the runner's dependency tree
stays free of the full TiTiler/GDAL/GeoZarr-reader stack.
