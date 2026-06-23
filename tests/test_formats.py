"""Tests for the format adapter registry, levers, and object kinds."""

import pytest

import cng_benchmark.formats  # noqa: F401 — triggers adapter registration
from cng_benchmark.formats.base import ObjectKind
from cng_benchmark.registry import FORMATS

EXPECTED_LEVERS = {
    "cog": "COG internal tiling",
    "geozarr": "Zarr v3 chunk and shard shape",
    "copc": "COPC octree",
    "geoparquet": "GeoParquet row-group size",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_LEVERS))
def test_grouping_lever_describes_the_format(name):
    adapter = FORMATS.get(name)()
    assert EXPECTED_LEVERS[name] in adapter.describe_grouping_lever()


# Each adapter materialises a particular object kind, which the runner branches on
# for upload, the read collector, and the display surface.
EXPECTED_KINDS = {
    "cog": (ObjectKind.RASTER_FILE, "cog.tif"),
    "geozarr": (ObjectKind.ZARR_STORE, "geozarr.zarr"),
    "geoparquet": (ObjectKind.VECTOR_FILE, "geoparquet.parquet"),
    "copc": (ObjectKind.POINT_CLOUD_FILE, "copc.laz"),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_KINDS))
def test_adapter_object_kind_and_basename(name):
    kind, basename = EXPECTED_KINDS[name]
    adapter = FORMATS.get(name)()
    assert adapter.object_kind is kind
    assert adapter.target_basename() == basename
