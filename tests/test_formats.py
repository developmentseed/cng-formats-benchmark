"""Tests for the format adapter stubs."""

import pytest

import cng_benchmark.formats  # noqa: F401 — triggers adapter registration
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


# COG (test_cog.py) and GeoZarr (test_geozarr.py) are implemented; the rest are
# stubs until their PRs.
STUB_FORMATS = sorted(set(EXPECTED_LEVERS) - {"cog", "geozarr"})


@pytest.mark.parametrize("name", STUB_FORMATS)
def test_convert_and_enumerate_not_implemented_yet(name):
    adapter = FORMATS.get(name)()
    with pytest.raises(NotImplementedError):
        adapter.convert("src", "dst", {})
    with pytest.raises(NotImplementedError):
        adapter.enumerate_objects("dst")


def test_geozarr_is_a_zarr_store_adapter():
    from cng_benchmark.formats.base import ObjectKind

    adapter = FORMATS.get("geozarr")()
    assert adapter.object_kind is ObjectKind.ZARR_STORE
    assert adapter.target_basename() == "geozarr.zarr"
