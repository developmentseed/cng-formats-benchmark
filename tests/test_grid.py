"""Tests for the shared grid-equality grouping helper (#102)."""

from cng_benchmark.formats.grid import GridKey, group_by_grid


def _key(name, shape=(10, 10), gt=(0.0, 1.0, 0.0, 10.0, 0.0, -1.0), crs="EPSG:4326"):
    return GridKey(name=name, shape=shape, geotransform=gt, crs=crs)


def test_all_same_grid_forms_one_group():
    items = [_key("wse"), _key("sig0"), _key("area")]
    assert group_by_grid(items) == [["wse", "sig0", "area"]]


def test_different_shape_splits_into_separate_groups():
    items = [_key("a"), _key("b", shape=(20, 20)), _key("c")]
    assert group_by_grid(items) == [["a", "c"], ["b"]]


def test_different_geotransform_splits_into_separate_groups():
    items = [_key("a"), _key("b", gt=(100.0, 1.0, 0.0, 10.0, 0.0, -1.0))]
    assert group_by_grid(items) == [["a"], ["b"]]


def test_different_crs_splits_into_separate_groups():
    items = [_key("a"), _key("b", crs="EPSG:3857")]
    assert group_by_grid(items) == [["a"], ["b"]]


def test_a_lone_component_is_its_own_group():
    assert group_by_grid([_key("solo")]) == [["solo"]]


def test_empty_input_yields_no_groups():
    assert group_by_grid([]) == []


def test_groups_are_returned_in_first_appearance_order():
    items = [_key("b", shape=(5, 5)), _key("a"), _key("c", shape=(5, 5))]
    # "b" (shape 5x5) appears first, so its group comes first even though
    # its second member ("c") appears after "a"'s solo group starts.
    assert group_by_grid(items) == [["b", "c"], ["a"]]
