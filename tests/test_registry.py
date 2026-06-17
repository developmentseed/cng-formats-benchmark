"""Tests for the generic registry and the built-in format registrations."""

import pytest

import cng_benchmark.formats  # noqa: F401 — triggers adapter registration
from cng_benchmark.registry import FORMATS, Registry


def test_register_and_get():
    reg: Registry[int] = Registry("thing")
    reg.register_instance("a", 1)
    assert reg.get("a") == 1
    assert "a" in reg
    assert reg.names() == ["a"]


def test_register_decorator():
    reg: Registry[type] = Registry("thing")

    @reg.register("widget")
    class Widget:
        pass

    assert reg.get("widget") is Widget


def test_duplicate_registration_rejected():
    reg: Registry[int] = Registry("thing")
    reg.register_instance("a", 1)
    with pytest.raises(ValueError):
        reg.register_instance("a", 2)


def test_unknown_name_lists_available():
    reg: Registry[int] = Registry("thing")
    reg.register_instance("a", 1)
    with pytest.raises(KeyError) as exc:
        reg.get("missing")
    assert "a" in str(exc.value)


def test_builtin_formats_registered():
    for name in ("cog", "geozarr", "copc", "geoparquet"):
        assert name in FORMATS
