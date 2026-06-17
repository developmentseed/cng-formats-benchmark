"""Pluggable registries for formats and datasets.

The harness is generic: formats and datasets are looked up by name so a config
file can name ``"cog"`` or a dataset id and the runner resolves the
implementation at runtime. Adding a new format or dataset (M3) is therefore a
registration, not a change to the core. A single generic :class:`Registry`
backs both the format and dataset namespaces.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cng_benchmark.datasets.base import Dataset
    from cng_benchmark.formats.base import FormatAdapter


class Registry[T]:
    """A name → object registry with decorator and imperative registration."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: register the decorated object under ``name``."""

        def decorator(obj: T) -> T:
            self.register_instance(name, obj)
            return obj

        return decorator

    def register_instance(self, name: str, obj: T) -> None:
        """Register ``obj`` under ``name``, rejecting duplicate names."""
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        """Look up a registered object, or raise ``KeyError`` listing names."""
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {available}"
            ) from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items


FORMATS: Registry[type[FormatAdapter]] = Registry("format")
DATASETS: Registry[type[Dataset]] = Registry("dataset")
