"""oneuniverse._registry — one registry utility for the whole package.

Replaces four hand-rolled registries (data loaders, simulation converters,
twin engines, oufsim native adapters) that had divergent duplicate-handling and
lookup semantics. Each call site wraps an instance of ``Registry`` and keeps its
existing public functions, so this is an internal consolidation only.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from types import MappingProxyType
from typing import Callable, Dict, Generic, List, Mapping, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → item registry with uniform semantics.

    Parameters
    ----------
    label : str
        Used in error messages (e.g. ``"survey loader"``).
    key : callable, optional
        Derives the registration name from the item when ``name=`` is omitted
        (e.g. ``lambda cls: cls.config.name``).
    """

    def __init__(self, label: str, *, key: Optional[Callable[[T], str]] = None):
        self._label = label
        self._key = key
        self._items: Dict[str, T] = {}

    def register(self, item: T, *, name: Optional[str] = None) -> T:
        if name is None:
            if self._key is None:
                raise ValueError(
                    f"{self._label}: cannot derive a name; pass name= or "
                    f"construct Registry with key="
                )
            name = self._key(item)
        if name in self._items:
            raise ValueError(
                f"{self._label}: '{name}' already registered "
                f"(by {self._items[name]!r})"
            )
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(
                f"{self._label}: unknown '{name}'; known: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    @property
    def items_dict(self) -> Dict[str, T]:
        """The live internal dict (for back-compat shims that exposed it)."""
        return self._items

    @property
    def mapping(self) -> Mapping[str, T]:
        """Read-only view of the registry."""
        return MappingProxyType(self._items)

    def load_entry_points(self, group: str) -> List[str]:
        """Register every plugin advertised under *group*. Returns names added."""
        added: List[str] = []
        for ep in entry_points(group=group):
            if ep.name in self._items:
                continue  # built-in of same name wins; never override silently
            self.register(ep.load(), name=ep.name)
            added.append(ep.name)
        return added
