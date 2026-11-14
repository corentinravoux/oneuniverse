"""Converter registry: native simulation formats to OUF-Sim stores.

A converter describes one native format (detection, manifest, native products);
the default :meth:`SimConverter.convert` hands those to ``build_store``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from oneuniverse._core.registry import Registry
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.manifest import OUFSimManifest
from oneuniverse.simulation.store.build import NativeProduct, build_store
from oneuniverse.simulation.store.read import SimStore


class SimConverter(ABC):
    """Base class of native-format converters; subclasses set ``code``."""

    code: ClassVar[str]

    @classmethod
    @abstractmethod
    def detect(cls, path: Path) -> bool:
        """True if ``path`` is a native simulation this converter reads."""

    @abstractmethod
    def manifest(self, path: Path) -> OUFSimManifest:
        """Manifest of the native simulation, without products."""

    @abstractmethod
    def products(self, path: Path) -> list[NativeProduct]:
        """Native products to put in the store."""

    def convert(self, path: Path, out: Path, *, plan: ExecutionPlan,
                projection: Literal["reencode", "reference"] = "reencode") -> SimStore:
        """Build the store at ``out`` (atomic) from the native simulation at ``path``."""
        path = Path(path)
        return build_store(out, manifest=self.manifest(path), products=self.products(path),
                           plan=plan, projection=projection)


_REGISTRY: Registry[type[SimConverter]] = Registry("simulation converter", key=lambda c: c.code)


def register(cls: type[SimConverter]) -> type[SimConverter]:
    """Register a converter class under ``cls.code``; usable as a decorator."""
    return _REGISTRY.register(cls)


def get_converter(code: str) -> type[SimConverter]:
    """Converter class for ``code``; ``NotFoundError`` if unknown."""
    return _REGISTRY.get(code)


def detect_converter(path) -> type[SimConverter] | None:
    """First registered converter (by code order) whose ``detect`` accepts ``path``, else None."""
    path = Path(path)
    for code in _REGISTRY.names():
        cls = _REGISTRY.get(code)
        if cls.detect(path):
            return cls
    return None


def registered_codes() -> tuple[str, ...]:
    return tuple(_REGISTRY.names())
