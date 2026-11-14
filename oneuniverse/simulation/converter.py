"""Converter registry: native simulation formats to OUF-Sim stores.

A converter describes one native format (detection, manifest, native products);
the default :meth:`SimConverter.convert` hands those to ``build_store``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

from oneuniverse._core.errors import ValidationError
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
                projection: Literal["reencode", "reference"] = "reencode",
                **build_kw) -> SimStore:
        """Build the store at ``out`` (atomic) from the native simulation at ``path``.

        ``build_kw`` (``n_side``, ``tile_cells``, ``nside_part``) goes to ``build_store``.
        """
        path = Path(path)
        return build_store(out, manifest=self.manifest(path), products=self.products(path),
                           plan=plan, projection=projection, **build_kw)


_REGISTRY: Registry[type[SimConverter]] = Registry("simulation converter", key=lambda c: c.code)


def register(cls: type[SimConverter]) -> type[SimConverter]:
    """Register a converter class under ``cls.code``; usable as a decorator."""
    return _REGISTRY.register(cls)


def get_converter(code: str) -> type[SimConverter]:
    """Converter class for ``code``; ``NotFoundError`` if unknown."""
    return _REGISTRY.get(code)


def detect_converter(path) -> type[SimConverter] | None:
    """The registered converter whose ``detect`` accepts ``path``; None if none does.

    ``ValidationError`` naming every matching code when more than one matches.
    """
    path = Path(path)
    hits = [_REGISTRY.get(code) for code in _REGISTRY.names() if _REGISTRY.get(code).detect(path)]
    if len(hits) > 1:
        raise ValidationError(
            f"{path} matches several converters: {sorted(c.code for c in hits)}; "
            f"pick one with get_converter(code)")
    return hits[0] if hits else None


def registered_codes() -> tuple[str, ...]:
    return tuple(_REGISTRY.names())
