"""Engine contracts and the engine registry of the digital twin.

Contracts only: no physics lives here. Engines are plugged in by name through
:func:`register_engine`; the reference engines are registered by
``oneuniverse.reference`` on import, never imported from this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.registry import Registry


@dataclass(frozen=True)
class Observation:
    """A gridded galaxy-count contrast field plus its survey description."""

    delta_g: np.ndarray
    nbar: float
    bias: float
    mask: np.ndarray | None = None
    box: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductBundle:
    """Named gridded products of a forward engine at redshift ``z``."""

    fields: dict[str, np.ndarray]
    z: float
    provenance: dict[str, Any]


class ReconstructionEngine(ABC):
    """Maps an :class:`Observation` to an estimate of the matter field."""

    name: ClassVar[str]

    @abstractmethod
    def reconstruct(self, observation: Observation, *, cosmo, z: float = 0.0) -> np.ndarray:
        """Return the reconstructed density contrast on the observation grid."""


class ForwardEngine(ABC):
    """Produces a :class:`ProductBundle` from cosmology, box and grid settings."""

    name: ClassVar[str]

    @abstractmethod
    def forward(self, *, cosmo, box: float, n_grid: int, z: float,
                seed: int | None = None, ic_field: np.ndarray | None = None) -> ProductBundle:
        """Run the forward model and return its products at redshift ``z``."""


_REGISTRY: Registry[type] = Registry("twin engine", key=lambda c: c.name)


def register_engine(cls: type) -> type:
    """Register a :class:`ForwardEngine` or :class:`ReconstructionEngine` subclass.

    Usable as a decorator; returns ``cls``. ``ValidationError`` if ``cls`` is not
    one of those subclasses, has no non-empty ``name``, or the name is taken.
    """
    if not (isinstance(cls, type) and issubclass(cls, (ForwardEngine, ReconstructionEngine))):
        raise ValidationError(
            f"{cls!r} is not a ForwardEngine or ReconstructionEngine subclass")
    name = getattr(cls, "name", "")
    if not isinstance(name, str) or not name:
        raise ValidationError(f"{cls.__qualname__} must define a non-empty `name`")
    return _REGISTRY.register(cls, name=name)


def get_engine(name: str) -> type:
    """Engine class registered as ``name``; ``NotFoundError`` if unknown."""
    return _REGISTRY.get(name)


def registered_engines() -> tuple[str, ...]:
    """Names of all registered engines, sorted."""
    return tuple(_REGISTRY.names())
