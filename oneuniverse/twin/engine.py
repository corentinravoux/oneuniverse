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

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse._core.registry import Registry


@dataclass(frozen=True, eq=False)
class Observation:
    """A gridded galaxy-count contrast field plus its survey description.

    Fields:

    - ``delta_g``: ``(n, n, n)`` galaxy count contrast ``n / <n>_mask - 1`` on the
      cell-centred mesh of the periodic box, where ``<n>_mask`` is the mean count per
      cell over the cells selected by ``mask``; it is 0 outside the mask.
    - ``nbar``: mean number density inside the mask, in ``(h/Mpc)^-3`` (i.e. objects per
      ``(Mpc/h)^3``). ``observe_from_view`` defaults it to ``N / box**3`` without a mask
      and to the in-mask density with one.
    - ``bias``: linear galaxy bias the contrast is to be read with.
    - ``mask``: ``None`` (every cell observed) or a bool ``(n, n, n)`` grid, True where
      the survey observes.
    - ``box``: periodic box side in Mpc/h; 0.0 means unset (engines that need it refuse).
    - ``meta``: free-form provenance of how the field was built.

    ``eq=False``: the ndarray fields make field-wise ``==`` and hashing ill-defined,
    so instances compare and hash by identity.
    """

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
        """Run the forward model and return its products at redshift ``z``.

        ``ic_field``, when given, is the **z = 0 linear density contrast** on the
        cell-centred ``n_grid`` mesh of the periodic box (units: Mpc/h box, no growth
        applied); the engine applies the growth to ``z`` itself. It wins over ``seed``.
        """


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
    """Engine class registered as ``name``; ``NotFoundError`` if unknown.

    Engines register when the module that defines them is imported, so an empty or
    incomplete list usually means the plugin module has not been imported yet.
    """
    try:
        return _REGISTRY.get(name)
    except NotFoundError as exc:
        raise NotFoundError(
            f"{exc.args[0] if exc.args else name} (engines register when their plugin "
            f"module is imported: import the module that defines the engine first)") from exc


def registered_engines() -> tuple[str, ...]:
    """Names of all registered engines, sorted."""
    return tuple(_REGISTRY.names())
