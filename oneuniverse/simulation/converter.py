"""SimConverter ABC + registry — the Layer-3 extensibility surface.

A new simulation code is added by subclassing SimConverter and
implementing four small methods (detect, declare_products,
read_cosmology, read_unit_frame), then @register. The concrete
``convert()`` orchestration (wrap native files + run Layer-1 index
builders + emit manifest) lands in Phase S4; here it raises
NotImplementedError so the contract is testable now.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Type

from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.unit_frame import UnitFrameSpec


class SimConverter(abc.ABC):
    """Per-code converter (Layer 3). Subclasses set ``code`` /
    ``sim_kind`` / ``capabilities`` and implement four methods."""

    code: ClassVar[str]
    sim_kind: ClassVar[str]
    capabilities: ClassVar[BackendCapabilities]

    @abc.abstractmethod
    def detect(self, path: Path) -> bool:
        """Return True if this converter handles the dataset at ``path``."""

    @abc.abstractmethod
    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        """List products found at ``src`` + which Layer-1 indexers each needs."""

    @abc.abstractmethod
    def read_cosmology(self, src: Path) -> CosmologySpec:
        """Parse the run cosmology from ``src``."""

    @abc.abstractmethod
    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        """Parse the unit/frame declaration from ``src``."""

    def convert(self, src: Path, out: Path, *, projection: str = "native",
                build_indexes: bool = True):
        """Wrap native files + build indexes + emit manifest.

        Concrete implementation lands in Phase S4 (needs the Layer-1
        IndexBuilder toolkit + ManifestWriter). Until then this raises.
        """
        raise NotImplementedError(
            "SimConverter.convert is implemented in Phase S4 "
            "(needs the Layer-1 IndexBuilder toolkit)."
        )


from oneuniverse._registry import Registry

_REG: "Registry[Type[SimConverter]]" = Registry("sim converter")
#: Live internal dict (back-compat: detect_converter iterates it directly).
_REGISTRY: Dict[str, Type[SimConverter]] = _REG.items_dict


def register(cls: Type[SimConverter]) -> Type[SimConverter]:
    """Class decorator: register a converter by its ``code``."""
    code = getattr(cls, "code", None)
    if not code:
        raise ValueError(
            f"register: {cls.__name__} must set a non-empty class "
            f"attribute `code`"
        )
    _REG.register(cls, name=code)
    return cls


def get_converter(code: str) -> Type[SimConverter]:
    return _REG.get(code)


def detect_converter(path: Path) -> Optional[Type[SimConverter]]:
    """Return the first registered converter whose ``detect`` matches."""
    for cls in _REGISTRY.values():
        if cls().detect(Path(path)):
            return cls
    return None


def registered_codes() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
