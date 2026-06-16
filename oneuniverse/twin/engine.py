"""Engine contracts — the pluggable forward / reconstruction interface.

The ADR's plug-in surface (substrate stays forward-model-agnostic; engines
own the physics + inference). Two roles the twin loop needs:

- ``ReconstructionEngine``: data (Observation) → constrained matter field.
  (Wiener now; Hoffman–Ribak, BORG, SBI later.)
- ``ForwardEngine``: field / IC (+ far-field later) → products.
  (linear now; fast-PM mini-sim S8.1 = the second one.)

Generality is *demonstrated* when ≥2 engines satisfy the contract — here
one of each role on the dummy. Kept deliberately thin (YAGNI); it widens
as real engines force it.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple, Type

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec


@dataclass(frozen=True)
class Observation:
    """Mock/real survey data handed to a ReconstructionEngine."""
    delta_g: np.ndarray
    nbar: float
    bias: float = 1.0
    mask: Optional[np.ndarray] = None


@dataclass
class ProductBundle:
    """Outputs a ForwardEngine hands back for ingest/verification."""
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


class ReconstructionEngine(abc.ABC):
    """data → constrained matter field."""
    name: ClassVar[str]
    role: ClassVar[str] = "reconstruction"

    @abc.abstractmethod
    def reconstruct(self, observation: Observation, *,
                    cosmo: CosmologySpec, box_size: float,
                    z: float = 0.0) -> np.ndarray:
        ...


class ForwardEngine(abc.ABC):
    """field / IC (+ far-field) → products."""
    name: ClassVar[str]
    role: ClassVar[str] = "forward"

    @abc.abstractmethod
    def forward(self, *, cosmo: CosmologySpec, box_size: float, n_grid: int,
                z: float = 0.0, seed: int = 0,
                ic: Optional[np.ndarray] = None) -> ProductBundle:
        ...


# --- registry (the plug-in mechanism) ------------------------------------
from oneuniverse._registry import Registry

_REG: "Registry[Type]" = Registry("twin engine")
#: Live internal dict (back-compat).
_ENGINES: Dict[str, Type] = _REG.items_dict


def register_engine(cls: Type) -> Type:
    """Class decorator: register an engine by its ``name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"register_engine: {cls.__name__} needs a `name`")
    _REG.register(cls, name=name)
    return cls


def get_engine(name: str) -> Type:
    return _REG.get(name)


def registered_engines() -> Tuple[str, ...]:
    return tuple(_REG.names())
