"""SimulationRequest — the orchestration output artefact.

Region selection emits a SimulationRequest describing what to
(re-)simulate. Pillar 3 stores it + tracks its lifecycle; it never
runs the simulation (Rule 4). The external runner updates ``status``
out-of-band and re-ingests output, closing the lineage loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.region import RegionSpec

_IC_STRATEGIES = frozenset({
    "zoom_from_parent_ic",
    "constrained_from_posterior",
    "fresh",
})
_STATUSES = frozenset({"pending", "dispatched", "running", "ingested"})
_PHYSICS = frozenset({"dm", "hydro", "mhd", "rt", "cr"})


@dataclass(frozen=True)
class SimulationRequest:
    request_id: str
    parent_sim: Optional[str]
    region: RegionSpec
    target_resolution: float           # mass or spatial resolution
    physics: Tuple[str, ...]
    cosmology: CosmologySpec
    ic_strategy: str
    code_hint: Optional[str] = None
    status: str = "pending"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ic_strategy not in _IC_STRATEGIES:
            raise ValueError(
                f"SimulationRequest: unknown ic_strategy "
                f"{self.ic_strategy!r}; allowed: {sorted(_IC_STRATEGIES)}"
            )
        if self.status not in _STATUSES:
            raise ValueError(
                f"SimulationRequest: unknown status {self.status!r}; "
                f"allowed: {sorted(_STATUSES)}"
            )
        bad = [p for p in self.physics if p not in _PHYSICS]
        if bad:
            raise ValueError(
                f"SimulationRequest: unknown physics {bad!r}; "
                f"allowed: {sorted(_PHYSICS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "parent_sim": self.parent_sim,
            "region": self.region.to_dict(),
            "target_resolution": float(self.target_resolution),
            "physics": list(self.physics),
            "cosmology": self.cosmology.to_dict(),
            "ic_strategy": self.ic_strategy,
            "code_hint": self.code_hint,
            "status": self.status,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationRequest":
        return cls(
            request_id=d["request_id"],
            parent_sim=d.get("parent_sim"),
            region=RegionSpec.from_dict(d["region"]),
            target_resolution=float(d["target_resolution"]),
            physics=tuple(d.get("physics", ())),
            cosmology=CosmologySpec.from_dict(d["cosmology"]),
            ic_strategy=d["ic_strategy"],
            code_hint=d.get("code_hint"),
            status=d.get("status", "pending"),
            provenance=dict(d.get("provenance", {})),
        )
