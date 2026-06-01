"""RegionSpec — a region of interest in the region catalog.

Carries both an Eulerian geometry (bbox / cone — for observed-structure
pinning) and an optional Lagrangian patch (for zoom-IC re-simulation).
``refs`` are file paths to Pillar-1 artefacts (cluster / void / PV
reconstructions) — paths, NOT Python imports (Rule 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from oneuniverse.simulation.selectors import Cone

# (xlo, xhi, ylo, yhi, zlo, zhi)
Bbox6 = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    kind: str                                  # cluster|void|filament|observed|lagrangian
    eulerian_bbox: Optional[Bbox6] = None
    lagrangian_patch: Optional[Bbox6] = None
    cone: Optional[Cone] = None
    z: Optional[float] = None
    mass: Optional[float] = None
    refs: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if (
            self.eulerian_bbox is None
            and self.lagrangian_patch is None
            and self.cone is None
        ):
            raise ValueError(
                "RegionSpec: at least one geometry "
                "(eulerian_bbox / lagrangian_patch / cone) is required"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "eulerian_bbox": (
                list(self.eulerian_bbox)
                if self.eulerian_bbox is not None else None
            ),
            "lagrangian_patch": (
                list(self.lagrangian_patch)
                if self.lagrangian_patch is not None else None
            ),
            "cone": (
                {"lon": self.cone.lon, "lat": self.cone.lat,
                 "radius_deg": self.cone.radius_deg}
                if self.cone is not None else None
            ),
            "z": self.z,
            "mass": self.mass,
            "refs": list(self.refs),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegionSpec":
        cone_raw = d.get("cone")
        bbox = d.get("eulerian_bbox")
        lag = d.get("lagrangian_patch")
        return cls(
            region_id=d["region_id"],
            kind=d["kind"],
            eulerian_bbox=tuple(bbox) if bbox is not None else None,
            lagrangian_patch=tuple(lag) if lag is not None else None,
            cone=(
                Cone(lon=cone_raw["lon"], lat=cone_raw["lat"],
                     radius_deg=cone_raw["radius_deg"])
                if cone_raw is not None else None
            ),
            z=d.get("z"),
            mass=d.get("mass"),
            refs=tuple(d.get("refs", ())),
        )
