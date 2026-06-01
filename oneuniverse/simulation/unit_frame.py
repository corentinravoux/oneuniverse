"""Sim-side unit + frame declaration.

The single most important metadata for cross-code comparison: every
simulation declares its length / mass / velocity units, h-factor,
comoving-vs-proper, frame, and endianness. Explicit attribution wins —
Gadget vs SWIFT vs CompaSO vs FLAMINGO each have different defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

_ALLOWED_VELOCITY = frozenset({
    "km/s peculiar",     # physical peculiar velocity
    "km/s a",            # a * dx/dt
    "km/s sqrt_a",       # Gadget: v_pec / sqrt(a)
    "code",              # code units; conversion via length/time
})


@dataclass(frozen=True)
class UnitFrameSpec:
    length_unit: str                 # "Mpc/h", "kpc/h", "Mpc"
    mass_unit: str                   # "Msun/h", "Msun", "1e10 Msun/h"
    velocity_unit: str               # one of _ALLOWED_VELOCITY
    time_unit: str = "Gyr"
    h_factor: bool = True            # quantities carry /h
    comoving: bool = True            # positions comoving (vs proper)
    frame: str = "icrs"              # "icrs" | "galactic" | "ecliptic" | "box"
    endianness: str = "native"       # "native" | "little" | "big"

    def __post_init__(self) -> None:
        if self.velocity_unit not in _ALLOWED_VELOCITY:
            raise ValueError(
                f"UnitFrameSpec: unknown velocity_unit "
                f"{self.velocity_unit!r}; allowed: {sorted(_ALLOWED_VELOCITY)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length_unit": self.length_unit,
            "mass_unit": self.mass_unit,
            "velocity_unit": self.velocity_unit,
            "time_unit": self.time_unit,
            "h_factor": bool(self.h_factor),
            "comoving": bool(self.comoving),
            "frame": self.frame,
            "endianness": self.endianness,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnitFrameSpec":
        return cls(
            length_unit=d["length_unit"],
            mass_unit=d["mass_unit"],
            velocity_unit=d["velocity_unit"],
            time_unit=d.get("time_unit", "Gyr"),
            h_factor=bool(d.get("h_factor", True)),
            comoving=bool(d.get("comoving", True)),
            frame=d.get("frame", "icrs"),
            endianness=d.get("endianness", "native"),
        )
