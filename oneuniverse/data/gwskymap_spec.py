"""Per-event GW sky-localisation map metadata for OUF 2.5.

GW LIGO/Virgo BAYESTAR / LALInference outputs typically ship as
multi-order MOC HEALPix; consumers rasterise via
:func:`oneuniverse.data.moc.rasterise_moc_to_healpix` to a
fixed-NSIDE numpy array before writing. This spec records that NSIDE
+ ordering + whether per-pixel 3-D distance extras (DISTMU /
DISTSIGMA / DISTNORM) are also stored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class GwSkymapSpec:
    map_nside: int
    map_nest: bool = True
    has_distance_extras: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _is_power_of_two(int(self.map_nside)):
            raise ValueError(
                f"GwSkymapSpec.map_nside must be a power of two, "
                f"got {self.map_nside!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_nside": int(self.map_nside),
            "map_nest": bool(self.map_nest),
            "has_distance_extras": bool(self.has_distance_extras),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GwSkymapSpec":
        return cls(
            map_nside=int(d["map_nside"]),
            map_nest=bool(d.get("map_nest", True)),
            has_distance_extras=bool(d.get("has_distance_extras", False)),
            extra=dict(d.get("extra", {})),
        )
