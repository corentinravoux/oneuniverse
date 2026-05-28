"""Observational coordinate metadata for OUF 2.2.

:class:`CoordinateSpec` is **observational only**: it records what
frame and epoch the survey published, plus whether proper-motion /
parallax columns are present. It does **not** assume any cosmology.
Frame conversion (e.g. ICRS to galactic) and epoch propagation
(PM-correction to a later epoch) happen in downstream pillars at
use-time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

_ALLOWED_FRAMES = frozenset({"icrs", "galactic", "ecliptic"})


@dataclass(frozen=True)
class CoordinateSpec:
    """Coordinate-frame metadata declared at dataset level.

    Parameters
    ----------
    frame
        One of ``"icrs"`` (default), ``"galactic"``, ``"ecliptic"``.
    epoch
        Decimal-year position epoch, e.g. ``2016.0`` for GAIA DR3.
        ``None`` means the survey did not declare an epoch.
    proper_motion_available
        True if the dataset carries PM columns.
    parallax_available
        True if the dataset carries a parallax column.
    """

    frame: str = "icrs"
    epoch: Optional[float] = None
    proper_motion_available: bool = False
    parallax_available: bool = False

    def __post_init__(self) -> None:
        if self.frame not in _ALLOWED_FRAMES:
            raise ValueError(
                f"unknown coordinate frame {self.frame!r}; "
                f"allowed: {sorted(_ALLOWED_FRAMES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "epoch": (float(self.epoch) if self.epoch is not None else None),
            "proper_motion_available": bool(self.proper_motion_available),
            "parallax_available": bool(self.parallax_available),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CoordinateSpec":
        return cls(
            frame=d.get("frame", "icrs"),
            epoch=(float(d["epoch"]) if d.get("epoch") is not None else None),
            proper_motion_available=bool(d.get("proper_motion_available", False)),
            parallax_available=bool(d.get("parallax_available", False)),
        )
