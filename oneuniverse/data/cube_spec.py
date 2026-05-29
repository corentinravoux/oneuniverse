"""Observational CUBE metadata for OUF 2.5.

Declares the axis layout, axis units, and (for spectral λ axes) the
wavelength convention of an observed cube. Pure observational — no
cosmological assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class CubeSpec:
    """Per-cube axis metadata.

    Parameters
    ----------
    axes
        Ordered tuple of axis names, e.g. ``("ra", "dec", "wavelength")``
        or ``("ra", "dec", "frequency")``.
    axis_units
        Same-length tuple of axis units (``"deg"``, ``"angstrom"``,
        ``"MHz"``, …).
    wavelength_convention
        ``"vacuum"`` or ``"air"`` when a spectral axis is present;
        ``None`` for non-spectral cubes (frequency-only).
    """

    axes: Tuple[str, ...]
    axis_units: Tuple[str, ...]
    wavelength_convention: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.axis_units):
            raise ValueError(
                f"CubeSpec: axes (len {len(self.axes)}) and axis_units "
                f"(len {len(self.axis_units)}) must have equal length"
            )
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))
        object.__setattr__(
            self, "axis_units", tuple(str(u) for u in self.axis_units),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": list(self.axes),
            "axis_units": list(self.axis_units),
            "wavelength_convention": self.wavelength_convention,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CubeSpec":
        return cls(
            axes=tuple(d["axes"]),
            axis_units=tuple(d["axis_units"]),
            wavelength_convention=d.get("wavelength_convention"),
            extra=dict(d.get("extra", {})),
        )
