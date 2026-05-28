"""Observational spectrum metadata for OUF 2.2 SIGHTLINE datasets.

Pure observational. Tells consumers whether wavelengths are vacuum or
air, log- or linear-binned, in what unit, and whether already
rest-frame-corrected. No cosmological choice; just a column-axis label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

_ALLOWED_CONVENTIONS = frozenset({"vacuum", "air"})
_ALLOWED_UNITS = frozenset({"angstrom", "nanometer", "micron"})


@dataclass(frozen=True)
class SpectrumSpec:
    """Wavelength-axis metadata for a SIGHTLINE dataset.

    Parameters
    ----------
    wavelength_convention
        ``"vacuum"`` (BOSS+ / DESI / Euclid) or ``"air"`` (legacy
        SDSS, some VIPERS).
    log_binned
        True if pixels are uniform in ``log10(lambda)``.
    rest_frame_corrected
        True if the wavelength axis has already been divided by
        ``(1 + z)``.
    wavelength_unit
        One of ``"angstrom"`` (default), ``"nanometer"``, ``"micron"``.
    """

    wavelength_convention: str
    log_binned: bool = True
    rest_frame_corrected: bool = False
    wavelength_unit: str = "angstrom"

    def __post_init__(self) -> None:
        if self.wavelength_convention not in _ALLOWED_CONVENTIONS:
            raise ValueError(
                f"unknown wavelength_convention "
                f"{self.wavelength_convention!r}; "
                f"allowed: {sorted(_ALLOWED_CONVENTIONS)}"
            )
        if self.wavelength_unit not in _ALLOWED_UNITS:
            raise ValueError(
                f"unknown wavelength_unit {self.wavelength_unit!r}; "
                f"allowed: {sorted(_ALLOWED_UNITS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength_convention": self.wavelength_convention,
            "log_binned": bool(self.log_binned),
            "rest_frame_corrected": bool(self.rest_frame_corrected),
            "wavelength_unit": self.wavelength_unit,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpectrumSpec":
        return cls(
            wavelength_convention=d["wavelength_convention"],
            log_binned=bool(d.get("log_binned", True)),
            rest_frame_corrected=bool(d.get("rest_frame_corrected", False)),
            wavelength_unit=d.get("wavelength_unit", "angstrom"),
        )
