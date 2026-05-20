"""Probabilistic-redshift support for OUF 2.1.

Defines:

* :class:`PdfParameterisation` — enum of supported PDF representations
  (``interp`` / ``quant`` / ``mixmod``). Mirrors the ``qp`` package
  (Malz & Marshall 2018, arXiv:1806.00014) but stays pure-numpy so
  ``qp`` remains an optional dependency.
* :class:`PdfSpec` — dataclass stored in ``Manifest.pdf_spec``; single
  source of truth for how to reconstruct a PDF from on-disk columns.
* :class:`ProbabilisticRedshift` — vectorised reader returning moments,
  CDF, PPF, samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PdfParameterisation(str, Enum):
    INTERP = "interp"
    QUANT = "quant"
    MIXMOD = "mixmod"


_KNOWN = {p.value for p in PdfParameterisation}


@dataclass(frozen=True)
class PdfSpec:
    """How to reconstruct a probabilistic redshift PDF from on-disk columns.

    Parameters
    ----------
    parameterisation
        One of ``"interp"``, ``"quant"``, ``"mixmod"``.
    n_components
        Fixed length of every PDF array in this dataset (grid points for
        interp, quantiles for quant, mixture components for mixmod).
    grid
        For ``interp``: the common z grid (length ``n_components``).
        For ``mixmod``: ignored. For ``quant``: ignored — use
        ``quant_levels`` instead.
    grid_kind
        ``"z"`` for redshift grid, ``"quantile"`` for quantile levels,
        ``"component"`` for mixture indices.
    quant_levels
        For ``quant``: quantile levels in [0, 1] (length ``n_components``).
    """

    parameterisation: str
    n_components: int
    grid: Optional[List[float]]
    grid_kind: str
    quant_levels: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameterisation not in _KNOWN:
            raise ValueError(
                f"unknown PDF parameterisation {self.parameterisation!r}; "
                f"allowed: {sorted(_KNOWN)}"
            )
        if self.n_components <= 0:
            raise ValueError("n_components must be > 0")
        if self.parameterisation == "interp" and not self.grid:
            raise ValueError("interp parameterisation requires a non-empty grid")
        if self.parameterisation == "quant" and not self.quant_levels:
            raise ValueError("quant parameterisation requires quant_levels")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameterisation": self.parameterisation,
            "n_components": int(self.n_components),
            "grid": list(self.grid) if self.grid is not None else None,
            "grid_kind": self.grid_kind,
            "quant_levels": (
                list(self.quant_levels) if self.quant_levels is not None else None
            ),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PdfSpec":
        return cls(
            parameterisation=d["parameterisation"],
            n_components=int(d["n_components"]),
            grid=list(d["grid"]) if d.get("grid") is not None else None,
            grid_kind=d["grid_kind"],
            quant_levels=(
                list(d["quant_levels"]) if d.get("quant_levels") is not None else None
            ),
            extra=dict(d.get("extra", {})),
        )
