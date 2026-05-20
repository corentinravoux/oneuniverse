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

import numpy as np
import pandas as pd


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
        # Normalise sequences to plain Python floats so JSON round-trips
        # (np.float32 / np.float64 are equality-noisy after str↔float).
        if self.grid is not None:
            object.__setattr__(self, "grid", [float(x) for x in self.grid])
        if self.quant_levels is not None:
            object.__setattr__(
                self, "quant_levels", [float(x) for x in self.quant_levels],
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameterisation": self.parameterisation,
            "n_components": int(self.n_components),
            "grid": [float(x) for x in self.grid] if self.grid is not None else None,
            "grid_kind": self.grid_kind,
            "quant_levels": (
                [float(x) for x in self.quant_levels]
                if self.quant_levels is not None else None
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


class ProbabilisticRedshift:
    """Vectorised PDF accessor bound to a :class:`PdfSpec`.

    Current Task 5 supports ``parameterisation == "interp"`` only; the
    ``quant`` and ``mixmod`` branches land in Task 8. All methods return
    one value per row (or ``(n_rows, ...)`` for sampling/CDF).
    """

    def __init__(
        self,
        spec: PdfSpec,
        values: np.ndarray,
        grid: np.ndarray,
    ) -> None:
        if spec.parameterisation != "interp":
            raise NotImplementedError(
                f"parameterisation {spec.parameterisation!r} not yet supported; "
                f"interp only at Task 5."
            )
        if values.ndim != 2:
            raise ValueError(f"values must be 2D, got shape {values.shape}")
        if values.shape[1] != len(grid):
            raise ValueError(
                f"values second axis ({values.shape[1]}) must match "
                f"grid length ({len(grid)})"
            )
        self.spec = spec
        self.values = np.asarray(values, dtype=np.float64)
        self.grid = np.asarray(grid, dtype=np.float64)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, spec: PdfSpec,
    ) -> "ProbabilisticRedshift":
        raw = df["z_pdf_values"].to_numpy()
        values = np.stack([np.asarray(r, dtype=np.float64) for r in raw])
        if spec.grid is None:
            raise ValueError("interp PdfSpec.grid must be set")
        return cls(spec, values, np.asarray(spec.grid, dtype=np.float64))

    def __len__(self) -> int:
        return self.values.shape[0]

    def mean(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        return (self.values * self.grid[None, :]).sum(axis=1) * dz

    def std(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        m = self.mean()
        var = (
            self.values * (self.grid[None, :] - m[:, None]) ** 2
        ).sum(axis=1) * dz
        return np.sqrt(np.maximum(var, 0.0))

    def cdf(self) -> np.ndarray:
        dz = self.grid[1] - self.grid[0]
        c = np.cumsum(self.values, axis=1) * dz
        c /= np.maximum(c[:, -1:], 1e-300)
        return c

    def ppf(self, q) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        c = self.cdf()
        out = np.empty((c.shape[0], q.size), dtype=np.float64)
        for i in range(c.shape[0]):
            out[i] = np.interp(q, c[i], self.grid)
        return out[:, 0] if q.size == 1 else out

    def sample(self, n_per: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q = rng.uniform(0.0, 1.0, size=(len(self), n_per))
        c = self.cdf()
        out = np.empty_like(q)
        for i in range(len(self)):
            out[i] = np.interp(q[i], c[i], self.grid)
        return out
