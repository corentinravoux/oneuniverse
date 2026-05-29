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
    SAMPLE = "sample"   # Phase 18 — variable-length per-row z-draws.
    HIST = "hist"       # Phase 18 — per-row bin heights on shared edges.


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
    hist_edges: Optional[List[float]] = None
    value_column: str = "z_pdf_values"
    sigma_column: str = "z_pdf_sigma"
    weights_column: str = "z_pdf_weights"
    grid_mask: Optional[List[bool]] = None
    axis_labels: tuple = ("z",)
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
        if self.parameterisation == "hist" and not self.hist_edges:
            raise ValueError(
                "hist parameterisation requires hist_edges of length "
                "n_components+1"
            )
        if (
            self.parameterisation == "hist"
            and self.hist_edges is not None
            and len(self.hist_edges) != self.n_components + 1
        ):
            raise ValueError(
                f"hist_edges length {len(self.hist_edges)} must be "
                f"n_components+1 ({self.n_components + 1})"
            )
        # Normalise sequences to plain Python floats so JSON round-trips
        # (np.float32 / np.float64 are equality-noisy after str↔float).
        if self.grid is not None:
            object.__setattr__(self, "grid", [float(x) for x in self.grid])
        if self.quant_levels is not None:
            object.__setattr__(
                self, "quant_levels", [float(x) for x in self.quant_levels],
            )
        if self.hist_edges is not None:
            object.__setattr__(
                self, "hist_edges", [float(x) for x in self.hist_edges],
            )
        if self.grid_mask is not None:
            object.__setattr__(
                self, "grid_mask", [bool(x) for x in self.grid_mask],
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
            "hist_edges": (
                [float(x) for x in self.hist_edges]
                if self.hist_edges is not None else None
            ),
            "value_column": self.value_column,
            "sigma_column": self.sigma_column,
            "weights_column": self.weights_column,
            "grid_mask": (
                [bool(x) for x in self.grid_mask]
                if self.grid_mask is not None else None
            ),
            "axis_labels": list(self.axis_labels),
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
            hist_edges=(
                list(d["hist_edges"]) if d.get("hist_edges") is not None else None
            ),
            value_column=d.get("value_column", "z_pdf_values"),
            sigma_column=d.get("sigma_column", "z_pdf_sigma"),
            weights_column=d.get("weights_column", "z_pdf_weights"),
            grid_mask=(
                list(d["grid_mask"]) if d.get("grid_mask") is not None else None
            ),
            axis_labels=tuple(d.get("axis_labels", ("z",))),
            extra=dict(d.get("extra", {})),
        )


class ProbabilisticRedshift:
    """Vectorised PDF accessor bound to a :class:`PdfSpec`.

    Supports all three parameterisations:

    * ``interp``: ``values`` are ``p(z)`` sampled on the common ``grid``.
    * ``quant``: ``values`` are ``z(q)`` at the common ``quant_levels``;
      pass ``grid=None`` (it is reconstructed from the spec).
    * ``mixmod``: build via :meth:`from_mixmod` with ``mu``, ``sigma``,
      ``w`` arrays of shape ``(n_rows, n_components)``.
    """

    def __init__(
        self,
        spec: PdfSpec,
        values: np.ndarray,
        grid,
    ) -> None:
        self.spec = spec
        self._mixmod = None

        if spec.parameterisation == "interp":
            if values.ndim != 2:
                raise ValueError(f"values must be 2D, got shape {values.shape}")
            if grid is None:
                raise ValueError("interp requires a grid")
            if values.shape[1] != len(grid):
                raise ValueError(
                    f"values second axis ({values.shape[1]}) must match "
                    f"grid length ({len(grid)})"
                )
            self.values = np.asarray(values, dtype=np.float64)
            self.grid = np.asarray(grid, dtype=np.float64)
        elif spec.parameterisation == "quant":
            if spec.quant_levels is None:
                raise ValueError("quant requires PdfSpec.quant_levels")
            self.values = np.asarray(values, dtype=np.float64)
            self.grid = np.asarray(spec.quant_levels, dtype=np.float64)
        elif spec.parameterisation == "mixmod":
            # Caller should use from_mixmod; this branch only handles the
            # rare direct-construction case for serialisation paths.
            self.values = np.asarray(values, dtype=np.float64)
            self.grid = np.arange(spec.n_components, dtype=np.float64)
        else:
            raise ValueError(
                f"unsupported parameterisation {spec.parameterisation!r}"
            )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, spec: PdfSpec,
    ) -> "ProbabilisticRedshift":
        if spec.parameterisation == "mixmod":
            mu = np.stack(df["z_pdf_values"].to_numpy())
            sigma = np.stack(df["z_pdf_sigma"].to_numpy())
            w = np.stack(df["z_pdf_weights"].to_numpy())
            return cls.from_mixmod(spec, mu, sigma, w)
        raw = df["z_pdf_values"].to_numpy()
        values = np.stack([np.asarray(r, dtype=np.float64) for r in raw])
        if spec.parameterisation == "interp":
            if spec.grid is None:
                raise ValueError("interp PdfSpec.grid must be set")
            return cls(spec, values, np.asarray(spec.grid, dtype=np.float64))
        return cls(spec, values, grid=None)

    @classmethod
    def from_mixmod(
        cls, spec: PdfSpec, mu: np.ndarray, sigma: np.ndarray, w: np.ndarray,
    ) -> "ProbabilisticRedshift":
        if spec.parameterisation != "mixmod":
            raise ValueError("from_mixmod requires parameterisation='mixmod'")
        obj = cls.__new__(cls)
        obj.spec = spec
        obj._mixmod = (
            np.asarray(mu, dtype=np.float64),
            np.asarray(sigma, dtype=np.float64),
            np.asarray(w, dtype=np.float64),
        )
        obj.values = obj._mixmod[0]
        obj.grid = np.arange(spec.n_components, dtype=np.float64)
        return obj

    def __len__(self) -> int:
        return self.values.shape[0]

    # ── Moments ────────────────────────────────────────────────────────

    def mean(self) -> np.ndarray:
        if self.spec.parameterisation == "interp":
            dz = self.grid[1] - self.grid[0]
            return (self.values * self.grid[None, :]).sum(axis=1) * dz
        if self.spec.parameterisation == "quant":
            return np.trapz(self.values, self.grid, axis=1)
        mu, _sigma, w = self._mixmod
        return (w * mu).sum(axis=1)

    def std(self) -> np.ndarray:
        if self.spec.parameterisation == "interp":
            dz = self.grid[1] - self.grid[0]
            m = self.mean()
            var = (
                self.values * (self.grid[None, :] - m[:, None]) ** 2
            ).sum(axis=1) * dz
            return np.sqrt(np.maximum(var, 0.0))
        if self.spec.parameterisation == "quant":
            m = self.mean()
            var = np.trapz((self.values - m[:, None]) ** 2, self.grid, axis=1)
            return np.sqrt(np.maximum(var, 0.0))
        mu, sigma, w = self._mixmod
        m = self.mean()
        second = (w * (mu ** 2 + sigma ** 2)).sum(axis=1)
        return np.sqrt(np.maximum(second - m ** 2, 0.0))

    # ── CDF / PPF / sampling ───────────────────────────────────────────

    def cdf(self) -> np.ndarray:
        if self.spec.parameterisation == "interp":
            dz = self.grid[1] - self.grid[0]
            c = np.cumsum(self.values, axis=1) * dz
            c /= np.maximum(c[:, -1:], 1e-300)
            return c
        raise NotImplementedError(
            "cdf() is only defined for 'interp'; use ppf for 'quant' or "
            "sample for 'mixmod'."
        )

    def ppf(self, q) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))
        if self.spec.parameterisation == "interp":
            c = self.cdf()
            out = np.empty((c.shape[0], q.size), dtype=np.float64)
            for i in range(c.shape[0]):
                out[i] = np.interp(q, c[i], self.grid)
            return out[:, 0] if q.size == 1 else out
        if self.spec.parameterisation == "quant":
            out = np.empty((self.values.shape[0], q.size), dtype=np.float64)
            for i in range(self.values.shape[0]):
                out[i] = np.interp(q, self.grid, self.values[i])
            return out[:, 0] if q.size == 1 else out
        raise NotImplementedError("mixmod.ppf not implemented")

    def sample(self, n_per: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.spec.parameterisation == "interp":
            q = rng.uniform(0.0, 1.0, size=(len(self), n_per))
            c = self.cdf()
            out = np.empty_like(q)
            for i in range(len(self)):
                out[i] = np.interp(q[i], c[i], self.grid)
            return out
        if self.spec.parameterisation == "quant":
            q = rng.uniform(0.0, 1.0, size=(self.values.shape[0], n_per))
            out = np.empty_like(q)
            for i in range(self.values.shape[0]):
                out[i] = np.interp(q[i], self.grid, self.values[i])
            return out
        mu, sigma, w = self._mixmod
        n_rows, K = mu.shape
        out = np.empty((n_rows, n_per), dtype=np.float64)
        for i in range(n_rows):
            comp = rng.choice(K, size=n_per, p=w[i])
            out[i] = rng.normal(mu[i, comp], sigma[i, comp])
        return out
