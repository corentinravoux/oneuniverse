"""
oneuniverse.weight.base
~~~~~~~~~~~~~~~~~~~~~~~
Weight primitives.

A :class:`Weight` is any callable that turns a catalog DataFrame into a 1-D
``(n_rows,)`` float array of non-negative weights.  Weights are *composable*:
multiply two Weight instances to get a product weight, matching the eBOSS /
DESI convention ``w_tot = w_systot · w_cp · w_noz · w_FKP``.

Provided objective classes
--------------------------
- :class:`InverseVarianceWeight` — ``w = 1/σ²`` on a user-chosen error column,
  with an optional velocity-dispersion floor (non-linear σ_* floor used for
  peculiar-velocity analyses; Howlett 2019, Carreres+ 2023).
- :class:`FKPWeight` — ``w = 1/(1 + n̄(z) P₀)`` with a user-supplied ``n̄(z)``
  callable (Feldman, Kaiser & Peacock 1994).
- :class:`ColumnWeight` — pass-through of an existing column
  (e.g. ``w_systot``, ``w_cp``, ``w_noz`` already stored in the catalog).
- :class:`QualityMaskWeight` — binary ``(quality >= threshold)`` mask, e.g.
  ``ZWARNING == 0`` for SDSS or ``Q >= 3`` for 6dFGS.
- :class:`ConstantWeight` — flat multiplier; used for subjective
  survey-priority knobs in :mod:`oneuniverse.weight.combine`.

All weights subclass :class:`Weight` and can be composed via ``*`` into
:class:`ProductWeight`.
"""

from __future__ import annotations

import abc
from typing import Callable, Optional

import numpy as np
import pandas as pd


class Weight(abc.ABC):
    """Abstract base class for per-object weights.

    Subclasses override :meth:`compute`.  The :meth:`__call__` wrapper
    sanity-checks the output shape and nonnegativity.
    """

    name: str = "weight"

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        w = np.asarray(self.compute(df), dtype=np.float64)
        if w.shape != (len(df),):
            raise ValueError(
                f"{type(self).__name__}: expected shape ({len(df)},), got {w.shape}"
            )
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError(
                f"{type(self).__name__}: weights must be finite and non-negative"
            )
        return w

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> np.ndarray: ...

    def __mul__(self, other: "Weight") -> "ProductWeight":
        return ProductWeight(self, other)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class ProductWeight(Weight):
    """Elementwise product of two weights (eBOSS/DESI composition)."""

    def __init__(self, a: Weight, b: Weight) -> None:
        self.a = a
        self.b = b
        self.name = f"{a.name}*{b.name}"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        return self.a(df) * self.b(df)


class ConstantWeight(Weight):
    """Flat multiplier, ``w_i = c``.  Used for survey-priority knobs."""

    def __init__(self, value: float = 1.0, name: str = "const") -> None:
        if value < 0:
            raise ValueError("ConstantWeight value must be >= 0")
        self.value = float(value)
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.value)


class ColumnWeight(Weight):
    """Use an existing DataFrame column as a weight (e.g. ``w_systot``)."""

    def __init__(self, column: str, name: Optional[str] = None) -> None:
        self.column = column
        self.name = name or column

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise KeyError(
                f"ColumnWeight: column '{self.column}' not in DataFrame "
                f"(available: {list(df.columns)})"
            )
        return df[self.column].to_numpy(dtype=np.float64)


class InverseVarianceWeight(Weight):
    """Objective inverse-variance weight ``w_i = 1 / (σ_i² + σ_*²)``.

    Parameters
    ----------
    error_column : str
        Name of the error column (e.g. ``"z_spec_err"``, ``"velocity_error"``,
        ``"dmu_error"``).
    floor : float, optional
        Noise floor added in quadrature (``σ_*``).  For peculiar velocities a
        typical choice is ``σ_* ≈ 250 km/s`` to absorb the non-linear velocity
        dispersion (Howlett 2019; Carreres+ 2023).  Default 0.
    """

    def __init__(
        self,
        error_column: str,
        floor: float = 0.0,
        name: Optional[str] = None,
    ) -> None:
        if floor < 0:
            raise ValueError("InverseVarianceWeight.floor must be >= 0")
        self.error_column = error_column
        self.floor = float(floor)
        self.name = name or f"ivar({error_column})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.error_column not in df.columns:
            raise KeyError(
                f"InverseVarianceWeight: missing column '{self.error_column}'"
            )
        sigma = df[self.error_column].to_numpy(dtype=np.float64)
        var = sigma * sigma + self.floor * self.floor
        # Guard against zero/negative variances
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(var > 0, 1.0 / var, 0.0)
        return w


class FKPWeight(Weight):
    """Feldman-Kaiser-Peacock (1994) weight ``w = 1/(1 + n̄(z) P₀)``.

    Parameters
    ----------
    nbar : callable
        ``z -> n̄(z)`` in units consistent with ``P0`` (usually
        ``(h/Mpc)³``).
    P0 : float
        Reference power; ``~1e4 (Mpc/h)³`` for LRGs, ``~3-4e3`` for QSOs.
    z_column : str
        Redshift column name (default ``"z"``).
    """

    def __init__(
        self,
        nbar: Callable[[np.ndarray], np.ndarray],
        P0: float,
        z_column: str = "z",
        name: str = "fkp",
    ) -> None:
        self.nbar = nbar
        self.P0 = float(P0)
        self.z_column = z_column
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        z = df[self.z_column].to_numpy(dtype=np.float64)
        n = np.asarray(self.nbar(z), dtype=np.float64)
        return 1.0 / (1.0 + n * self.P0)


class QualityMaskWeight(Weight):
    """Binary quality mask: ``w = 1`` where ``column {op} threshold`` else 0.

    Used for e.g. ``ZWARNING == 0`` (SDSS) or ``Q >= 3`` (6dFGS).
    """

    _OPS = {
        "==": np.equal, "!=": np.not_equal,
        ">": np.greater, ">=": np.greater_equal,
        "<": np.less, "<=": np.less_equal,
    }

    def __init__(
        self,
        column: str,
        op: str,
        threshold,
        name: Optional[str] = None,
    ) -> None:
        if op not in self._OPS:
            raise ValueError(f"Unknown op '{op}'. Use one of {list(self._OPS)}")
        self.column = column
        self.op = op
        self.threshold = threshold
        self.name = name or f"quality({column}{op}{threshold})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise KeyError(f"QualityMaskWeight: missing column '{self.column}'")
        vals = df[self.column].to_numpy()
        return self._OPS[self.op](vals, self.threshold).astype(np.float64)
