"""
oneuniverse.combine.weights.base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract :class:`Weight` and the multiplicative :class:`ProductWeight`
composition. Concrete families live in sibling modules (``ivar``,
``fkp``, ``quality``).
"""
from __future__ import annotations

import abc

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
                f"{type(self).__name__}: expected shape ({len(df)},), "
                f"got {w.shape}"
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
