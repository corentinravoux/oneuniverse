"""
oneuniverse.combine.weights.quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quality-mask, column pass-through, and constant-multiplier weights.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class ConstantWeight(Weight):
    """Flat multiplier ``w_i = c``. Used for survey-priority knobs."""

    def __init__(self, value: float = 1.0, name: str = "const") -> None:
        if value < 0:
            raise ValueError("ConstantWeight value must be >= 0")
        self.value = float(value)
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.value)


class ColumnWeight(Weight):
    """Pass through an existing DataFrame column (e.g. ``w_systot``)."""

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


class QualityMaskWeight(Weight):
    """Binary mask ``w = 1`` where ``column {op} threshold`` else 0."""

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
            raise ValueError(
                f"Unknown op '{op}'. Use one of {list(self._OPS)}"
            )
        self.column = column
        self.op = op
        self.threshold = threshold
        self.name = name or f"quality({column}{op}{threshold})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.column not in df.columns:
            raise KeyError(
                f"QualityMaskWeight: missing column '{self.column}'"
            )
        vals = df[self.column].to_numpy()
        return self._OPS[self.op](vals, self.threshold).astype(np.float64)
