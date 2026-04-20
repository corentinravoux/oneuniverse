"""
oneuniverse.combine.weights.ivar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Objective inverse-variance weight ``w = 1 / (σ² + σ_*²)`` with an
optional non-linear velocity-dispersion floor (Howlett 2019; Carreres+
2023).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class InverseVarianceWeight(Weight):
    """Objective inverse-variance weight ``w_i = 1 / (σ_i² + σ_*²)``.

    Parameters
    ----------
    error_column : str
        Error-column name (``"z_spec_err"``, ``"velocity_error"``, ...).
    floor : float, optional
        Noise floor added in quadrature (``σ_*``).
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
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(var > 0, 1.0 / var, 0.0)
        return w
