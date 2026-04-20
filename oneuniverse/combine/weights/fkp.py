"""
oneuniverse.combine.weights.fkp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feldman-Kaiser-Peacock (1994) weight ``w = 1/(1 + n̄(z) P₀)``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class FKPWeight(Weight):
    """FKP weight ``w = 1/(1 + n̄(z) P₀)``.

    Parameters
    ----------
    nbar : callable
        ``z -> n̄(z)`` in units consistent with ``P0``.
    P0 : float
        Reference power (``~1e4 (Mpc/h)³`` LRG, ``~3-4e3`` QSO).
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
