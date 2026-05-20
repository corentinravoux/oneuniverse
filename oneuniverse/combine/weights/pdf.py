"""PDF-aware per-object weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class PdfWidthIVarWeight(Weight):
    """``w = 1 / z_pdf_std**2`` — inverse variance on the PDF width."""

    def __init__(
        self, std_column: str = "z_pdf_std", name: str = "ivar(pdf_std)",
    ) -> None:
        self.std_column = std_column
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.std_column not in df.columns:
            raise KeyError(f"PdfWidthIVarWeight: missing '{self.std_column}'")
        s = df[self.std_column].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, 1.0 / (s * s), 0.0)


class PdfMeanRedshiftWeight(Weight):
    """Pass-through of the PDF first moment as a weight.

    Useful when downstream code wants ``<z>`` as a radial weight; keeps
    composition uniform with other :class:`Weight` primitives.
    """

    def __init__(
        self, mean_column: str = "z_pdf_mean", name: str = "pdf_mean",
    ) -> None:
        self.mean_column = mean_column
        self.name = name

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_column not in df.columns:
            raise KeyError(f"PdfMeanRedshiftWeight: missing '{self.mean_column}'")
        return df[self.mean_column].to_numpy(dtype=np.float64)
