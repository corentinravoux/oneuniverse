"""
oneuniverse.combine.weights.shear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shear-catalogue weights for both metacalibration and lensfit
pipelines.

For metacalibration (DES Y3, HSC-Y3 metadetect, Rubin), the effective
shear response is

    R_eff = (R11 + R22) / 2 + R_S    (R_S optional)

For lensfit (KiDS-1000, KiDS-450, CFHTLenS), the effective response is

    R_eff = 1 + m_bias

The output weight per row is

    w = shear_weight / (R_eff² + σ_e²)

where σ_e² is optional; if ``sigma_e_cols`` is given, it is computed
as ``e1_err² + e2_err²``, matching the standard shape-noise
convention.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight

_ALLOWED_KINDS = frozenset({"metacal", "lensfit"})


class ShearWeight(Weight):
    """Per-object shear weight for metacal or lensfit pipelines.

    Parameters
    ----------
    kind : str
        ``"metacal"`` (DES Y3, HSC metadetect, Rubin) or
        ``"lensfit"`` (KiDS-1000, CFHTLenS).
    shape_weight_col : str
        Column carrying the catalog-published per-object shape weight.
        Default ``"shear_weight"``.
    R11_col, R22_col : str
        Metacal response columns. Used only when ``kind == "metacal"``.
    R_S_col : str or None
        Optional selection-response column added to ``R_eff`` when
        ``kind == "metacal"``. ``None`` to skip.
    m_col : str
        Lensfit multiplicative-bias column. Used only when
        ``kind == "lensfit"``.
    sigma_e_cols : (str, str) or None
        Per-component shape-noise columns. When given, the row-level
        ``σ_e²`` is added in quadrature to ``R_eff²`` in the
        denominator. Default ``None`` (no shape-noise floor).
    name : str or None
        Override for ``repr``.
    """

    def __init__(
        self,
        kind: str,
        *,
        shape_weight_col: str = "shear_weight",
        R11_col: str = "R11",
        R22_col: str = "R22",
        R_S_col: Optional[str] = "R_S",
        m_col: str = "m_bias",
        sigma_e_cols: Optional[Tuple[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        if kind not in _ALLOWED_KINDS:
            raise ValueError(
                f"unknown ShearWeight kind {kind!r}; "
                f"allowed: {sorted(_ALLOWED_KINDS)}"
            )
        self.kind = kind
        self.shape_weight_col = shape_weight_col
        self.R11_col = R11_col
        self.R22_col = R22_col
        self.R_S_col = R_S_col
        self.m_col = m_col
        self.sigma_e_cols = sigma_e_cols
        self.name = name or f"shear_weight({kind})"

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        if self.shape_weight_col not in df.columns:
            raise KeyError(
                f"ShearWeight: missing shape-weight column "
                f"{self.shape_weight_col!r}"
            )
        w = df[self.shape_weight_col].to_numpy(dtype=np.float64)
        if self.kind == "metacal":
            for c in (self.R11_col, self.R22_col):
                if c not in df.columns:
                    raise KeyError(
                        f"ShearWeight(metacal): missing response column "
                        f"{c!r}"
                    )
            r11 = df[self.R11_col].to_numpy(dtype=np.float64)
            r22 = df[self.R22_col].to_numpy(dtype=np.float64)
            r_eff = 0.5 * (r11 + r22)
            if (
                self.R_S_col is not None
                and self.R_S_col in df.columns
            ):
                r_eff = r_eff + df[self.R_S_col].to_numpy(dtype=np.float64)
        else:  # lensfit
            if self.m_col not in df.columns:
                raise KeyError(
                    f"ShearWeight(lensfit): missing bias column "
                    f"{self.m_col!r}"
                )
            m = df[self.m_col].to_numpy(dtype=np.float64)
            r_eff = 1.0 + m
        denom = r_eff * r_eff
        if self.sigma_e_cols is not None:
            for c in self.sigma_e_cols:
                if c not in df.columns:
                    raise KeyError(
                        f"ShearWeight: missing sigma_e column {c!r}"
                    )
            s1 = df[self.sigma_e_cols[0]].to_numpy(dtype=np.float64)
            s2 = df[self.sigma_e_cols[1]].to_numpy(dtype=np.float64)
            denom = denom + s1 * s1 + s2 * s2
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(denom > 0, w / denom, 0.0)
        return out
