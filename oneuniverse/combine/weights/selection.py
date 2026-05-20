"""Named selection-weight primitives + BOSS/eBOSS combiner.

Thin, self-documenting wrappers around :class:`ColumnWeight` so a call
site like ``FiberCollisionWeight("w_cp")`` reads the same as the
BOSS-DR12 catalog column name it mirrors.

:func:`boss_total_weight` packages the industry-standard composition
``w = w_sys * (w_cp + w_noz - 1) * [w_fkp]`` (Reid et al. 2016) as one
callable.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight
from oneuniverse.combine.weights.quality import ColumnWeight


class FiberCollisionWeight(ColumnWeight):
    """Close-pair / fiber-collision weight ``w_cp``."""

    def __init__(
        self, column: str = "w_cp", name: Optional[str] = None,
    ) -> None:
        super().__init__(column=column, name=name or "w_cp")


class ZFailureWeight(ColumnWeight):
    """Redshift-failure weight ``w_noz`` (BOSS) / ``w_zfail`` (DESI)."""

    def __init__(
        self, column: str = "w_noz", name: Optional[str] = None,
    ) -> None:
        super().__init__(column=column, name=name or "w_zfail")


class CompletenessWeight(ColumnWeight):
    """Per-object completeness weight ``w_comp``."""

    def __init__(
        self, column: str = "w_comp", name: Optional[str] = None,
    ) -> None:
        super().__init__(column=column, name=name or "w_comp")


class _BossCompositeWeight(Weight):
    """``w = w_sys * (w_cp + w_noz - 1) * [w_fkp]``."""

    def __init__(
        self,
        w_sys: Weight,
        w_cp: Weight,
        w_noz: Weight,
        w_fkp: Optional[Weight],
    ) -> None:
        self.w_sys = w_sys
        self.w_cp = w_cp
        self.w_noz = w_noz
        self.w_fkp = w_fkp
        tag = "sys*(cp+noz-1)"
        if w_fkp is not None:
            tag += "*fkp"
        self.name = tag

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        sys_ = self.w_sys(df)
        cp = self.w_cp(df)
        noz = self.w_noz(df)
        total = sys_ * (cp + noz - 1.0)
        if self.w_fkp is not None:
            total = total * self.w_fkp(df)
        return total


def boss_total_weight(
    w_sys: Weight,
    w_cp: Weight,
    w_noz: Weight,
    w_fkp: Optional[Weight] = None,
) -> Weight:
    """BOSS/eBOSS canonical composition.

    Formula (Reid et al. 2016): ``WEIGHT = WEIGHT_SYSTOT *
    (WEIGHT_NOZ + WEIGHT_CP - 1)`` with optional FKP factor
    ``WEIGHT_FKP``.
    """
    return _BossCompositeWeight(
        w_sys=w_sys, w_cp=w_cp, w_noz=w_noz, w_fkp=w_fkp,
    )
