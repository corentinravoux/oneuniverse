"""BOSS/eBOSS canonical weight composition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.quality import ColumnWeight
from oneuniverse.combine.weights.selection import (
    FiberCollisionWeight, ZFailureWeight, boss_total_weight,
)


def _df():
    return pd.DataFrame({
        "w_cp": [1.0, 1.5, 2.0],
        "w_noz": [1.0, 1.2, 0.8],
        "w_sys": [1.0, 0.9, 1.1],
        "w_fkp": [0.3, 0.2, 0.4],
    })


def test_boss_total_formula_no_fkp():
    w = boss_total_weight(
        w_sys=ColumnWeight("w_sys"),
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
    )
    got = w(_df())
    d = _df()
    expected = d["w_sys"].to_numpy() * (
        d["w_cp"].to_numpy() + d["w_noz"].to_numpy() - 1.0
    )
    np.testing.assert_allclose(got, expected)


def test_boss_total_formula_with_fkp():
    w = boss_total_weight(
        w_sys=ColumnWeight("w_sys"),
        w_cp=FiberCollisionWeight("w_cp"),
        w_noz=ZFailureWeight("w_noz"),
        w_fkp=ColumnWeight("w_fkp"),
    )
    got = w(_df())
    d = _df()
    expected = (
        d["w_sys"].to_numpy()
        * (d["w_cp"].to_numpy() + d["w_noz"].to_numpy() - 1.0)
        * d["w_fkp"].to_numpy()
    )
    np.testing.assert_allclose(got, expected)
