"""Named selection-weight wrappers passthrough tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from oneuniverse.combine.weights.selection import (
    CompletenessWeight, FiberCollisionWeight, ZFailureWeight,
)


def _df():
    return pd.DataFrame({
        "w_cp": [1.0, 1.5, 2.0],
        "w_noz": [1.0, 1.2, 0.8],
        "w_comp": [0.95, 0.9, 1.0],
    })


def test_fiber_collision_weight_passthrough():
    np.testing.assert_allclose(
        FiberCollisionWeight("w_cp")(_df()),
        _df()["w_cp"].to_numpy(dtype=np.float64),
    )


def test_z_failure_weight_passthrough():
    np.testing.assert_allclose(
        ZFailureWeight("w_noz")(_df()),
        _df()["w_noz"].to_numpy(dtype=np.float64),
    )


def test_completeness_weight_passthrough():
    np.testing.assert_allclose(
        CompletenessWeight("w_comp")(_df()),
        _df()["w_comp"].to_numpy(dtype=np.float64),
    )
