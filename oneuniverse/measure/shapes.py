"""Shear-source atoms: validate shape columns + assemble the shear weight."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from oneuniverse.combine.weights import ShearWeight

_REQUIRED = ("e1", "e2", "shear_weight")


def attach_shear(catalog: pd.DataFrame, *, kind: str = "metacal",
                 out_column: str = "weight") -> Tuple[pd.DataFrame, str]:
    """Validate shape columns and set ``out_column`` = ShearWeight(kind)."""
    missing = [c for c in _REQUIRED if c not in catalog.columns]
    if missing:
        raise ValueError(f"attach_shear: missing shape column(s) {missing}")
    out = catalog.copy()
    w = ShearWeight(kind=kind)
    out[out_column] = w(out)
    return out, repr(w)
