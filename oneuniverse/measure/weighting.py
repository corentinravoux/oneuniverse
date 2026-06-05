"""Step 3: assemble a total weight from oneuniverse.combine primitives."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from oneuniverse.combine.weights import Weight


def assemble_weight(catalog: pd.DataFrame, weights: Sequence[Weight],
                    *, out_column: str = "weight"
                    ) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Return ``(catalog with out_column = product of weights, recipe)``."""
    out = catalog.copy()
    total = np.ones(len(out), dtype=float)
    recipe = []
    for w in weights:
        total = total * np.asarray(w(out), dtype=float)
        recipe.append(repr(w))
    out[out_column] = total
    return out, tuple(recipe)
