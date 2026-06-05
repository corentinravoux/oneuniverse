"""Step 3: assemble a total weight from oneuniverse.combine primitives.

``assemble_weight`` collapses to a single ``weight`` column (+ recipe).
``assemble_named_weights`` additionally **keeps every component array** (the
general path: FKP / completeness / systematics / PIP separable for audit and
for estimators that need a specific family).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

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


@dataclass
class NamedWeights:
    total: np.ndarray
    components: Dict[str, np.ndarray]
    recipe: Tuple[str, ...]


def assemble_named_weights(catalog: pd.DataFrame,
                           weights: Mapping[str, Weight],
                           *, out_column: str = "weight"
                           ) -> Tuple[pd.DataFrame, NamedWeights]:
    """Product weight + the per-named-component arrays kept for audit/reuse."""
    out = catalog.copy()
    total = np.ones(len(out), dtype=float)
    components: Dict[str, np.ndarray] = {}
    recipe = []
    for name, w in weights.items():
        a = np.asarray(w(out), dtype=float)
        components[name] = a
        total = total * a
        recipe.append(f"{name}={w!r}")
    out[out_column] = total
    return out, NamedWeights(total=total, components=components,
                             recipe=tuple(recipe))
