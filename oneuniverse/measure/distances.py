"""Distance-indicator atoms for PV/SN (μ, η, v_pec, σ_v). No cosmology."""
from __future__ import annotations

from typing import Sequence, Tuple

import pandas as pd


def attach_distances(catalog: pd.DataFrame, *, columns: Sequence[str]
                     ) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Validate that the requested distance-indicator columns are present."""
    missing = [c for c in columns if c not in catalog.columns]
    if missing:
        raise ValueError(f"attach_distances: missing distance column(s) "
                         f"{missing}")
    return catalog.copy(), tuple(columns)
