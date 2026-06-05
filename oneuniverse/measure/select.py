"""Step 1-2 of the P1->P2 transform: select a tracer + clean it."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from oneuniverse.data.dataset_view import DatasetView


def select_clean(view: DatasetView, *,
                 z_range: Optional[Tuple[float, float]] = None,
                 columns: Optional[Sequence[str]] = None,
                 quality_column: Optional[str] = None,
                 quality_min: float = 1.0,
                 dropna: bool = True) -> pd.DataFrame:
    """Read + clean a tracer catalog from an OUF POINT view.

    Pushes ``z_range`` to the reader (partition pruning); applies the quality
    cut and drops NaN positions/redshifts in pandas.
    """
    cat = view.read(columns=columns, z_range=z_range)
    if quality_column is not None and quality_column in cat.columns:
        cat = cat[cat[quality_column] >= quality_min]
    if dropna:
        cat = cat.dropna(subset=[c for c in ("ra", "dec", "z")
                                 if c in cat.columns])
    return cat.reset_index(drop=True)
