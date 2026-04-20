"""
oneuniverse.combine.measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`CombinedMeasurements` — frozen output of
:func:`oneuniverse.combine.strategies.combine_weights`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CombinedMeasurements:
    """One row per universal object after a combine strategy.

    Attributes
    ----------
    table : pd.DataFrame
        Columns: ``universal_id``, ``value``, ``variance``, ``weight``,
        ``n_surveys``, ``surveys``.
    strategy : str
    """
    table: pd.DataFrame
    strategy: str

    def __len__(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (
            f"<CombinedMeasurements n={len(self)} strategy={self.strategy!r}>"
        )
