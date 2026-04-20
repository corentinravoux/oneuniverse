"""
oneuniverse.combine.catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`WeightedCatalog` — high-level facade for cross-survey weighting.

Workflow
--------
1. Build a ONEUID index once via ``database.build_oneuid()``.
2. Construct :class:`WeightedCatalog` with :meth:`from_oneuid`.
3. Register :class:`Weight` per survey via :meth:`add_weight`.
4. Call :meth:`combine` to merge concurrences.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.measurements import CombinedMeasurements
from oneuniverse.combine.strategies import combine_weights
from oneuniverse.combine.weights.base import Weight
from oneuniverse.data.oneuid import OneuidIndex
from oneuniverse.data.oneuid_crossmatch import CrossMatchResult

logger = logging.getLogger(__name__)


class WeightedCatalog:
    """Cross-survey catalog with composable per-survey weights.

    Parameters
    ----------
    catalogs : dict[str, DataFrame]
        Mapping ``survey_name -> catalog``.  Each must have at least
        ``ra``, ``dec`` columns; ``z`` is needed for the redshift cut and
        the inverse-variance combine step.
    """

    def __init__(self, catalogs: Dict[str, pd.DataFrame]) -> None:
        if not catalogs:
            raise ValueError("WeightedCatalog needs at least one survey")
        self.catalogs: Dict[str, pd.DataFrame] = {
            name: df.reset_index(drop=True).copy()
            for name, df in catalogs.items()
        }
        self._weights: Dict[str, List[Weight]] = {n: [] for n in self.catalogs}
        self._match: Optional[CrossMatchResult] = None
        self._weighted_long: Optional[pd.DataFrame] = None

    def add_weight(self, survey: str, weight: Weight) -> "WeightedCatalog":
        """Attach a :class:`Weight` to *survey*.  Multiple weights are multiplied."""
        if survey not in self.catalogs:
            raise KeyError(f"Unknown survey '{survey}'")
        self._weights[survey].append(weight)
        return self

    def total_weight(self, survey: str) -> np.ndarray:
        """Return the elementwise product of all weights for *survey*."""
        df = self.catalogs[survey]
        ws = self._weights[survey]
        if not ws:
            return np.ones(len(df), dtype=np.float64)
        out = np.ones(len(df), dtype=np.float64)
        for w in ws:
            out *= w(df)
        return out

    @classmethod
    def from_oneuid(
        cls, index: OneuidIndex, database,
    ) -> "WeightedCatalog":
        """Build a :class:`WeightedCatalog` from a prebuilt ONEUID index.

        The catalog is already keyed on ``index.oneuid``; no self-managed
        cross-match runs. Register weights with :meth:`add_weight` then
        call :meth:`combine`.
        """
        from oneuniverse.data.converter import read_oneuniverse_parquet

        datasets = index.datasets()
        catalogs: Dict[str, pd.DataFrame] = {
            ds: read_oneuniverse_parquet(database.get_path(ds))
            for ds in datasets
        }
        wc = cls(catalogs)
        match_tbl = index.table.rename(
            columns={"oneuid": "universal_id", "dataset": "survey"},
        ).copy()
        wc._match = CrossMatchResult(
            table=match_tbl,
            n_groups=index.n_unique,
            n_multi=index.n_multi,
        )
        return wc

    def _build_long_table(self) -> pd.DataFrame:
        """Long-format table: one row per (universal_id, survey, original-row)."""
        if self._match is None:
            raise RuntimeError(
                "WeightedCatalog has no cross-match. Build via "
                "WeightedCatalog.from_oneuid(index, database)."
            )
        parts = []
        for survey, df in self.catalogs.items():
            sub = self._match.table[self._match.table["survey"] == survey]
            idx = sub["row_index"].to_numpy()
            joined = df.iloc[idx].reset_index(drop=True)
            joined["universal_id"] = sub["universal_id"].to_numpy()
            joined["survey"] = survey
            joined["weight"] = self.total_weight(survey)[idx]
            parts.append(joined)
        return pd.concat(parts, ignore_index=True)

    def combine(
        self,
        value_col: str,
        variance_col: str,
        strategy: str = "best_only",
        survey_alpha: Optional[Dict[str, float]] = None,
    ) -> CombinedMeasurements:
        """Combine concurrences into one row per universal object.

        See :func:`oneuniverse.combine.strategies.combine_weights`.
        """
        self._ensure_long_table()
        return combine_weights(
            self._weighted_long,
            value_col=value_col,
            variance_col=variance_col,
            strategy=strategy,
            survey_alpha=survey_alpha,
        )

    def concurrences(self, universal_id: int) -> pd.DataFrame:
        """All rows (one per survey) for a universal object, with the
        per-survey weight applied as a ``weight`` column."""
        self._ensure_long_table()
        return self._weighted_long[
            self._weighted_long["universal_id"] == universal_id
        ].reset_index(drop=True)

    def n_universal(self) -> int:
        if self._match is None:
            raise RuntimeError("no cross-match; use from_oneuid()")
        return self._match.n_groups

    def n_multi_survey(self) -> int:
        if self._match is None:
            raise RuntimeError("no cross-match; use from_oneuid()")
        return self._match.n_multi

    def _ensure_long_table(self) -> None:
        if self._weighted_long is not None:
            return
        if self._match is None:
            raise RuntimeError(
                "WeightedCatalog has no cross-match. Build via "
                "WeightedCatalog.from_oneuid(index, database)."
            )
        self._weighted_long = self._build_long_table()

    def __repr__(self) -> str:
        n = ", ".join(f"{k}={len(v)}" for k, v in self.catalogs.items())
        return f"<WeightedCatalog surveys=[{n}]>"
