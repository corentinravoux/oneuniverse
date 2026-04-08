"""
oneuniverse.weight.catalog
~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`WeightedCatalog` — high-level facade for cross-survey weighting.

Workflow
--------
1. Build a :class:`WeightedCatalog` from a dict of survey DataFrames.
2. Register one or more :class:`Weight` per survey via :meth:`add_weight`.
3. Call :meth:`crossmatch` to assign ``universal_id`` to each (survey, row).
4. Call :meth:`combine` to merge concurrences with the chosen strategy.
5. Query a single object with :meth:`concurrences`.

This is the public entry point of :mod:`oneuniverse.weight`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.weight.base import Weight
from oneuniverse.weight.combine import CombinedMeasurements, combine_weights
from oneuniverse.weight.crossmatch import CrossMatchResult, cross_match_surveys

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

    # ── Weight registration ─────────────────────────────────────────────

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

    # ── Pipeline ────────────────────────────────────────────────────────

    def crossmatch(
        self,
        sky_tol_arcsec: float = 1.0,
        dz_tol: Optional[float] = 1e-3,
    ) -> CrossMatchResult:
        """Run the sky+z cross-match across all registered surveys."""
        self._match = cross_match_surveys(
            self.catalogs,
            sky_tol_arcsec=sky_tol_arcsec,
            dz_tol=dz_tol,
        )
        # Pre-build the long table augmented with the user's weights and the
        # original observable columns, so combine() can be called many times
        # without re-running the match.
        self._weighted_long = self._build_long_table()
        return self._match

    def _build_long_table(self) -> pd.DataFrame:
        """Long-format table: one row per (universal_id, survey, original-row)."""
        if self._match is None:
            raise RuntimeError("Call .crossmatch() first")
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

        See :func:`oneuniverse.weight.combine.combine_weights` for the
        strategy semantics.
        """
        if self._weighted_long is None:
            self.crossmatch()
        return combine_weights(
            self._weighted_long,
            value_col=value_col,
            variance_col=variance_col,
            strategy=strategy,
            survey_alpha=survey_alpha,
        )

    # ── Query API ───────────────────────────────────────────────────────

    def concurrences(self, universal_id: int) -> pd.DataFrame:
        """Return all rows (one per survey) for a given universal object,
        with the per-survey weight already applied as a ``weight`` column."""
        if self._weighted_long is None:
            self.crossmatch()
        return self._weighted_long[
            self._weighted_long["universal_id"] == universal_id
        ].reset_index(drop=True)

    def n_universal(self) -> int:
        if self._match is None:
            self.crossmatch()
        return self._match.n_groups

    def n_multi_survey(self) -> int:
        if self._match is None:
            self.crossmatch()
        return self._match.n_multi

    def __repr__(self) -> str:
        n = ", ".join(f"{k}={len(v)}" for k, v in self.catalogs.items())
        return f"<WeightedCatalog surveys=[{n}]>"
