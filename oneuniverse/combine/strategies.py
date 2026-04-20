"""
oneuniverse.combine.strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Combine weights for a single universal object measured by several surveys.

Strategies (research summary in ``docs/weight_normalization_research.md``):

- ``"best_only"`` (default) — keep the lowest-variance measurement, discard
  the rest.  Matches DESI / eBOSS practice and never breaks the
  inverse-variance interpretation downstream codes (``flip``) rely on.
- ``"ivar_average"`` — BLUE inverse-variance weighted mean of the measurements
  on a *common* observable.  Valid if errors are independent.
- ``"hyperparameter"`` — Lahav (2000) / Hobson, Bridle & Lahav (2002)
  hyperparameters: each survey gets a multiplicative ``α_s`` (default 1.0)
  applied as ``w_{s,i} → α_s · w_{s,i}``.
- ``"unit_mean"`` — rescale each survey's weights by their per-survey mean.
  Used in clustering pipelines (``pypower``); **destroys the 1/σ² scale** so
  it must NOT be fed to a Gaussian likelihood.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.combine.measurements import CombinedMeasurements

logger = logging.getLogger(__name__)


_STRATEGIES = ("best_only", "ivar_average", "hyperparameter", "unit_mean")


def combine_weights(
    matched: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
    survey_col: str = "survey",
    universal_col: str = "universal_id",
    strategy: str = "best_only",
    survey_alpha: Optional[Dict[str, float]] = None,
) -> CombinedMeasurements:
    """Combine per-(object,survey) measurements into one entry per universal object.

    Parameters
    ----------
    matched : DataFrame
        Long-format table with one row per (universal_id, survey) entry.
        Must contain at minimum ``universal_col``, ``survey_col``,
        ``value_col``, ``variance_col``.
    value_col : str
        Name of the homogenised observable (e.g. ``"v_pec"``, ``"z"``,
        ``"mu"``).  All surveys must already be on the *same* scale.
    variance_col : str
        Per-row variance ``σ²`` of the observable.
    survey_col, universal_col : str
        Column overrides.
    strategy : {"best_only", "ivar_average", "hyperparameter", "unit_mean"}
    survey_alpha : dict, optional
        Per-survey multiplicative weight (Lahav hyperparameters).  Only used
        by ``"hyperparameter"``.  Missing surveys default to ``1.0``.

    Returns
    -------
    CombinedMeasurements
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Pick one of {_STRATEGIES}."
        )
    for col in (universal_col, survey_col, value_col, variance_col):
        if col not in matched.columns:
            raise KeyError(f"combine_weights: missing column '{col}'")

    if strategy == "unit_mean":
        logger.warning(
            "combine_weights: 'unit_mean' destroys the inverse-variance scale; "
            "do NOT feed the result to a Gaussian likelihood (e.g. flip)."
        )

    df = matched[[universal_col, survey_col, value_col, variance_col]].copy()
    df = df.rename(columns={
        universal_col: "universal_id",
        survey_col: "survey",
        value_col: "value",
        variance_col: "variance",
    })
    good = np.isfinite(df["variance"]) & (df["variance"] > 0) & np.isfinite(df["value"])
    n_dropped = int((~good).sum())
    if n_dropped:
        logger.info("combine_weights: dropping %d rows with bad variance/value", n_dropped)
    df = df[good].reset_index(drop=True)

    df["weight"] = 1.0 / df["variance"]

    if strategy == "hyperparameter" and survey_alpha:
        alpha = df["survey"].map(lambda s: float(survey_alpha.get(s, 1.0)))
        df["weight"] = df["weight"] * alpha.to_numpy()

    if strategy == "unit_mean":
        means = df.groupby("survey")["weight"].transform("mean")
        df["weight"] = df["weight"] / means.replace(0, np.nan)

    out_rows: List[dict] = []
    for uid, grp in df.groupby("universal_id", sort=False):
        surveys = sorted(grp["survey"].unique())
        if strategy == "best_only":
            i = int(grp["variance"].idxmin())
            row = grp.loc[i]
            out_rows.append({
                "universal_id": uid,
                "value": float(row["value"]),
                "variance": float(row["variance"]),
                "weight": float(row["weight"]),
                "n_surveys": len(surveys),
                "surveys": ",".join(surveys),
            })
        else:
            w = grp["weight"].to_numpy()
            v = grp["value"].to_numpy()
            wsum = w.sum()
            if wsum <= 0:
                continue
            value = float((w * v).sum() / wsum)
            if strategy == "ivar_average" or strategy == "hyperparameter":
                variance = float(1.0 / wsum)
            else:
                variance = float(np.nan)
            out_rows.append({
                "universal_id": uid,
                "value": value,
                "variance": variance,
                "weight": float(wsum),
                "n_surveys": len(surveys),
                "surveys": ",".join(surveys),
            })

    out = pd.DataFrame(out_rows)
    return CombinedMeasurements(table=out, strategy=strategy)
