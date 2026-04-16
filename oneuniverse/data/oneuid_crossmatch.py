"""
oneuniverse.data.oneuid_crossmatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cross-match surveys driven by a :class:`CrossMatchRules` policy.

This is the *canonical* home for the cross-matcher — Phase 4. The legacy
entry point lives in :mod:`oneuniverse.weight.crossmatch` as a thin shim
during the transition; Phase 6 deletes it.

The matcher is z-type aware: per-row ``z_type`` drives both the Δz
tolerance (via :meth:`CrossMatchRules.dz_tol_for`) and the reject list
(via :meth:`CrossMatchRules.accepts`). Same-survey links are never made
— same-survey duplicates are typically real close pairs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from oneuniverse.data.oneuid_rules import CrossMatchRules

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────


@dataclass
class CrossMatchResult:
    """Output of :func:`cross_match_surveys`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per (survey, object) pair with ``universal_id`` (int),
        ``survey`` (str), ``row_index`` (int), ``ra``, ``dec``, ``z``,
        and ``z_type``.
    n_groups : int
        Number of distinct universal objects.
    n_multi : int
        Number of universal objects seen in more than one survey.
    """
    table: pd.DataFrame
    n_groups: int
    n_multi: int

    def group(self, universal_id: int) -> pd.DataFrame:
        return self.table[self.table["universal_id"] == universal_id]

    def multi_survey(self) -> pd.DataFrame:
        counts = self.table.groupby("universal_id")["survey"].nunique()
        keep = counts[counts > 1].index
        return self.table[self.table["universal_id"].isin(keep)]

    def __repr__(self) -> str:
        return (
            f"<CrossMatchResult n_rows={len(self.table)} "
            f"n_groups={self.n_groups} n_multi={self.n_multi}>"
        )


def cross_match_surveys(
    catalogs: Dict[str, pd.DataFrame],
    rules: CrossMatchRules,
    *,
    survey_ztype: Optional[Mapping[str, str]] = None,
) -> CrossMatchResult:
    """Cross-match objects across surveys using a :class:`CrossMatchRules`.

    Parameters
    ----------
    catalogs
        Mapping ``survey_name -> catalog``. Each catalog must have
        ``ra`` and ``dec`` columns in degrees. ``z`` is required iff
        ``rules.dz_tol_default`` or any pair override is not ``None``.
        Per-row ``z_type`` column is preferred; otherwise
        ``survey_ztype[name]`` is used as a fallback; otherwise
        ``"none"``.
    rules
        The cross-match policy.
    survey_ztype
        Optional per-survey z-type fallback when rows lack ``z_type``.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if not catalogs:
        return CrossMatchResult(
            _empty_table(), 0, 0,
        )

    # Flatten into a single stacked table.
    stacked: List[pd.DataFrame] = []
    for name, df in catalogs.items():
        for col in ("ra", "dec"):
            if col not in df.columns:
                raise KeyError(f"Survey '{name}': missing column '{col}'")
        n = len(df)
        ztype_col = _resolve_ztype(df, name, survey_ztype)
        part = pd.DataFrame({
            "universal_id": np.zeros(n, dtype=np.int64),
            "survey": name,
            "row_index": np.arange(n, dtype=np.int64),
            "ra": df["ra"].to_numpy(dtype=np.float64),
            "dec": df["dec"].to_numpy(dtype=np.float64),
            "z": (
                df["z"].to_numpy(dtype=np.float64)
                if "z" in df.columns
                else np.full(n, np.nan)
            ),
            "z_type": ztype_col,
        })
        stacked.append(part)

    table = pd.concat(stacked, ignore_index=True)
    n_total = len(table)

    # Sky neighbour list.
    coords = SkyCoord(
        ra=table["ra"].to_numpy() * u.deg,
        dec=table["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    idx1, idx2, _sep, _ = coords.search_around_sky(
        coords, rules.sky_tol_arcsec * u.arcsec,
    )
    mask = idx1 < idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Same-survey links never contribute to universal IDs.
    surveys_arr = table["survey"].to_numpy()
    diff_survey = surveys_arr[idx1] != surveys_arr[idx2]
    idx1 = idx1[diff_survey]
    idx2 = idx2[diff_survey]

    # Rule-driven filtering (per-pair Δz tolerance + reject list).
    if idx1.size:
        z = table["z"].to_numpy()
        ztypes = table["z_type"].to_numpy()
        keep = np.ones(idx1.size, dtype=bool)
        for i in range(idx1.size):
            za, zb = ztypes[idx1[i]], ztypes[idx2[i]]
            if not rules.accepts(za, zb):
                keep[i] = False
                continue
            dz_tol = rules.dz_tol_for(za, zb)
            if dz_tol is None:
                continue
            dz = abs(z[idx1[i]] - z[idx2[i]])
            if np.isnan(dz):  # missing z: keep sky-only link
                continue
            if dz > dz_tol:
                keep[i] = False
        idx1 = idx1[keep]
        idx2 = idx2[keep]

    # Connected-components labelling.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if idx1.size:
        data = np.ones(idx1.size, dtype=np.uint8)
        adj = coo_matrix(
            (data, (idx1, idx2)), shape=(n_total, n_total),
        ).tocsr()
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(n_total, dtype=np.int64)

    _, inv = np.unique(labels, return_inverse=True)
    table["universal_id"] = inv.astype(np.int64)
    n_groups = int(inv.max()) + 1 if n_total else 0
    n_multi = int(
        (table.groupby("universal_id")["survey"].nunique() > 1).sum()
    )

    logger.info(
        "cross_match: %d rows, %d unique, %d multi-survey (rules=%s)",
        n_total, n_groups, n_multi, rules.hash(),
    )
    return CrossMatchResult(
        table=table, n_groups=n_groups, n_multi=n_multi,
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "universal_id", "survey", "row_index", "ra", "dec", "z", "z_type",
    ])


def _resolve_ztype(
    df: pd.DataFrame,
    survey_name: str,
    survey_ztype: Optional[Mapping[str, str]],
) -> np.ndarray:
    """Pick the z_type column: per-row → per-survey fallback → ``'none'``."""
    n = len(df)
    if "z_type" in df.columns:
        return df["z_type"].astype(str).to_numpy()
    fallback = "none"
    if survey_ztype is not None and survey_name in survey_ztype:
        fallback = str(survey_ztype[survey_name])
    return np.full(n, fallback, dtype=object)
