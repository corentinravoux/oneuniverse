"""
oneuniverse.weight.crossmatch  (legacy shim — Phase 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Kept for API stability. The canonical implementation now lives at
:mod:`oneuniverse.data.oneuid_crossmatch`. Full removal in Phase 6.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from oneuniverse.data.oneuid_crossmatch import (  # noqa: F401
    CrossMatchResult,
    cross_match_surveys as _cross_match_v2,
)
from oneuniverse.data.oneuid_rules import CrossMatchRules


def cross_match_surveys(
    catalogs: Dict[str, pd.DataFrame],
    sky_tol_arcsec: float = 1.0,
    dz_tol: Optional[float] = 1e-3,
    ra_col: str = "ra",
    dec_col: str = "dec",
    z_col: str = "z",
) -> CrossMatchResult:
    """Legacy wrapper. Prefer :func:`oneuniverse.data.oneuid_crossmatch.cross_match_surveys`."""
    rules = CrossMatchRules(sky_tol_arcsec=sky_tol_arcsec, dz_tol_default=dz_tol)
    # Column-rename shim for callers using custom column names.
    renamed: Dict[str, pd.DataFrame] = {}
    rename_map = {}
    if ra_col != "ra":
        rename_map[ra_col] = "ra"
    if dec_col != "dec":
        rename_map[dec_col] = "dec"
    if z_col != "z":
        rename_map[z_col] = "z"
    for name, df in catalogs.items():
        renamed[name] = df.rename(columns=rename_map) if rename_map else df
    return _cross_match_v2(renamed, rules)
