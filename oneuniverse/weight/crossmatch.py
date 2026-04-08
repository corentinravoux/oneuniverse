"""
oneuniverse.weight.crossmatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cross-match astrophysical objects across multiple survey catalogs.

The matcher takes ``{survey_name: DataFrame}`` and assigns a shared
``universal_id`` to objects that are consistent across catalogs — i.e. both
a sky-separation and a redshift-difference cut are satisfied.

It uses Astropy's ball-tree sky matching (``SkyCoord.search_around_sky``) for
an all-to-all neighbour list, then runs a simple union-find to form match
groups.  This scales to ~10⁶ objects per survey comfortably.

Notes
-----
The match is *transitive*: if A~B and B~C then A, B, C share the same
``universal_id`` (though not necessarily within tolerance of A~C).  For
heterogeneous surveys this is usually the desired behaviour — a tight-z
survey linked to a loose-z survey through an intermediate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────


@dataclass
class CrossMatchResult:
    """Output of :func:`cross_match_surveys`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per (survey, object) pair with a ``universal_id`` column
        plus ``survey``, ``row_index``, ``ra``, ``dec``, ``z``.
    n_groups : int
        Number of distinct universal objects.
    n_multi : int
        Number of universal objects seen in more than one survey.
    """
    table: pd.DataFrame
    n_groups: int
    n_multi: int

    def group(self, universal_id: int) -> pd.DataFrame:
        """Return all concurrences for one universal object."""
        return self.table[self.table["universal_id"] == universal_id]

    def multi_survey(self) -> pd.DataFrame:
        """Rows belonging to objects seen by at least two surveys."""
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
    sky_tol_arcsec: float = 1.0,
    dz_tol: Optional[float] = 1e-3,
    ra_col: str = "ra",
    dec_col: str = "dec",
    z_col: str = "z",
) -> CrossMatchResult:
    """Cross-match objects across surveys by sky position and redshift.

    Parameters
    ----------
    catalogs : dict[str, DataFrame]
        Mapping ``survey_name -> catalog``.  Each catalog must contain
        columns ``ra`` and ``dec`` in degrees; ``z`` is optional but
        required if ``dz_tol`` is not None.
    sky_tol_arcsec : float
        Maximum angular separation in arcseconds (default 1.0″).
    dz_tol : float or None
        Maximum Δz between linked objects.  ``None`` disables the cut.
    ra_col, dec_col, z_col : str
        Override column names.

    Returns
    -------
    CrossMatchResult
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if not catalogs:
        return CrossMatchResult(
            pd.DataFrame(columns=["universal_id", "survey", "row_index", "ra", "dec", "z"]),
            0, 0,
        )

    # Flatten into a single stacked table indexed 0..N-1 for the DSU.
    stacked: List[pd.DataFrame] = []
    offset = 0
    offsets: Dict[str, int] = {}
    for name, df in catalogs.items():
        for col in (ra_col, dec_col):
            if col not in df.columns:
                raise KeyError(f"Survey '{name}': missing column '{col}'")
        part = pd.DataFrame({
            "universal_id": np.zeros(len(df), dtype=np.int64),  # filled below
            "survey": name,
            "row_index": np.arange(len(df), dtype=np.int64),
            "ra": df[ra_col].to_numpy(dtype=np.float64),
            "dec": df[dec_col].to_numpy(dtype=np.float64),
            "z": (
                df[z_col].to_numpy(dtype=np.float64)
                if z_col in df.columns
                else np.full(len(df), np.nan)
            ),
        })
        stacked.append(part)
        offsets[name] = offset
        offset += len(df)

    table = pd.concat(stacked, ignore_index=True)
    n_total = len(table)

    # Global sky match — all-to-all neighbours within tolerance.
    coords = SkyCoord(
        ra=table["ra"].to_numpy() * u.deg,
        dec=table["dec"].to_numpy() * u.deg,
        frame="icrs",
    )
    sep_limit = sky_tol_arcsec * u.arcsec
    idx1, idx2, _sep, _ = coords.search_around_sky(coords, sep_limit)

    # Drop self-matches and the i>j duplicates.
    mask = idx1 < idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # Apply Δz cut
    if dz_tol is not None:
        z = table["z"].to_numpy()
        dz = np.abs(z[idx1] - z[idx2])
        # If either z is NaN, fall back to sky match only.
        nan_mask = np.isnan(dz)
        good = nan_mask | (dz <= dz_tol)
        idx1 = idx1[good]
        idx2 = idx2[good]

    # Union pairs from *different surveys only* — same-survey duplicates are
    # usually real distinct objects (close pairs); leaving them alone.
    surveys_arr = table["survey"].to_numpy()
    diff_survey = surveys_arr[idx1] != surveys_arr[idx2]
    idx1 = idx1[diff_survey]
    idx2 = idx2[diff_survey]

    # Group merging via scipy's connected_components on a sparse adjacency.
    # Strict O((N + E) α(N)) and fully vectorised — much faster than a
    # Python-level union-find for N ~ 10⁶ rows.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if idx1.size:
        data = np.ones(idx1.size, dtype=np.uint8)
        adj = coo_matrix(
            (data, (idx1, idx2)),
            shape=(n_total, n_total),
        ).tocsr()
        _, labels = connected_components(adj, directed=False)
    else:
        labels = np.arange(n_total, dtype=np.int64)

    # Compact label range to 0..n_groups-1.
    _, inv = np.unique(labels, return_inverse=True)
    table["universal_id"] = inv.astype(np.int64)

    n_groups = int(inv.max()) + 1 if n_total else 0
    multi = (
        table.groupby("universal_id")["survey"].nunique() > 1
    ).sum()

    logger.info(
        "cross_match: %d rows, %d unique objects, %d multi-survey",
        n_total, n_groups, int(multi),
    )
    return CrossMatchResult(table=table, n_groups=n_groups, n_multi=int(multi))
