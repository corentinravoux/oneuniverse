"""
oneuniverse.data.oneuid
~~~~~~~~~~~~~~~~~~~~~~~
**ONEUID** — the unified object identifier of the oneuniverse database.

A ONEUID is an integer that labels a single astrophysical object across all
surveys.  Two catalog rows share the same ONEUID iff they pass the cross-match
tolerance (sky separation + Δz).  Singletons get their own ONEUID.

The index is *materialised* to a single sidecar file at the database root::

    {database_root}/_oneuid_index.parquet

Schema (one row per (dataset, original-row) pair):

    oneuid       int64    universal object id, contiguous from 0
    dataset      string   dataset name as registered in OneuniverseDatabase
    row_index    int64    row index in the converted Parquet partition stack
    ra           float64  degrees, ICRS
    dec          float64  degrees, ICRS
    z            float64  redshift used for the Δz cut (NaN if not available)

Loading is constant-time and the index is keyed on ``oneuid`` so any query
"give me all surveys that observed object N" is a single boolean mask.

The cross-match itself is the optimised path:

- Loads only ``ra, dec, z`` from each dataset (Parquet column pushdown).
- Vectorised ball-tree neighbour search via Astropy's ``search_around_sky``.
- Vectorised connected-components via scipy.sparse instead of a Python loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from oneuniverse.data.converter import read_oneuniverse_parquet
from oneuniverse.data.selection import (
    Cone,
    Selection,
    Shell,
    SkyPatch,
    apply_selections,
)

logger = logging.getLogger(__name__)


ONEUID_INDEX_FILENAME = "_oneuid_index.parquet"


# ── Data class ───────────────────────────────────────────────────────────


@dataclass
class OneuidIndex:
    """In-memory representation of the ONEUID index.

    Attributes
    ----------
    table : DataFrame
        Long-format index, see module docstring.
    n_unique : int
        Number of distinct ONEUIDs.
    n_multi : int
        Number of ONEUIDs observed in ≥ 2 datasets.
    sky_tol_arcsec, dz_tol : float
        Tolerances used to build it.
    """
    table: pd.DataFrame
    n_unique: int
    n_multi: int
    sky_tol_arcsec: float
    dz_tol: Optional[float]

    def datasets(self) -> List[str]:
        return sorted(self.table["dataset"].unique().tolist())

    def of(self, oneuid: int) -> pd.DataFrame:
        """All concurrences for one universal object."""
        return self.table[self.table["oneuid"] == oneuid].reset_index(drop=True)

    def lookup(self, dataset: str, row_index: int) -> int:
        """Return the ONEUID for a (dataset, row) pair."""
        m = (self.table["dataset"] == dataset) & (self.table["row_index"] == row_index)
        if not m.any():
            raise KeyError(f"({dataset!r}, row_index={row_index}) not in index")
        return int(self.table.loc[m, "oneuid"].iloc[0])

    def multi_only(self) -> pd.DataFrame:
        """Subset of the index for objects seen by ≥ 2 datasets."""
        counts = self.table.groupby("oneuid")["dataset"].nunique()
        keep = counts[counts > 1].index
        return self.table[self.table["oneuid"].isin(keep)].reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"<OneuidIndex n_rows={len(self.table)} "
            f"n_unique={self.n_unique} n_multi={self.n_multi}>"
        )


# ── Build / load / persist ───────────────────────────────────────────────


def build_oneuid_index(
    database,
    sky_tol_arcsec: float = 1.0,
    dz_tol: Optional[float] = 1e-3,
    persist: bool = True,
) -> OneuidIndex:
    """Build the ONEUID index for an :class:`OneuniverseDatabase`.

    Loads only the cross-match columns (``ra, dec, z``) from each dataset's
    Parquet stack, runs the optimised cross-match, assigns ONEUIDs, and
    optionally writes the result next to the database root.

    Parameters
    ----------
    database : OneuniverseDatabase
    sky_tol_arcsec : float
        Sky separation cut (default 1.0″).
    dz_tol : float or None
        Redshift cut.  ``None`` disables it.
    persist : bool
        If True, write ``{root}/_oneuid_index.parquet``.

    Returns
    -------
    OneuidIndex
    """
    from oneuniverse.weight.crossmatch import cross_match_surveys

    catalogs: Dict[str, pd.DataFrame] = {}
    for name in database:
        path = database.get_path(name)
        # Pushdown read: only the columns we actually need.
        try:
            df = read_oneuniverse_parquet(path, columns=["ra", "dec", "z"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONEUID: skipping %s — cannot read ra/dec/z (%s)", name, exc)
            continue
        catalogs[name] = df
        logger.info("ONEUID: loaded %s (%d rows)", name, len(df))

    if not catalogs:
        raise RuntimeError("ONEUID: no datasets in database have ra/dec/z columns")

    res = cross_match_surveys(
        catalogs,
        sky_tol_arcsec=sky_tol_arcsec,
        dz_tol=dz_tol,
    )

    table = res.table.rename(columns={
        "universal_id": "oneuid",
        "survey": "dataset",
    })[["oneuid", "dataset", "row_index", "ra", "dec", "z"]]
    table = table.sort_values(["oneuid", "dataset"]).reset_index(drop=True)

    index = OneuidIndex(
        table=table,
        n_unique=res.n_groups,
        n_multi=res.n_multi,
        sky_tol_arcsec=sky_tol_arcsec,
        dz_tol=dz_tol,
    )

    if persist:
        out = database.root / ONEUID_INDEX_FILENAME
        _write_index(index, out)
        logger.info("ONEUID: wrote %s", out)

    return index


def load_oneuid_index(database) -> OneuidIndex:
    """Load a previously persisted ONEUID index."""
    path = database.root / ONEUID_INDEX_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No ONEUID index at {path}. Run database.build_oneuid()."
        )
    table, meta = _read_index(path)
    n_multi = int(
        (table.groupby("oneuid")["dataset"].nunique() > 1).sum()
    )
    return OneuidIndex(
        table=table,
        n_unique=int(table["oneuid"].max()) + 1 if len(table) else 0,
        n_multi=n_multi,
        sky_tol_arcsec=meta.get("sky_tol_arcsec", float("nan")),
        dz_tol=meta.get("dz_tol"),
    )


def _write_index(index: OneuidIndex, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(index.table, preserve_index=False)
    metadata = {
        b"oneuniverse.sky_tol_arcsec": str(index.sky_tol_arcsec).encode(),
        b"oneuniverse.dz_tol": (
            "" if index.dz_tol is None else str(index.dz_tol)
        ).encode(),
        b"oneuniverse.n_unique": str(index.n_unique).encode(),
        b"oneuniverse.n_multi": str(index.n_multi).encode(),
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
    pq.write_table(table, path, compression="zstd")


def _read_index(path: Path):
    import pyarrow.parquet as pq

    pq_table = pq.read_table(path)
    df = pq_table.to_pandas()
    md = pq_table.schema.metadata or {}
    meta = {}
    if b"oneuniverse.sky_tol_arcsec" in md:
        meta["sky_tol_arcsec"] = float(md[b"oneuniverse.sky_tol_arcsec"])
    if b"oneuniverse.dz_tol" in md:
        v = md[b"oneuniverse.dz_tol"].decode()
        meta["dz_tol"] = float(v) if v else None
    return df, meta


# ── Optimised cross-survey loader ────────────────────────────────────────


def load_universal(
    database,
    columns: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    only_multi: bool = False,
    index: Optional[OneuidIndex] = None,
) -> pd.DataFrame:
    """Load and stack rows from selected datasets, with the ``oneuid`` column.

    Parameters
    ----------
    database : OneuniverseDatabase
    columns : list[str] or None
        Columns to read from each dataset (besides the join keys).  ``None``
        loads everything — use sparingly.
    datasets : list[str] or None
        Restrict to a subset of dataset names.  ``None`` = all in the index.
    only_multi : bool
        If True, keep only ONEUIDs observed by ≥ 2 datasets.
    index : OneuidIndex or None
        Pre-loaded index.  If None, attempt to load from disk.

    Returns
    -------
    DataFrame
        Long format: one row per (oneuid, dataset, original-row), with the
        requested columns plus ``oneuid`` and ``dataset``.

    Notes
    -----
    Optimised path: each dataset is read once with a column pushdown, then
    sliced by the ONEUID index's ``row_index`` array — no per-row Python
    loops, no repeated I/O.
    """
    if index is None:
        index = load_oneuid_index(database)

    table = index.multi_only() if only_multi else index.table
    if datasets is not None:
        table = table[table["dataset"].isin(datasets)]

    pieces: List[pd.DataFrame] = []
    for ds, sub in table.groupby("dataset", sort=False):
        path = database.get_path(ds)
        # Intersect requested columns with those in this dataset.
        ds_columns = columns
        if columns is not None:
            from oneuniverse.data.converter import get_manifest
            manifest = get_manifest(path)
            available = {c.name for c in manifest.schema}
            ds_columns = [c for c in columns if c in available]
        df = read_oneuniverse_parquet(path, columns=ds_columns or None)
        # Vectorised slice via row_index — single allocation, no loop.
        sliced = df.iloc[sub["row_index"].to_numpy()].reset_index(drop=True)
        # Fill missing columns with NaN.
        if columns is not None:
            for c in columns:
                if c not in sliced.columns:
                    sliced[c] = np.nan
        sliced.insert(0, "oneuid", sub["oneuid"].to_numpy())
        sliced.insert(1, "dataset", ds)
        pieces.append(sliced)

    if not pieces:
        cols = ["oneuid", "dataset"] + (columns or [])
        return pd.DataFrame(columns=cols)
    return pd.concat(pieces, ignore_index=True)


# ── Tiered query API ─────────────────────────────────────────────────────


class OneuidQuery:
    """Tiered, selector-based query API over the ONEUID index.

    The class implements two orthogonal axes:

    1. **Selectors** — choose which ONEUIDs you want.  Cheap because they
       only touch the in-memory index (which already carries ``ra/dec/z``):

       - :meth:`from_id` / :meth:`from_ids` — direct lookup
       - :meth:`from_foreign_ids` — by ``row_index`` in another dataset
         (the closest analog to "give me ONEUID for galaxy_id=… in DESI")
       - :meth:`from_cone` — angular cone on the sky
       - :meth:`from_shell` — redshift range
       - :meth:`from_skypatch` — RA/Dec rectangle
       - :meth:`from_selection` — any :class:`Selection` (or list of them)

       Selectors return a 1-D ``ndarray[int64]`` of unique ONEUIDs.

    2. **Hydration levels** — choose how much you want to load.  Each level
       reads strictly more data than the last:

       - :meth:`index_for(uids)` — *level 0*: just the index slice
         ``(oneuid, dataset, row_index, ra, dec, z)``.  Zero dataset I/O.
       - :meth:`partial_for(uids, columns=…)` — *level 1*: index + a small
         column subset from each dataset (Parquet column pushdown).
       - :meth:`full_for(uids)` — *level 2*: the entire row from every
         occurrence in every dataset.

    Selectors can be chained with hydrators:

    >>> q = db.oneuid_query()
    >>> uids = q.from_cone(ra=185, dec=15, radius=5)
    >>> df_light  = q.index_for(uids)
    >>> df_medium = q.partial_for(uids, columns=["z_spec_err", "psf_g"])
    >>> df_full   = q.full_for(uids)
    """

    def __init__(self, database, index: Optional[OneuidIndex] = None) -> None:
        self.database = database
        self.index: OneuidIndex = index if index is not None else load_oneuid_index(database)
        # Pre-extract numpy views for fast vectorised filters.
        self._uid_arr = self.index.table["oneuid"].to_numpy()
        self._ds_arr = self.index.table["dataset"].to_numpy()
        self._row_arr = self.index.table["row_index"].to_numpy()
        self._ra = self.index.table["ra"].to_numpy()
        self._dec = self.index.table["dec"].to_numpy()
        self._z = self.index.table["z"].to_numpy()

    # ── Selectors ────────────────────────────────────────────────────────

    def from_id(self, oneuid: int) -> np.ndarray:
        """Return ``np.array([oneuid])`` after validating it exists."""
        if not (self._uid_arr == int(oneuid)).any():
            raise KeyError(f"ONEUID {oneuid} not in index")
        return np.array([int(oneuid)], dtype=np.int64)

    def from_ids(self, oneuids: Iterable[int]) -> np.ndarray:
        """Return the unique, sorted intersection of *oneuids* with the index."""
        uids = np.unique(np.asarray(list(oneuids), dtype=np.int64))
        present = np.isin(uids, self._uid_arr)
        return uids[present]

    def from_foreign_ids(
        self,
        dataset: str,
        row_indices: Iterable[int],
    ) -> np.ndarray:
        """Map a list of ``row_index`` values *in dataset* to ONEUIDs.

        This is the bridge from a per-survey identifier (e.g. ``galaxy_id``,
        which corresponds to ``row_index`` in the converted Parquet) to the
        universal ONEUID.
        """
        rows = np.asarray(list(row_indices), dtype=np.int64)
        mask = (self._ds_arr == dataset) & np.isin(self._row_arr, rows)
        uids = np.unique(self._uid_arr[mask])
        return uids.astype(np.int64)

    def from_cone(
        self,
        ra: float,
        dec: float,
        radius: float,
    ) -> np.ndarray:
        """All ONEUIDs whose *any* occurrence sits inside a sky cone (degrees)."""
        return self.from_selection(Cone(ra=ra, dec=dec, radius=radius))

    def from_shell(self, z_min: float, z_max: float) -> np.ndarray:
        return self.from_selection(Shell(z_min=z_min, z_max=z_max))

    def from_skypatch(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
    ) -> np.ndarray:
        return self.from_selection(
            SkyPatch(ra_min=ra_min, ra_max=ra_max, dec_min=dec_min, dec_max=dec_max)
        )

    def from_selection(
        self,
        selection: Union[Selection, Sequence[Selection]],
    ) -> np.ndarray:
        """Apply any :class:`Selection` (or list, AND-combined) to the index.

        Coordinates come from the index itself, so this never reads a
        dataset Parquet.  An object is kept if *any* of its concurrences
        passes — useful when one survey is much more complete in z.
        """
        mask = apply_selections(self._ra, self._dec, self._z, selection)
        uids = np.unique(self._uid_arr[mask])
        return uids.astype(np.int64)

    # ── Hydration levels ─────────────────────────────────────────────────

    def index_for(self, oneuids: Iterable[int]) -> pd.DataFrame:
        """Level 0: the raw index slice.  Zero dataset I/O.

        Returns one row per (oneuid, dataset, row_index) with the columns
        already in the index: ``ra, dec, z``.
        """
        uids = np.asarray(list(oneuids), dtype=np.int64)
        mask = np.isin(self._uid_arr, uids)
        return self.index.table[mask].reset_index(drop=True)

    def partial_for(
        self,
        oneuids: Iterable[int],
        columns: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Level 1: index + a column subset, with Parquet column pushdown.

        For each dataset that has at least one of the requested ONEUIDs,
        the converted Parquet stack is opened *once* with only ``columns``
        loaded, then sliced by ``row_index``.  ``columns=None`` is the level-2
        path: load every column.
        """
        uids = np.asarray(list(oneuids), dtype=np.int64)
        sub = self.index.table[np.isin(self._uid_arr, uids)]
        if datasets is not None:
            sub = sub[sub["dataset"].isin(datasets)]
        if sub.empty:
            extra_cols = list(columns) if columns is not None else []
            return pd.DataFrame(
                columns=["oneuid", "dataset", "row_index", "ra", "dec", "z", *extra_cols]
            )

        col_arg = list(columns) if columns is not None else None
        pieces: List[pd.DataFrame] = []
        for ds, grp in sub.groupby("dataset", sort=False):
            path = self.database.get_path(ds)
            # Intersect requested columns with those available in this
            # dataset so that missing survey-specific columns are filled
            # with NaN rather than raising an error.
            ds_col_arg = col_arg
            if col_arg is not None:
                manifest = self.database.get_manifest(ds)
                available = {c.name for c in manifest.schema}
                ds_col_arg = [c for c in col_arg if c in available]
            df = read_oneuniverse_parquet(path, columns=ds_col_arg or None)
            sliced = df.iloc[grp["row_index"].to_numpy()].reset_index(drop=True)
            # Drop any column that would collide with the index columns we
            # are about to prepend (ra/dec/z come from the dataset itself).
            sliced = sliced.drop(
                columns=[c for c in ("ra", "dec", "z") if c in sliced.columns]
            )
            # Add NaN columns for any requested column absent in this dataset.
            if col_arg is not None:
                for c in col_arg:
                    if c not in sliced.columns and c not in ("ra", "dec", "z"):
                        sliced[c] = np.nan
            sliced.insert(0, "oneuid", grp["oneuid"].to_numpy())
            sliced.insert(1, "dataset", ds)
            sliced.insert(2, "row_index", grp["row_index"].to_numpy())
            sliced.insert(3, "ra", grp["ra"].to_numpy())
            sliced.insert(4, "dec", grp["dec"].to_numpy())
            sliced.insert(5, "z", grp["z"].to_numpy())
            pieces.append(sliced)

        return pd.concat(pieces, ignore_index=True)

    def full_for(
        self,
        oneuids: Iterable[int],
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Level 2: every column from every concurrence.

        Most expensive — equivalent to ``partial_for`` with ``columns=None``.
        """
        return self.partial_for(oneuids, columns=None, datasets=datasets)

    def concurrences(self, oneuid: int) -> pd.DataFrame:
        """Index slice for one ONEUID — pretty alias for ``index_for([uid])``."""
        return self.index_for([oneuid])

    def __repr__(self) -> str:
        return (
            f"<OneuidQuery n_unique={self.index.n_unique} "
            f"n_multi={self.index.n_multi} datasets={self.index.datasets()}>"
        )
