"""
oneuniverse.data.oneuid
~~~~~~~~~~~~~~~~~~~~~~~
**ONEUID** — the unified object identifier of the oneuniverse database.

A ONEUID labels a single astrophysical object across all surveys. Two
catalog rows share the same ONEUID iff they pass the cross-match rules
(sky separation + optional Δz + rule-based reject list). Singletons get
their own ONEUID.

Named, versioned indices
------------------------
Indices are persisted at::

    {database_root}/_oneuid/<name>.parquet
    {database_root}/_oneuid/<name>.manifest.json

so multiple indices (e.g. ``default``, ``spec_only``, ``desi_only``) can
coexist. The manifest JSON records the datasets used, the
:class:`CrossMatchRules` used (including its hash), and basic stats.

Schema (one row per (dataset, original-row) pair):

    oneuid       int64    universal id, contiguous from 0
    dataset      string   dataset name as registered in the database
    row_index    int64    row index in the converted Parquet stack
    ra           float64  degrees, ICRS
    dec          float64  degrees, ICRS
    z            float64  redshift used for the Δz cut (NaN if absent)
    z_type       string   per-row ztype (spec/phot/pv/none) — audit column
    survey_type  string   dataset's ``manifest.survey_type`` — audit column

Legacy layout
-------------
The single-file layout ``{root}/_oneuid_index.parquet`` remains readable
for back-compat. It has no audit columns; they come back as ``"none"`` /
``"unknown"``. Full removal is deferred to Phase 6.
"""
from __future__ import annotations

import datetime as dt_mod
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from oneuniverse.data.converter import read_oneuniverse_parquet
from oneuniverse.data.oneuid_crossmatch import cross_match_surveys
from oneuniverse.data.oneuid_rules import CrossMatchRules
from oneuniverse.data.selection import (
    Cone,
    Selection,
    Shell,
    SkyPatch,
    apply_selections,
)
from oneuniverse.data.validity import DatasetValidity

logger = logging.getLogger(__name__)


ONEUID_DIR = "_oneuid"
ONEUID_MANIFEST_FORMAT_VERSION = 1


# ── Path helpers ─────────────────────────────────────────────────────────


def _index_path(root: Path, name: str) -> Path:
    return root / ONEUID_DIR / f"{name}.parquet"


def _index_manifest_path(root: Path, name: str) -> Path:
    return root / ONEUID_DIR / f"{name}.manifest.json"


# ── Data class ───────────────────────────────────────────────────────────


@dataclass
class OneuidIndex:
    """In-memory representation of a ONEUID index.

    Attributes
    ----------
    table : DataFrame
        Long-format index — see module docstring.
    n_unique : int
        Number of distinct ONEUIDs.
    n_multi : int
        Number of ONEUIDs observed in ≥ 2 datasets.
    name : str
        Identifier of this index (``"default"``, ``"spec_only"``, …).
    rules : CrossMatchRules or None
        The cross-match policy used to build this index.
    """
    table: pd.DataFrame
    n_unique: int
    n_multi: int
    name: str = "default"
    rules: Optional[CrossMatchRules] = None
    validity: Optional[DatasetValidity] = None

    # ── Back-compat shims (derived from rules) ──────────────────────

    @property
    def sky_tol_arcsec(self) -> float:
        return self.rules.sky_tol_arcsec if self.rules is not None else float("nan")

    @property
    def dz_tol(self) -> Optional[float]:
        return self.rules.dz_tol_default if self.rules is not None else None

    # ── Lookup API ──────────────────────────────────────────────────

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

    # ── Dataset subsetting ──────────────────────────────────────────

    def restrict_to(self, datasets: Sequence[str]) -> "OneuidIndex":
        """Return a new index restricted to rows from *datasets*.

        ONEUIDs are re-indexed to a contiguous ``0..n_unique-1`` range.
        """
        sub = self.table[self.table["dataset"].isin(list(datasets))].copy()
        if sub.empty:
            empty = sub.reset_index(drop=True)
            return OneuidIndex(
                table=empty, n_unique=0, n_multi=0,
                name=f"{self.name}.restricted", rules=self.rules,
            )
        _, inv = np.unique(sub["oneuid"].to_numpy(), return_inverse=True)
        sub["oneuid"] = inv.astype(np.int64)
        counts = sub.groupby("oneuid")["dataset"].nunique()
        n_multi = int((counts > 1).sum())
        return OneuidIndex(
            table=sub.reset_index(drop=True),
            n_unique=int(inv.max()) + 1,
            n_multi=n_multi,
            name=f"{self.name}.restricted",
            rules=self.rules,
        )

    def __repr__(self) -> str:
        return (
            f"<OneuidIndex name={self.name!r} n_rows={len(self.table)} "
            f"n_unique={self.n_unique} n_multi={self.n_multi}>"
        )


# ── Build / load / persist ───────────────────────────────────────────────


def build_oneuid_index(
    database,
    *,
    datasets: Optional[Sequence[str]] = None,
    rules: Optional[CrossMatchRules] = None,
    name: str = "default",
    persist: bool = True,
    # Legacy kwargs (kept for back-compat — pre-Phase-4 callers)
    sky_tol_arcsec: Optional[float] = None,
    dz_tol: Optional[float] = None,
) -> OneuidIndex:
    """Build a ONEUID index for *database*.

    Parameters
    ----------
    database
        :class:`OneuniverseDatabase` instance.
    datasets
        Restrict the build to this subset of dataset names. ``None``
        means every dataset in *database*.
    rules
        Cross-match policy. If ``None``, either
        :class:`CrossMatchRules`\\ () defaults are used, or — for
        back-compat — the ``sky_tol_arcsec``/``dz_tol`` kwargs are
        translated into a policy.
    name
        Identifier for this index. Controls the on-disk layout.
    persist
        Write the index + manifest to ``{root}/_oneuid/<name>.*``.
    sky_tol_arcsec, dz_tol
        Back-compat only. Ignored when *rules* is provided.
    """
    rules = _resolve_rules(rules, sky_tol_arcsec, dz_tol)

    # Dataset selection.
    all_names = list(database)
    if datasets is None:
        selected = all_names
    else:
        missing = set(datasets) - set(all_names)
        if missing:
            raise KeyError(f"Unknown dataset(s): {sorted(missing)}")
        selected = list(datasets)

    # Load only the cross-match columns + audit columns from each dataset.
    catalogs: Dict[str, pd.DataFrame] = {}
    survey_ztype: Dict[str, str] = {}
    survey_type_map: Dict[str, str] = {}
    for ds in selected:
        path = database.get_path(ds)
        manifest = database.get_manifest(ds)
        survey_type_map[ds] = manifest.survey_type or "unknown"
        available = {c.name for c in manifest.schema}
        wanted = [c for c in ("ra", "dec", "z", "z_type") if c in available]
        try:
            df = read_oneuniverse_parquet(path, columns=wanted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONEUID: skipping %s — cannot read (%s)", ds, exc)
            continue
        catalogs[ds] = df
        logger.info("ONEUID: loaded %s (%d rows)", ds, len(df))

    if not catalogs:
        raise RuntimeError("ONEUID: no datasets in selection have ra/dec columns")

    res = cross_match_surveys(catalogs, rules, survey_ztype=survey_ztype or None)

    # Rename to the ONEUID schema + attach audit columns.
    table = res.table.rename(columns={
        "universal_id": "oneuid", "survey": "dataset",
    }).copy()
    table["survey_type"] = table["dataset"].map(survey_type_map).fillna("unknown")
    # Ensure audit columns exist even if cross_match omitted them.
    if "z_type" not in table.columns:
        table["z_type"] = "none"
    table = table[[
        "oneuid", "dataset", "row_index", "ra", "dec", "z",
        "z_type", "survey_type",
    ]]
    table = table.sort_values(["oneuid", "dataset"]).reset_index(drop=True)

    # Phase 6: compact dtypes. `dataset` and `survey_type` are low-cardinality
    # strings — categorical saves memory on wide tables and speeds up groupby.
    # `row_index` fits in int32 unless a single dataset has >2.1 B rows.
    table["dataset"] = pd.Categorical(table["dataset"])
    table["survey_type"] = pd.Categorical(table["survey_type"])
    table["z_type"] = pd.Categorical(table["z_type"])
    if len(table) and int(table["row_index"].max()) < np.iinfo(np.int32).max:
        table["row_index"] = table["row_index"].astype(np.int32)

    index = OneuidIndex(
        table=table,
        n_unique=res.n_groups,
        n_multi=res.n_multi,
        name=name,
        rules=rules,
        validity=DatasetValidity(
            valid_from_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )

    if persist:
        _archive_previous(database.root, name)
        _write_index(index, database.root)
        logger.info(
            "ONEUID: wrote %s", _index_path(database.root, name),
        )

    return index


def load_oneuid_index(
    database,
    name: str = "default",
    *,
    as_of: Optional[dt_mod.datetime] = None,
) -> OneuidIndex:
    """Load a previously persisted ONEUID index.

    With ``as_of=None`` (default), returns the currently-valid index.
    With a timezone-aware ``as_of``, walks the live + archived manifests
    and returns the index whose :class:`DatasetValidity` contains
    ``as_of``.
    """
    root = database.root
    if as_of is None:
        new_path = _index_path(root, name)
        if new_path.is_file():
            return _read_index(root, name)
        raise FileNotFoundError(
            f"No ONEUID index '{name}' at {new_path}. "
            f"Run database.build_oneuid(name={name!r})."
        )

    if as_of.tzinfo is None:
        raise ValueError("load_oneuid_index: as_of must be tz-aware")

    candidates: List[Path] = []
    live = _index_manifest_path(root, name)
    if live.is_file():
        candidates.append(live)
    archive_dir = root / ONEUID_DIR
    if archive_dir.is_dir():
        candidates.extend(
            sorted(archive_dir.glob(f"{name}__*.manifest.json"))
        )

    for man in candidates:
        with open(man) as fh:
            raw = json.load(fh)
        v_raw = raw.get("validity")
        if v_raw is None:
            continue
        v = DatasetValidity.from_dict(v_raw)
        if v.contains(as_of):
            return _read_index_at(root, man, raw)

    raise FileNotFoundError(
        f"load_oneuid_index: no version of {name!r} valid at {as_of}"
    )


def list_oneuids(database, *, include_archived: bool = False) -> List[str]:
    """Return the names of persisted ONEUID indices under *database*.

    With ``include_archived=True``, also includes archived versions named
    ``{name}__{YYYYmmddTHHMMSSZ}``.
    """
    d = database.root / ONEUID_DIR
    if not d.is_dir():
        return []
    live = sorted(
        p.stem for p in d.glob("*.parquet") if "__" not in p.stem
    )
    if not include_archived:
        return live
    archived = sorted(
        p.stem for p in d.glob("*__*.parquet")
    )
    return live + archived


# ── Internal — I/O ───────────────────────────────────────────────────────


def _resolve_rules(
    rules: Optional[CrossMatchRules],
    sky_tol_arcsec: Optional[float],
    dz_tol: Optional[float],
) -> CrossMatchRules:
    if rules is not None:
        return rules
    kwargs = {}
    if sky_tol_arcsec is not None:
        kwargs["sky_tol_arcsec"] = sky_tol_arcsec
    if dz_tol is not None or sky_tol_arcsec is not None:
        # When the caller used legacy kwargs, honour their dz_tol (None ok).
        kwargs["dz_tol_default"] = dz_tol
    return CrossMatchRules(**kwargs) if kwargs else CrossMatchRules()


def _archive_previous(root: Path, name: str) -> None:
    """If a live index named *name* exists, rename its parquet + manifest
    with a timestamp suffix and close its validity window."""
    idx_path = _index_path(root, name)
    man_path = _index_manifest_path(root, name)
    if not (idx_path.exists() and man_path.exists()):
        return
    now = datetime.now(timezone.utc)
    ts_suffix = now.strftime("%Y%m%dT%H%M%SZ")
    archived_name = f"{name}__{ts_suffix}"
    archived_idx = root / ONEUID_DIR / f"{archived_name}.parquet"
    archived_man = root / ONEUID_DIR / f"{archived_name}.manifest.json"
    idx_path.rename(archived_idx)
    with open(man_path) as fh:
        raw = json.load(fh)
    v = raw.get("validity")
    if v is not None and v.get("valid_to_utc") is None:
        v["valid_to_utc"] = now.isoformat()
        raw["validity"] = v
    man_path.unlink()
    with open(archived_man, "w") as fh:
        json.dump(raw, fh, indent=2, sort_keys=True)


def _write_index(index: OneuidIndex, root: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = root / ONEUID_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(
        pa.Table.from_pandas(index.table, preserve_index=False),
        _index_path(root, index.name),
        compression="zstd",
    )

    # Sidecar manifest.
    rules = index.rules or CrossMatchRules()
    validity = index.validity or DatasetValidity(
        valid_from_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest = {
        "format_version": ONEUID_MANIFEST_FORMAT_VERSION,
        "name": index.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": index.datasets(),
        "n_unique": int(index.n_unique),
        "n_multi": int(index.n_multi),
        "n_rows": int(len(index.table)),
        "rules": {
            "hash": rules.hash(),
            "sky_tol_arcsec": rules.sky_tol_arcsec,
            "dz_tol_default": rules.dz_tol_default,
            "dz_tol_by_ztype": [
                [list(k), v] for k, v in rules.dz_tol_by_ztype.items()
            ],
            "reject_ztype": [list(k) for k in rules.reject_ztype],
        },
        "validity": validity.to_dict(),
    }
    with open(_index_manifest_path(root, index.name), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)


def _read_index(root: Path, name: str) -> OneuidIndex:
    manifest_path = _index_manifest_path(root, name)
    meta: dict = {}
    if manifest_path.is_file():
        with open(manifest_path) as fh:
            meta = json.load(fh)
    return _read_index_at(root, manifest_path, meta, parquet_name=name)


def _read_index_at(
    root: Path, manifest_path: Path, meta: dict,
    parquet_name: Optional[str] = None,
) -> OneuidIndex:
    """Read an index given its manifest path and parsed metadata.

    *parquet_name* overrides the parquet stem used for the table file;
    otherwise the stem is derived from *manifest_path* by dropping
    ``.manifest.json``.
    """
    import pyarrow.parquet as pq

    if parquet_name is None:
        stem = manifest_path.name
        if stem.endswith(".manifest.json"):
            stem = stem[: -len(".manifest.json")]
        parquet_name = stem
    parquet_path = manifest_path.parent / f"{parquet_name}.parquet"
    table = pq.read_table(parquet_path).to_pandas()
    rules = _rules_from_manifest(meta.get("rules", {}))
    name = meta.get("name", parquet_name)
    v_raw = meta.get("validity")
    validity = DatasetValidity.from_dict(v_raw) if v_raw is not None else None
    n_unique = int(meta.get(
        "n_unique", (table["oneuid"].max() + 1) if len(table) else 0,
    ))
    n_multi = int(meta.get(
        "n_multi",
        int((table.groupby("oneuid")["dataset"].nunique() > 1).sum())
        if len(table) else 0,
    ))
    return OneuidIndex(
        table=table, n_unique=n_unique, n_multi=n_multi,
        name=name, rules=rules, validity=validity,
    )


def _rules_from_manifest(d: dict) -> CrossMatchRules:
    pairs = {
        tuple(k): float(v)
        for k, v in (d.get("dz_tol_by_ztype") or [])
    }
    rejects = frozenset(tuple(p) for p in (d.get("reject_ztype") or []))
    return CrossMatchRules(
        sky_tol_arcsec=float(d.get("sky_tol_arcsec", 1.0)),
        dz_tol_default=d.get("dz_tol_default"),
        dz_tol_by_ztype=pairs,
        reject_ztype=rejects,
    )


# ── Tiered query API ─────────────────────────────────────────────────────


class OneuidQuery:
    """Tiered, selector-based query API over the ONEUID index.

    Selectors → ``ndarray[int64]`` of ONEUIDs:
      :meth:`from_id`, :meth:`from_ids`, :meth:`from_foreign_ids`,
      :meth:`from_cone`, :meth:`from_shell`, :meth:`from_skypatch`,
      :meth:`from_selection`.

    Hydration levels → ``DataFrame``:
      :meth:`index_for` (level 0), :meth:`partial_for` (level 1),
      :meth:`full_for` (level 2).
    """

    def __init__(
        self, database, index: Optional[OneuidIndex] = None, *,
        name: str = "default",
    ) -> None:
        self.database = database
        self.index: OneuidIndex = (
            index if index is not None
            else load_oneuid_index(database, name=name)
        )
        self._uid_arr = self.index.table["oneuid"].to_numpy()
        self._ds_arr = self.index.table["dataset"].to_numpy()
        self._row_arr = self.index.table["row_index"].to_numpy()
        self._ra = self.index.table["ra"].to_numpy()
        self._dec = self.index.table["dec"].to_numpy()
        self._z = self.index.table["z"].to_numpy()

    # ── Selectors ────────────────────────────────────────────────────────

    def from_id(self, oneuid: int) -> np.ndarray:
        if not (self._uid_arr == int(oneuid)).any():
            raise KeyError(f"ONEUID {oneuid} not in index")
        return np.array([int(oneuid)], dtype=np.int64)

    def from_ids(self, oneuids: Iterable[int]) -> np.ndarray:
        uids = np.unique(np.asarray(list(oneuids), dtype=np.int64))
        present = np.isin(uids, self._uid_arr)
        return uids[present]

    def from_foreign_ids(
        self, dataset: str, row_indices: Iterable[int],
    ) -> np.ndarray:
        rows = np.asarray(list(row_indices), dtype=np.int64)
        mask = (self._ds_arr == dataset) & np.isin(self._row_arr, rows)
        return np.unique(self._uid_arr[mask]).astype(np.int64)

    def from_cone(self, ra: float, dec: float, radius: float) -> np.ndarray:
        return self.from_selection(Cone(ra=ra, dec=dec, radius=radius))

    def from_shell(self, z_min: float, z_max: float) -> np.ndarray:
        return self.from_selection(Shell(z_min=z_min, z_max=z_max))

    def from_skypatch(
        self, ra_min: float, ra_max: float, dec_min: float, dec_max: float,
    ) -> np.ndarray:
        return self.from_selection(SkyPatch(
            ra_min=ra_min, ra_max=ra_max,
            dec_min=dec_min, dec_max=dec_max,
        ))

    def from_selection(
        self, selection: Union[Selection, Sequence[Selection]],
    ) -> np.ndarray:
        mask = apply_selections(self._ra, self._dec, self._z, selection)
        return np.unique(self._uid_arr[mask]).astype(np.int64)

    # ── Hydration levels ─────────────────────────────────────────────────

    def index_for(self, oneuids: Iterable[int]) -> pd.DataFrame:
        uids = np.asarray(list(oneuids), dtype=np.int64)
        mask = np.isin(self._uid_arr, uids)
        return self.index.table[mask].reset_index(drop=True)

    def iter_partial(
        self,
        oneuids: Iterable[int],
        columns: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ) -> Iterator[pd.DataFrame]:
        """Stream hydrated rows batch-by-batch.

        Yields one :class:`pandas.DataFrame` per batch of up to
        ``batch_size`` ONEUIDs. Peak memory is proportional to
        ``batch_size``, not to ``len(oneuids)``. ``batch_size=None``
        yields a single batch covering every requested ONEUID.
        """
        uids = np.asarray(list(oneuids), dtype=np.int64)
        if uids.size == 0:
            return
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        step = batch_size if batch_size is not None else uids.size
        for start in range(0, uids.size, step):
            chunk = uids[start:start + step]
            sub = self.index.table[np.isin(self._uid_arr, chunk)]
            yield self._hydrate_batch(sub, columns, datasets)

    def partial_for(
        self,
        oneuids: Iterable[int],
        columns: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        uids = np.asarray(list(oneuids), dtype=np.int64)
        pieces = list(
            self.iter_partial(uids, columns=columns, datasets=datasets)
        )
        if not pieces:
            extra = list(columns) if columns is not None else []
            return pd.DataFrame(columns=[
                "oneuid", "dataset", "row_index", "ra", "dec", "z", *extra,
            ])
        return pd.concat(pieces, ignore_index=True)

    def full_for(
        self, oneuids: Iterable[int],
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        return self.partial_for(oneuids, columns=None, datasets=datasets)

    # ── Internals ──────────────────────────────────────────────────────

    def _hydrate_batch(
        self,
        sub: pd.DataFrame,
        columns: Optional[Sequence[str]],
        datasets: Optional[Sequence[str]],
    ) -> pd.DataFrame:
        """Hydrate the ``(dataset, row_index)`` pairs in *sub* into one
        frame via :meth:`DatasetView.scan` with an
        ``_original_row_index`` pushdown filter.
        """
        if datasets is not None:
            sub = sub[sub["dataset"].isin(list(datasets))]
        if sub.empty:
            extra = list(columns) if columns is not None else []
            return pd.DataFrame(columns=[
                "oneuid", "dataset", "row_index", "ra", "dec", "z", *extra,
            ])

        col_arg = list(columns) if columns is not None else None
        pieces: List[pd.DataFrame] = []
        for ds_name, grp in sub.groupby("dataset", sort=False):
            view = self.database[ds_name]
            available = set(view.columns)
            want: Optional[List[str]] = None
            if col_arg is not None:
                want = [c for c in col_arg if c in available]
                if "_original_row_index" not in want:
                    want = want + ["_original_row_index"]
            rows = grp["row_index"].to_numpy()
            expr = pc.field("_original_row_index").isin(
                pa.array(rows, type=pa.int64()),
            )
            tbl = view.scan(columns=want, filter=expr)
            part = tbl.to_pandas()
            if "_original_row_index" in part.columns:
                part = (
                    part.set_index("_original_row_index")
                    .reindex(rows)
                    .reset_index()
                )
                # Drop only if we added _original_row_index internally
                # (caller didn't explicitly request it).
                added_internally = (
                    col_arg is not None
                    and "_original_row_index" not in col_arg
                )
                if added_internally:
                    part = part.drop(columns=["_original_row_index"])
            part = part.drop(
                columns=[c for c in ("ra", "dec", "z") if c in part.columns],
                errors="ignore",
            )
            if col_arg is not None:
                for c in col_arg:
                    if c not in part.columns and c not in ("ra", "dec", "z"):
                        part[c] = np.nan
            part.insert(0, "oneuid", grp["oneuid"].to_numpy())
            part.insert(1, "dataset", ds_name)
            part.insert(2, "row_index", grp["row_index"].to_numpy())
            part.insert(3, "ra", grp["ra"].to_numpy())
            part.insert(4, "dec", grp["dec"].to_numpy())
            part.insert(5, "z", grp["z"].to_numpy())
            pieces.append(part)

        return pd.concat(pieces, ignore_index=True)

    def concurrences(self, oneuid: int) -> pd.DataFrame:
        return self.index_for([oneuid])

    def __repr__(self) -> str:
        return (
            f"<OneuidQuery index={self.index.name!r} "
            f"n_unique={self.index.n_unique} "
            f"n_multi={self.index.n_multi} datasets={self.index.datasets()}>"
        )
