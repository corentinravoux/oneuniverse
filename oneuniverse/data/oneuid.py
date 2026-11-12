"""ONEUID: cross-survey identity index with position-based hydration.

An index maps every member row of the chosen datasets to an ``oneuid`` (the
connected component of the pair-wise cross-match graph). Each index row stores
the member's *position* in its dataset -- ``(partition, offset)`` -- next to
``source_row``, so hydration reads exactly the rows the index points at. The
index records the ``DatasetView.version`` of every dataset it was built from;
hydrating after a dataset was rewritten raises ``FormatError`` instead of
returning another version's rows.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.geometry import Cone, SkyPatch, ZRange
from oneuniverse.data.crossmatch import CrossMatchRules, components, match_pair
from oneuniverse.data.sidecar import VersionedSidecar

__all__ = ["ONEUID_SCHEMA", "OneuidIndex", "OneuidQuery", "build_oneuid", "load_oneuid"]

ONEUID_SCHEMA = pa.schema([
    ("oneuid", pa.int64()),
    ("dataset", pa.dictionary(pa.int16(), pa.string())),
    ("source_row", pa.int64()),
    ("partition", pa.int32()),
    ("offset", pa.int64()),
    ("ra", pa.float64()),
    ("dec", pa.float64()),
    ("z", pa.float64()),
    ("z_type", pa.dictionary(pa.int8(), pa.string())),
])

_KIND = "oneuid"
_MATCH_COLUMNS = ["ra", "dec", "z", "z_type", "source_row", "hpix32"]


def _dict(strings: pa.Array | pa.ChunkedArray, index_type: pa.DataType) -> pa.Array:
    if isinstance(strings, pa.ChunkedArray):
        strings = strings.combine_chunks()
    enc = strings.dictionary_encode()
    return pa.DictionaryArray.from_arrays(enc.indices.cast(index_type), enc.dictionary)


def _load_dataset(view) -> tuple[pa.Table, int]:
    """ra/dec/z/z_type/source_row/hpix32 of all rows, plus partition and offset."""
    pieces = []
    for part, tab in view.iter_partitions(columns=_MATCH_COLUMNS):
        n = tab.num_rows
        zt = tab["z_type"]
        if pa.types.is_dictionary(zt.type):
            zt = zt.cast(pa.string())
        tab = tab.set_column(tab.schema.get_field_index("z_type"), "z_type", zt)
        tab = tab.append_column("partition", pa.array(np.full(n, part, np.int32)))
        tab = tab.append_column("offset", pa.array(np.arange(n, dtype=np.int64)))
        pieces.append(tab)
    if not pieces:
        raise ValidationError("dataset has no rows to index")
    return pa.concat_tables(pieces).combine_chunks(), sum(p.num_rows for p in pieces)


def build_oneuid(db, datasets: Sequence[str], *, rules: CrossMatchRules,
                 name: str = "default", block_nside: int = 8) -> OneuidIndex:
    """Cross-match *datasets* pair-wise and publish the ONEUID index *name*.

    Identity is per dataset: one int16 code per dataset is the "survey" given
    to :func:`components`, so what ``allow_same_survey_merge=False`` forbids is
    two members of the same *dataset* sharing an oneuid. A dataset is never
    matched against itself. Datasets are used in sorted name order.

    Memory: ra, dec, z, z_type, source_row and hpix32 plus positions of **all**
    rows of the chosen datasets are held in memory at once (about 60 bytes per
    row).
    """
    names = sorted(set(datasets))
    if not names:
        raise ValidationError("build_oneuid needs at least one dataset")
    if len(names) > 32767:
        raise ValidationError("too many datasets")
    views = [db[n] for n in names]  # NotFoundError for unknown names
    versions = {n: list(v.version) for n, v in zip(names, views)}
    tables = [_load_dataset(v)[0] for v in views]
    sizes = [t.num_rows for t in tables]
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    total = int(off[-1])
    L, R, S = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = match_pair(tables[i], tables[j], rules, block_nside=block_nside)
            L.append(p.left + off[i])
            R.append(p.right + off[j])
            S.append(p.sep_arcsec)
    left = np.concatenate(L) if L else np.empty(0, np.int64)
    right = np.concatenate(R) if R else np.empty(0, np.int64)
    sep = np.concatenate(S) if S else np.empty(0, np.float64)
    dcode = np.repeat(np.arange(len(names), dtype=np.int16), sizes)
    label = components(total, left, right, sep, survey_of=dcode,
                       allow_same_survey_merge=rules.allow_same_survey_merge)
    full = pa.concat_tables(tables).combine_chunks()
    src = full["source_row"].to_numpy()
    order = np.lexsort((src, dcode, label))
    ds_dict = pa.array(names, pa.string())
    table = pa.table({
        "oneuid": pa.array(label[order].astype(np.int64)),
        "dataset": pa.DictionaryArray.from_arrays(pa.array(dcode[order]), ds_dict),
        "source_row": pa.array(src[order].astype(np.int64)),
        "partition": full["partition"].take(pa.array(order)),
        "offset": full["offset"].take(pa.array(order)),
        "ra": full["ra"].take(pa.array(order)),
        "dec": full["dec"].take(pa.array(order)),
        "z": full["z"].take(pa.array(order)),
        "z_type": _dict(full["z_type"].take(pa.array(order)), pa.int8()),
    }, schema=ONEUID_SCHEMA)
    n_objects = int(label.max()) + 1 if total else 0
    counts = np.bincount(label, minlength=n_objects)
    meta = {
        "rules": rules.to_dict(), "rules_hash": rules.hash(), "datasets": names,
        "dataset_versions": versions, "n_objects": n_objects,
        "n_multi": int((counts >= 2).sum()), "block_nside": int(block_nside),
    }
    VersionedSidecar(db.root, _KIND, name).write(table, meta)
    return load_oneuid(db, name)


def load_oneuid(db, name: str = "default", *, as_of=None) -> OneuidIndex:
    """Load ONEUID index *name* (the version live at *as_of* if given)."""
    table, meta = VersionedSidecar(db.root, _KIND, name).load(as_of=as_of)
    return OneuidIndex(table, meta, name)


class OneuidIndex:
    """An ONEUID table (``ONEUID_SCHEMA``, sorted by oneuid, dataset, source_row) + meta."""

    def __init__(self, table: pa.Table, meta: dict, name: str) -> None:
        self.table = table
        self.meta = meta
        self.name = name

    @property
    def datasets(self) -> list[str]:
        return list(self.meta["datasets"])

    @property
    def rules(self) -> CrossMatchRules:
        return CrossMatchRules.from_dict(self.meta["rules"])

    @property
    def n_objects(self) -> int:
        return int(self.meta["n_objects"])

    @property
    def n_multi(self) -> int:
        return int(self.meta["n_multi"])

    def multi_only(self) -> pa.Table:
        """Index rows of oneuids with at least two members."""
        ids = self.table["oneuid"].to_numpy()
        u, c = np.unique(ids, return_counts=True)
        return self.table.filter(pa.array(np.isin(ids, u[c >= 2])))

    def of(self, oneuid: int) -> pa.Table:
        """Index rows of one oneuid."""
        return self.table.filter(pc.equal(self.table["oneuid"], pa.scalar(int(oneuid), pa.int64())))

    def restrict_to(self, datasets: Sequence[str]) -> OneuidIndex:
        """Keep only rows of *datasets*; oneuid values are unchanged."""
        keep = sorted(set(datasets))
        missing = [d for d in keep if d not in self.meta["datasets"]]
        if missing:
            raise ValidationError(f"datasets {missing} not in index {self.name!r}")
        ds = self.table["dataset"].combine_chunks().cast(pa.string())
        t = self.table.filter(pc.is_in(ds, pa.array(keep, pa.string())))
        ids = t["oneuid"].to_numpy()
        _, c = np.unique(ids, return_counts=True)
        meta = dict(self.meta)
        meta.update(
            datasets=keep, n_objects=len(c), n_multi=int((c >= 2).sum()),
            dataset_versions={d: self.meta["dataset_versions"][d] for d in keep})
        return OneuidIndex(t, meta, self.name)


class OneuidQuery:
    """Selection and hydration of the objects of an :class:`OneuidIndex`."""

    def __init__(self, index: OneuidIndex, db) -> None:
        self.index = index
        self.db = db

    def _ds_names(self) -> np.ndarray:
        d = self.index.table["dataset"].combine_chunks()
        return np.asarray(d.dictionary.to_pylist(), dtype=object)[
            d.indices.to_numpy(zero_copy_only=False)]

    def select(self, *, cone: Cone | None = None, patch: SkyPatch | None = None, z=None,
               datasets: Sequence[str] | None = None) -> np.ndarray:
        """Unique oneuids (sorted int64) with at least one member row matching every filter."""
        t = self.index.table
        mask = np.ones(t.num_rows, bool)
        if cone is not None or patch is not None:
            ra, dec = t["ra"].to_numpy(), t["dec"].to_numpy()
            if cone is not None:
                mask &= cone.contains(ra, dec)
            if patch is not None:
                mask &= patch.contains(ra, dec)
        if z is not None:
            if not isinstance(z, ZRange):
                try:
                    lo, hi = z
                    lo, hi = float(lo), float(hi)
                except (TypeError, ValueError) as e:
                    raise ValidationError(f"z must be a ZRange or (lo, hi), got {z!r}") from e
                if not lo <= hi:
                    raise ValidationError(f"z requires lo <= hi, got ({lo}, {hi})")
                zz = t["z"].to_numpy()
                mask &= (zz >= lo) & (zz <= hi)
            else:
                mask &= z.contains(t["z"].to_numpy())
        if datasets is not None:
            mask &= np.isin(self._ds_names(), list(datasets))
        return np.unique(t["oneuid"].to_numpy()[mask]).astype(np.int64)

    def rows(self, oneuids) -> pa.Table:
        """Index rows of *oneuids* (unknown ids ignored), sorted by oneuid then dataset."""
        ids = np.asarray(oneuids, dtype=np.int64).ravel()
        t = self.index.table
        return t.filter(pa.array(np.isin(t["oneuid"].to_numpy(), ids)))

    def _check_version(self, name: str, view) -> None:
        recorded = self.index.meta["dataset_versions"].get(name)
        if recorded is None or list(view.version) != list(recorded):
            raise FormatError(
                f"dataset {name} changed since ONEUID {self.index.name} was built; "
                f"rebuild the index")

    def hydrate(self, oneuids, *, columns=None, datasets: Sequence[str] | None = None,
                batch_size: int = 100_000) -> Iterator[pa.Table]:
        """Yield the dataset rows of *oneuids*, per dataset and per batch.

        Each batch holds ``columns`` (all dataset columns if None) plus
        ``oneuid`` (int64) and ``dataset`` (string); rows are ordered by
        ``(oneuid, source_row)`` within a dataset. Raises FormatError if a
        dataset changed since the index was built.
        """
        if batch_size < 1:
            raise ValidationError("batch_size must be >= 1")
        if columns is not None and {"oneuid", "dataset"} & set(columns):
            raise ValidationError("columns must not include 'oneuid' or 'dataset'")
        rows = self.rows(oneuids)
        ds_col = rows["dataset"].combine_chunks()
        ds_all = np.asarray(ds_col.dictionary.to_pylist(), dtype=object)[
            ds_col.indices.to_numpy(zero_copy_only=False)]
        wanted = sorted(set(self.index.datasets if datasets is None else datasets))
        unknown = [d for d in wanted if d not in self.index.datasets]
        if unknown:
            raise NotFoundError(f"datasets {unknown} not in ONEUID {self.index.name!r}")
        oid = rows["oneuid"].to_numpy()
        part = rows["partition"].to_numpy()
        off = rows["offset"].to_numpy()
        for name in wanted:
            sel = np.flatnonzero(ds_all == name)
            if sel.size == 0:
                continue
            view = self.db[name]
            self._check_version(name, view)
            for s in range(0, sel.size, batch_size):
                ix = sel[s:s + batch_size]
                t = view.take(part[ix], off[ix], columns=columns)
                t = t.append_column("oneuid", pa.array(oid[ix], pa.int64()))
                t = t.append_column("dataset", pa.array([name] * len(ix), pa.string()))
                yield t

    def hydrate_table(self, oneuids, *, columns=None,
                      datasets: Sequence[str] | None = None) -> pa.Table:
        """All batches of :meth:`hydrate` concatenated (schemas promoted across datasets)."""
        batches = list(self.hydrate(oneuids, columns=columns, datasets=datasets))
        if not batches:
            cols = ([] if columns is None else list(columns)) + ["oneuid", "dataset"]
            fields = [pa.field(c, pa.int64() if c == "oneuid" else pa.string()) for c in cols]
            return pa.schema(fields).empty_table()
        return pa.concat_tables(batches, promote_options="default")
