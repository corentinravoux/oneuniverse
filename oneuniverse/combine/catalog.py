"""WeightedCatalog: one hydrated long table over an ONEUID index, combined per object."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa

from oneuniverse._core.errors import NotFoundError, SchemaError, ValidationError
from oneuniverse.combine.defaults import default_weight_for
from oneuniverse.combine.weights import ConstantWeight, Weight
from oneuniverse.data._names import name_list
from oneuniverse.data.database import Database
from oneuniverse.data.oneuid import OneuidIndex, OneuidQuery

_STRATEGIES = ("ivar", "best", "mean", "priority")
_INDEX_COLUMNS = ("oneuid", "dataset")


def _strings(col: pa.ChunkedArray) -> np.ndarray:
    if pa.types.is_dictionary(col.type):
        col = col.cast(col.type.value_type)
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=object)


def _numeric(table: pa.Table, name: str, what: str) -> np.ndarray:
    if name not in table.column_names:
        raise SchemaError(f"{what} column {name!r} not in the catalog; has {table.column_names}")
    if name in _INDEX_COLUMNS:
        raise SchemaError(f"{what} column {name!r} is an index column, not a numeric value")
    col = table[name]
    if not (pa.types.is_integer(col.type) or pa.types.is_floating(col.type)):
        raise SchemaError(f"{what} column {name!r} must be numeric, got {col.type}")
    return np.asarray(col.cast(pa.float64()).to_numpy(zero_copy_only=False), dtype=np.float64)


class WeightedCatalog:
    """Long table ``(oneuid, dataset, z_type, <columns>)`` plus per-dataset weights.

    Build it with :meth:`from_oneuid` (hydrates the index once). Weights are
    survey weights for downstream measurements; :meth:`combine` merges the
    per-object values of several surveys with vectorised group operations.
    """

    def __init__(self, table: pa.Table, datasets: Sequence[str], *,
                 db: Database | None = None) -> None:
        for c in _INDEX_COLUMNS:
            if c not in table.column_names:
                raise SchemaError(f"table lacks the {c!r} column")
        ds_idx = table.schema.get_field_index("dataset")
        if pa.types.is_dictionary(table.schema.field(ds_idx).type):
            table = table.set_column(ds_idx, "dataset", table["dataset"].cast(pa.string()))
        names = name_list(datasets, "datasets")
        if len(set(names)) != len(names):
            raise ValidationError(f"datasets has duplicates: {names}")
        present = set(_strings(table["dataset"]).tolist())
        if set(names) != present:
            raise ValidationError(
                f"datasets {sorted(names)} do not match the datasets in the table {sorted(present)}")
        self._table = table
        self._datasets = names
        self._db = db
        self._ds = _strings(table["dataset"])
        self._rows = {d: np.flatnonzero(self._ds == d) for d in self._datasets}
        self._explicit: dict[str, Weight] = {}
        self._filled: dict[str, np.ndarray] = {}

    @classmethod
    def from_oneuid(cls, index: OneuidIndex, db: Database, *,
                    columns: Sequence[str] | None = None) -> WeightedCatalog:
        """Hydrate every object of *index* once.

        *columns* defaults to the columns present in every dataset of the
        index. ``z_type`` is always hydrated (default weights need it).
        """
        if columns is None:
            shared: list[str] | None = None
            for name in index.datasets:
                names = list(db[name].schema.names)
                shared = names if shared is None else [c for c in shared if c in names]
            cols = [c for c in (shared or []) if c not in _INDEX_COLUMNS]
        else:
            cols = name_list(columns, "columns")
        if "z_type" not in cols:
            cols.append("z_type")
        oneuids = np.unique(index.table["oneuid"].to_numpy())
        table = OneuidQuery(index, db).hydrate_table(oneuids, columns=cols)
        return cls(table, index.datasets, db=db)

    @property
    def datasets(self) -> list[str]:
        return list(self._datasets)

    @property
    def table(self) -> pa.Table:
        return self._table

    def add_weight(self, dataset: str, weight: Weight) -> WeightedCatalog:
        """Set the weight of *dataset* (replacing an earlier one); returns self."""
        if dataset not in self._rows:
            raise NotFoundError(f"dataset {dataset!r} not in the catalog; has {self._datasets}")
        self._explicit[dataset] = weight
        self._filled.pop(dataset, None)
        return self

    def fill_defaults(self, db: Database | None = None, *, skip_unknown: bool = False) -> WeightedCatalog:
        """Give every dataset without an explicit weight its default weight.

        The default depends on the dataset's survey kind (manifest) and on
        each row's ``z_type``. Unknown ``(kind, z_type)`` pairs raise
        NotFoundError, or get ``ConstantWeight(1.0)`` with *skip_unknown*.
        *db* defaults to the Database given to :meth:`from_oneuid`.
        """
        db = db if db is not None else self._db
        if db is None:
            raise ValidationError("fill_defaults needs a Database: pass db= or build with from_oneuid")
        if "z_type" not in self._table.column_names:
            raise SchemaError("fill_defaults needs a 'z_type' column in the catalog")
        zt_all = _strings(self._table["z_type"])
        for ds in self._datasets:
            if ds in self._explicit:
                continue
            kind = db.manifest(ds).survey.kind
            rows = self._rows[ds]
            out = np.empty(rows.size, dtype=np.float64)
            zt = zt_all[rows]
            null = pd.isna(zt)
            groups = [(z, np.flatnonzero(~null & (zt == z))) for z in sorted(set(zt[~null]))]
            if null.any():
                groups.append(("None", np.flatnonzero(null)))
            for z_type, sel in groups:
                try:
                    weight = default_weight_for(kind, z_type)
                except NotFoundError:
                    if not skip_unknown:
                        raise
                    weight = ConstantWeight(1.0)
                out[sel] = weight(self._table.take(rows[sel]))
            self._filled[ds] = out
        return self

    def weights(self) -> pa.Table:
        """``(oneuid, dataset, weight)`` in the row order of :attr:`table`."""
        missing = [d for d in self._datasets if d not in self._explicit and d not in self._filled]
        if missing:
            raise ValidationError(
                f"no weight for datasets {missing}; call add_weight or fill_defaults")
        w = np.full(self._table.num_rows, np.nan)
        for ds in self._datasets:
            rows = self._rows[ds]
            if ds in self._explicit:
                w[rows] = self._explicit[ds](self._table.take(rows))
            else:
                w[rows] = self._filled[ds]
        return pa.table({"oneuid": self._table["oneuid"], "dataset": self._table["dataset"],
                         "weight": pa.array(w)})

    def combine(self, value: str, *, variance: str | None = None,
                strategy: Literal["ivar", "best", "mean", "priority"] = "ivar",
                priority: Sequence[str] | None = None) -> pa.Table:
        """Combine *value* across the datasets of each object.

        *variance* names the 1-sigma **uncertainty** column; the output column
        ``variance`` is the combined sigma^2. Output columns: ``oneuid`` (sorted),
        ``value``, ``variance``, ``n_sources``, ``best_dataset``.

        A sigma is *valid* when finite and > 0. Surveys without redshift errors
        (real eBOSS rows have ``z_err`` = NaN) keep their objects: a member with
        a finite value always counts, and a missing sigma only makes the
        combined variance NaN. Strategies (s = sigma, v = value):

        - ``"ivar"`` (requires *variance*): members with valid s combine as
          value = sum(v/s^2)/sum(1/s^2), variance = 1/sum(1/s^2); best_dataset =
          smallest valid s. An object with no valid-s member gets the unweighted
          mean of its finite values and variance NaN (best_dataset = first
          finite member in catalog dataset order). Catalog weights are unused.
        - ``"best"`` (requires *variance*): the member with the smallest valid s
          gives value and s^2; with none, the first finite-value member in
          catalog dataset order, variance NaN.
        - ``"mean"`` (needs weights for every dataset): every finite-value
          member with finite w > 0 counts in value = sum(w v)/sum(w).
          With *variance*, variance = sum(w^2 s^2)/sum(w)^2 only if every
          counted member has a valid s, else NaN; without it, the weighted
          sample variance sum(w (v-mean)^2)/sum(w). best_dataset = largest w.
        - ``"priority"`` (*variance* optional): value of the first member with a
          finite value in the order *priority* (dataset names; default catalog
          order; datasets not listed come after, in catalog order); variance =
          s^2 of that member if *variance* is given and s is valid, else NaN.
          Unknown names raise NotFoundError. This is the usual choice for
          multi-survey QSO redshifts.

        ``n_sources`` is the number of members that contributed to the value
        (0 and NaN value/variance when none did; best_dataset is then None).
        Sigmas from different pipelines may not be comparable; prefer
        ``"priority"`` when they are not. Positions (``ra``, ``dec``) are not
        combinable: they live in the ONEUID index.
        """
        if strategy not in _STRATEGIES:
            raise ValidationError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        if value in ("ra", "dec"):
            raise ValidationError(
                "positions live in the ONEUID index; combine does not average angles")
        if priority is not None and strategy != "priority":
            raise ValidationError("priority= is only used by strategy 'priority'")
        v = _numeric(self._table, value, "value")
        sigma = None if variance is None else _numeric(self._table, variance, "uncertainty")
        if strategy in ("ivar", "best") and sigma is None:
            raise ValidationError(f"strategy {strategy!r} requires the uncertainty column (variance=)")
        uid, inv = np.unique(self._table["oneuid"].to_numpy(), return_inverse=True)
        k = uid.size
        n_rows = v.size
        # position of each row's dataset: catalog order, or priority order (unlisted last)
        order_names = list(self._datasets)
        if strategy == "priority" and priority is not None:
            listed = name_list(priority, "priority")
            if len(set(listed)) != len(listed):
                raise ValidationError(f"priority has duplicate datasets: {listed}")
            unknown = [d for d in listed if d not in self._rows]
            if unknown:
                raise NotFoundError(f"priority datasets {unknown} not in the catalog; has {self._datasets}")
            order_names = listed + [d for d in self._datasets if d not in listed]
        pos = np.empty(n_rows, dtype=np.int64)
        for i, d in enumerate(order_names):
            pos[self._rows[d]] = i

        fin = np.isfinite(v)
        if sigma is not None:
            with np.errstate(invalid="ignore"):
                sv = np.isfinite(sigma) & (sigma > 0)
        else:
            sv = np.zeros(n_rows, dtype=bool)
        vv = np.where(fin, v, 0.0)
        s2 = np.where(sv, sigma if sigma is not None else 0.0, 0.0) ** 2

        def bsum(x):
            return np.bincount(inv, x, k)

        with np.errstate(invalid="ignore", divide="ignore"):
            if strategy == "ivar":
                use = fin & sv
                n_valid = np.bincount(inv, weights=use, minlength=k)
                iv = np.where(use, 1.0 / np.where(sv, s2, 1.0), 0.0)
                isum = bsum(iv)
                nf = np.bincount(inv, weights=fin, minlength=k)
                fallback = n_valid == 0
                val = np.where(fallback, bsum(vv) / nf, bsum(iv * vv) / isum)
                var = np.where(fallback, np.nan, 1.0 / isum)
                n = np.where(fallback, nf, n_valid)
                key1 = np.where(use, 0, np.where(fin, 1, 2))
                key2 = np.where(use, np.sqrt(s2), pos)
                pick = None
            elif strategy == "mean":
                w = self.weights()["weight"].to_numpy()
                use = fin & np.isfinite(w) & (w > 0)
                ww = np.where(use, w, 0.0)
                wsum = bsum(ww)
                val = bsum(ww * vv) / wsum
                if sigma is not None:
                    bad = np.bincount(inv, weights=use & ~sv, minlength=k)
                    var = np.where(bad > 0, np.nan, bsum(ww**2 * s2) / wsum**2)
                else:
                    var = bsum(ww * (vv - val[inv]) ** 2) / wsum
                n = np.bincount(inv, weights=use, minlength=k)
                key1 = np.where(use, 0, 1)
                key2 = np.where(use, -w, 0.0)
                pick = None
            else:
                use = fin
                n = np.bincount(inv, weights=use, minlength=k)
                key1 = np.where(use, 0, 1)
                key2 = np.where(use, pos, 0)
                if strategy == "best":
                    key1 = np.where(fin & sv, 0, np.where(fin, 1, 2))
                    key2 = np.where(fin & sv, np.sqrt(s2), pos)
                pick = True
        order = np.lexsort((key2, key1, inv))
        first = order[np.searchsorted(inv[order], np.arange(k))]
        if pick:
            val = v[first]
            var = np.where(sv[first], s2[first], np.nan)
            n = np.minimum(n, 1)
        n = np.asarray(n).astype(np.int64)
        has = n > 0
        val = np.where(has, val, np.nan)
        var = np.where(has, var, np.nan)
        best = pa.array(self._ds[first], pa.string(), mask=~has)
        return pa.table({"oneuid": pa.array(uid, pa.int64()), "value": pa.array(val),
                         "variance": pa.array(var), "n_sources": pa.array(n),
                         "best_dataset": best})
