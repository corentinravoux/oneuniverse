"""WeightedCatalog: one hydrated long table over an ONEUID index, combined per object."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import NotFoundError, SchemaError, ValidationError
from oneuniverse.combine.defaults import default_weight_for
from oneuniverse.combine.weights import ConstantWeight, Weight
from oneuniverse.data._names import name_list
from oneuniverse.data.database import Database
from oneuniverse.data.oneuid import OneuidIndex, OneuidQuery

_STRATEGIES = ("ivar", "best", "mean")
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

    def __init__(self, table: pa.Table, datasets: Sequence[str]) -> None:
        for c in _INDEX_COLUMNS:
            if c not in table.column_names:
                raise SchemaError(f"table lacks the {c!r} column")
        ds_idx = table.schema.get_field_index("dataset")
        if pa.types.is_dictionary(table.schema.field(ds_idx).type):
            table = table.set_column(ds_idx, "dataset", table["dataset"].cast(pa.string()))
        self._table = table
        self._datasets = list(datasets)
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
        return cls(table, index.datasets)

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

    def fill_defaults(self, db: Database, *, skip_unknown: bool = False) -> WeightedCatalog:
        """Give every dataset without an explicit weight its default weight.

        The default depends on the dataset's survey kind (manifest) and on
        each row's ``z_type``. Unknown ``(kind, z_type)`` pairs raise
        NotFoundError, or get ``ConstantWeight(1.0)`` with *skip_unknown*.
        """
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
            for z_type in sorted({str(z) for z in zt}):
                sel = np.flatnonzero(zt == (None if z_type == "None" else z_type))
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
                strategy: Literal["ivar", "best", "mean"] = "ivar") -> pa.Table:
        """Combine *value* across the datasets of each object.

        *variance* names the 1-sigma **uncertainty** column; the output column
        ``variance`` is the combined sigma^2. Output columns: ``oneuid`` (sorted),
        ``value``, ``variance``, ``n_sources``, ``best_dataset``.

        - ``"ivar"``: value = sum(v/s^2)/sum(1/s^2), variance = 1/sum(1/s^2).
          Catalog weights are not used. Requires *variance*.
        - ``"best"``: value and s^2 of the member with the smallest s. Requires
          *variance*.
        - ``"mean"``: catalog-weighted mean sum(w v)/sum(w); variance =
          sum(w^2 s^2)/sum(w)^2 with *variance*, else the weighted sample
          variance sum(w (v-mean)^2)/sum(w). Needs weights for every dataset.

        Members with non-finite value, or (ivar/best) NaN or <= 0 s, (mean)
        non-finite w or w <= 0 or non-finite s, are excluded; ``n_sources``
        counts the valid members and an object with none gets NaN.
        ``best_dataset`` is the dataset of the member with the smallest s
        (ivar/best) or the largest weight (mean); None when no member is valid.
        """
        if strategy not in _STRATEGIES:
            raise ValidationError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
        v = _numeric(self._table, value, "value")
        sigma = None if variance is None else _numeric(self._table, variance, "uncertainty")
        if strategy != "mean" and sigma is None:
            raise ValidationError(f"strategy {strategy!r} requires the uncertainty column (variance=)")
        uid, inv = np.unique(self._table["oneuid"].to_numpy(), return_inverse=True)
        k = uid.size
        if strategy == "mean":
            w = self.weights()["weight"].to_numpy()
            valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
            if sigma is not None:
                valid &= np.isfinite(sigma)
            rank = np.where(valid, -w, np.inf)  # best member = largest weight
        else:
            with np.errstate(invalid="ignore"):
                valid = np.isfinite(v) & np.isfinite(sigma) & (sigma > 0)
            rank = np.where(valid, sigma, np.inf)  # best member = smallest sigma
        n = np.bincount(inv, weights=valid, minlength=k).astype(np.int64)
        has = n > 0
        vv = np.where(valid, v, 0.0)

        if strategy == "mean":
            ww = np.where(valid, w, 0.0)
            wsum = np.bincount(inv, ww, k)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.bincount(inv, ww * vv, k) / wsum
                if sigma is not None:
                    s2 = np.where(valid, sigma, 0.0) ** 2
                    var = np.bincount(inv, ww**2 * s2, k) / wsum**2
                else:
                    var = np.bincount(inv, ww * (vv - mean[inv]) ** 2, k) / wsum
            val = mean
        elif strategy == "ivar":
            with np.errstate(invalid="ignore", divide="ignore"):
                iv = np.where(valid, 1.0 / sigma**2, 0.0)
                isum = np.bincount(inv, iv, k)
                val = np.bincount(inv, iv * vv, k) / isum
                var = 1.0 / isum
        order = np.lexsort((rank, inv))
        first = order[np.searchsorted(inv[order], np.arange(k))]
        if strategy == "best":
            val = np.where(has, v[first], np.nan)
            var = np.where(has, sigma[first] ** 2, np.nan)
        val = np.where(has, val, np.nan)
        var = np.where(has, var, np.nan)
        best = pa.array(self._ds[first], pa.string(), mask=~has)
        return pa.table({"oneuid": pa.array(uid, pa.int64()), "value": pa.array(val),
                         "variance": pa.array(var), "n_sources": pa.array(n),
                         "best_dataset": best})
