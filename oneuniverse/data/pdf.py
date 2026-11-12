"""Photo-z probability densities stored as a manifest extension.

Three representations of ``p(z)`` per object, all stored in one list column
(default ``z_pdf``):

* ``interp``: ``p(z)`` sampled on a shared, strictly increasing ``grid``
  (fixed-size list of ``len(grid)`` float32);
* ``quant``: the redshift quantiles at shared ``quant_levels`` in (0, 1)
  (fixed-size list of ``len(quant_levels)`` float32);
* ``sample``: a ragged list of posterior draws per object.

The manifest carries ``extensions["pdf"] = {"version": 1, **spec.to_dict()}``.
Everything is vectorised over rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import SchemaError, ValidationError

_KINDS = ("interp", "sample", "quant")


@dataclass(frozen=True, slots=True)
class PdfSpec:
    """How the photo-z PDF is stored. ``n_components`` is the per-row width
    (``None`` for the ragged ``sample`` kind)."""

    kind: Literal["interp", "sample", "quant"]
    grid: tuple[float, ...] | None = None
    quant_levels: tuple[float, ...] | None = None
    value_column: str = "z_pdf"

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValidationError(f"pdf kind must be one of {_KINDS}, got {self.kind!r}")
        grid = None if self.grid is None else tuple(float(g) for g in self.grid)
        lev = None if self.quant_levels is None else tuple(float(q) for q in self.quant_levels)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "quant_levels", lev)
        if self.kind == "interp":
            if grid is None or len(grid) < 2 or not np.all(np.diff(grid) > 0):
                raise ValidationError(
                    "interp pdf needs a grid of >= 2 strictly increasing values")
        elif self.kind == "quant":
            if (lev is None or len(lev) < 2 or not np.all(np.diff(lev) > 0)
                    or lev[0] <= 0.0 or lev[-1] >= 1.0):
                raise ValidationError(
                    "quant pdf needs >= 2 strictly increasing quant_levels inside (0, 1)")
        if not self.value_column:
            raise ValidationError("value_column must be a non-empty string")

    @property
    def n_components(self) -> int | None:
        if self.kind == "interp":
            return len(self.grid)
        if self.kind == "quant":
            return len(self.quant_levels)
        return None

    def to_dict(self) -> dict:
        return {"kind": self.kind,
                "grid": None if self.grid is None else list(self.grid),
                "quant_levels": None if self.quant_levels is None else list(self.quant_levels),
                "value_column": self.value_column}

    @classmethod
    def from_dict(cls, d: dict) -> PdfSpec:
        try:
            return cls(kind=d["kind"],
                       grid=None if d.get("grid") is None else tuple(d["grid"]),
                       quant_levels=(None if d.get("quant_levels") is None
                                     else tuple(d["quant_levels"])),
                       value_column=d.get("value_column", "z_pdf"))
        except KeyError as e:
            raise ValidationError(f"pdf spec is missing key {e}") from e


class ProbabilisticRedshift:
    """Per-object photo-z PDFs with vectorised summaries and sampling."""

    def __init__(self, spec: PdfSpec, values) -> None:
        self.spec = spec
        if spec.kind == "sample":
            if isinstance(values, np.ndarray) and values.ndim == 2:
                values = list(values)
            rows = [np.asarray(v, dtype=np.float64).ravel() for v in values]
            self._counts = np.array([r.size for r in rows], np.int64)
            self._offsets = np.concatenate([[0], np.cumsum(self._counts)])
            self._flat = np.concatenate(rows) if rows else np.empty(0)
            self._n = len(rows)
            self._values = None
        else:
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != spec.n_components:
                raise ValidationError(
                    f"{spec.kind} values must have shape (n, {spec.n_components}), "
                    f"got {arr.shape}")
            self._values = arr
            self._n = arr.shape[0]
            self._cdf = None

    def __len__(self) -> int:
        return self._n

    # ---- construction -------------------------------------------------
    @classmethod
    def from_table(cls, table: pa.Table, spec: PdfSpec) -> ProbabilisticRedshift:
        name = spec.value_column
        if name not in table.schema.names:
            raise SchemaError(f"pdf column {name!r} not in table")
        col = table[name]
        arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)
                or pa.types.is_fixed_size_list(arr.type)):
            raise SchemaError(f"pdf column {name!r} must be a list type, got {arr.type}")
        if arr.null_count:
            raise SchemaError(f"pdf column {name!r} has null rows")
        n = len(arr)
        if spec.kind == "sample":
            if pa.types.is_fixed_size_list(arr.type):
                w = arr.type.list_size
                return cls(spec, arr.flatten().to_numpy(zero_copy_only=False).reshape(n, w))
            off = arr.offsets.to_numpy()
            flat = arr.flatten().to_numpy(zero_copy_only=False)
            off = off - off[0]
            obj = cls.__new__(cls)
            obj.spec = spec
            obj._flat = flat.astype(np.float64, copy=False)
            obj._offsets = off.astype(np.int64)
            obj._counts = np.diff(obj._offsets)
            obj._n = n
            obj._values = None
            return obj
        k = spec.n_components
        flat = arr.flatten().to_numpy(zero_copy_only=False)
        if flat.size != n * k or (not pa.types.is_fixed_size_list(arr.type)
                                  and np.any(np.diff(arr.offsets.to_numpy()) != k)):
            raise SchemaError(f"pdf column {name!r} rows must all have {k} components")
        return cls(spec, flat.reshape(n, k))

    # ---- interp helpers ----------------------------------------------
    def _need(self, *kinds: str, op: str) -> None:
        if self.spec.kind not in kinds:
            raise NotImplementedError(
                f"{op}() is not available for pdf kind {self.spec.kind!r} "
                f"(supported: {', '.join(kinds)})")

    def _grid(self) -> np.ndarray:
        return np.asarray(self.spec.grid, np.float64)

    def _norm_cdf(self) -> np.ndarray:
        """Row-normalised cumulative trapezoid, shape (n, ng); NaN rows if p integrates to <= 0."""
        if self._cdf is None:
            g, p = self._grid(), self._values
            seg = 0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(g)[None]
            c = np.concatenate([np.zeros((self._n, 1)), np.cumsum(seg, axis=1)], axis=1)
            tot = c[:, -1:]
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.where(tot > 0, c / np.where(tot > 0, tot, 1.0), np.nan)
            c = np.clip(c, 0.0, 1.0)
            c[:, -1] = np.where(np.isnan(c[:, -1]), np.nan, 1.0)
            self._cdf = c
        return self._cdf

    def _moments(self) -> tuple[np.ndarray, np.ndarray]:
        g, p = self._grid(), self._values
        w = np.diff(g)[None]
        tot = np.sum(0.5 * (p[:, 1:] + p[:, :-1]) * w, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = np.sum(0.5 * (p[:, 1:] * g[1:] + p[:, :-1] * g[:-1]) * w, axis=1) / tot
            m2 = np.sum(0.5 * (p[:, 1:] * g[1:] ** 2 + p[:, :-1] * g[:-1] ** 2) * w,
                        axis=1) / tot
        return m1, np.sqrt(np.maximum(m2 - m1 ** 2, 0.0))

    def _sample_moments(self) -> tuple[np.ndarray, np.ndarray]:
        c = self._counts
        cs = np.concatenate([[0.0], np.cumsum(self._flat)])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (cs[self._offsets[1:]] - cs[self._offsets[:-1]]) / c
            dev2 = (self._flat - np.repeat(mean, c)) ** 2
            cs2 = np.concatenate([[0.0], np.cumsum(dev2)])
            std = np.sqrt((cs2[self._offsets[1:]] - cs2[self._offsets[:-1]]) / c)
        return mean, std

    # ---- summaries ----------------------------------------------------
    def mean(self) -> np.ndarray:
        self._need("interp", "sample", op="mean")
        return self._moments()[0] if self.spec.kind == "interp" else self._sample_moments()[0]

    def std(self) -> np.ndarray:
        self._need("interp", "sample", op="std")
        return self._moments()[1] if self.spec.kind == "interp" else self._sample_moments()[1]

    def cdf(self, z) -> np.ndarray:
        """CDF at ``z`` (scalar or one value per row), linearly interpolated (interp only)."""
        self._need("interp", op="cdf")
        g, c = self._grid(), self._norm_cdf()
        zz = np.broadcast_to(np.asarray(z, np.float64), (self._n,))
        i = np.clip(np.searchsorted(g, zz), 1, g.size - 1)
        w = np.clip((zz - g[i - 1]) / (g[i] - g[i - 1]), 0.0, 1.0)
        c0 = np.take_along_axis(c, (i - 1)[:, None], 1)[:, 0]
        c1 = np.take_along_axis(c, i[:, None], 1)[:, 0]
        return c0 + w * (c1 - c0)

    def _ppf2d(self, q: np.ndarray) -> np.ndarray:
        """Quantile function for q of shape (n, k), values clamped to the stored range."""
        n, k = q.shape
        if self.spec.kind == "quant":
            lev = np.asarray(self.spec.quant_levels, np.float64)
            i = np.clip(np.searchsorted(lev, q), 1, lev.size - 1)
            w = np.clip((q - lev[i - 1]) / (lev[i] - lev[i - 1]), 0.0, 1.0)
            v0 = np.take_along_axis(self._values, i - 1, 1)
            v1 = np.take_along_axis(self._values, i, 1)
            return v0 + w * (v1 - v0)
        g, c = self._grid(), self._norm_cdf()
        ng = g.size
        rows = np.arange(n)[:, None]
        # Row-offset trick: rows of c are in [0, 1], so flat + 2*row is globally sorted.
        flat = (c + 2.0 * rows).ravel()
        idx = np.searchsorted(flat, q + 2.0 * rows) - rows * ng
        i = np.clip(idx, 1, ng - 1)
        c0 = np.take_along_axis(c, i - 1, 1)
        c1 = np.take_along_axis(c, i, 1)
        dc = c1 - c0
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(dc > 0, (q - c0) / np.where(dc > 0, dc, 1.0), 0.0)
        w = np.clip(w, 0.0, 1.0)
        z0, z1 = g[i - 1], g[i]
        out = z0 + w * (z1 - z0)
        return np.where(np.isnan(c[:, :1]), np.nan, out)

    def ppf(self, q) -> np.ndarray:
        """Quantile at level ``q`` (scalar or one level per row); interp and quant only."""
        self._need("interp", "quant", op="ppf")
        qq = np.broadcast_to(np.asarray(q, np.float64), (self._n,))
        return self._ppf2d(np.ascontiguousarray(qq)[:, None])[:, 0]

    def sample(self, n_per: int, seed: int) -> np.ndarray:
        """``n_per`` draws per row, float64 of shape (n, n_per), deterministic for ``seed``."""
        if n_per < 1:
            raise ValidationError("n_per must be >= 1")
        rng = np.random.default_rng(seed)
        if self.spec.kind == "sample":
            u = rng.random((self._n, n_per))
            c = self._counts[:, None]
            pick = np.minimum((u * c).astype(np.int64), np.maximum(c - 1, 0))
            idx = np.minimum(self._offsets[:-1, None] + pick, max(self._flat.size - 1, 0))
            out = self._flat[idx] if self._flat.size else np.full((self._n, n_per), np.nan)
            return np.where(c > 0, out, np.nan)
        return self._ppf2d(rng.random((self._n, n_per)))
