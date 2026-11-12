"""Composable per-object weights evaluated on Arrow tables.

A :class:`Weight` maps a table (or DataFrame) to a float64 vector of length
``n``. Weights are frozen dataclasses and compose with ``*``. Every result is
checked: right shape, finite, non-negative. Zero weights are allowed.
"""
from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import healpy as hp
import numpy as np
import pandas as pd
import pyarrow as pa

from oneuniverse._core.errors import SchemaError, ValidationError
from oneuniverse._core.healpix import radec_to_pix


def _as_table(table: pa.Table | pd.DataFrame) -> pa.Table:
    if isinstance(table, pd.DataFrame):
        return pa.Table.from_pandas(table, preserve_index=False)
    if isinstance(table, pa.Table):
        return table
    raise ValidationError(f"expected a pyarrow Table or pandas DataFrame, got {type(table).__name__}")


def _col(table: pa.Table, name: str) -> np.ndarray:
    """Column *name* as float64 with nulls as NaN."""
    typ = table[name].type
    if not (pa.types.is_integer(typ) or pa.types.is_floating(typ) or pa.types.is_boolean(typ)):
        raise SchemaError(f"column {name!r} must be numeric or boolean, got {typ}")
    arr = table[name].cast(pa.float64()).combine_chunks()
    return np.asarray(arr.fill_null(np.nan).to_numpy(zero_copy_only=False), dtype=np.float64)


class Weight(ABC):
    """Per-object weight. Subclasses define ``columns`` and ``compute``."""

    @property
    @abstractmethod
    def columns(self) -> frozenset[str]:
        """Table columns this weight reads."""

    @abstractmethod
    def compute(self, table: pa.Table) -> np.ndarray:
        """Raw weights for *table* (columns already checked)."""

    def describe(self) -> str:
        return type(self).__name__

    def __call__(self, table: pa.Table | pd.DataFrame) -> np.ndarray:
        tbl = _as_table(table)
        missing = sorted(self.columns - set(tbl.column_names))
        if missing:
            raise SchemaError(f"{self.describe()}: missing columns {missing}")
        out = np.asarray(self.compute(tbl), dtype=np.float64)
        if out.shape != (tbl.num_rows,):
            raise ValidationError(
                f"{self.describe()}: weight shape {out.shape}, expected ({tbl.num_rows},)")
        bad = int(np.count_nonzero(~np.isfinite(out)))
        if bad:
            raise ValidationError(f"{self.describe()}: {bad} non-finite weight(s)")
        neg = int(np.count_nonzero(out < 0))
        if neg:
            raise ValidationError(f"{self.describe()}: {neg} negative weight(s)")
        return out

    def __mul__(self, other: Weight) -> ProductWeight:
        if not isinstance(other, Weight):
            return NotImplemented
        left = self.factors if isinstance(self, ProductWeight) else (self,)
        right = other.factors if isinstance(other, ProductWeight) else (other,)
        return ProductWeight(left + right)


@dataclass(frozen=True)
class ProductWeight(Weight):
    factors: tuple[Weight, ...]

    @property
    def columns(self) -> frozenset[str]:
        return frozenset().union(*(f.columns for f in self.factors))

    def compute(self, table: pa.Table) -> np.ndarray:
        out = np.ones(table.num_rows)
        for f in self.factors:
            out = out * f(table)
        return out

    def describe(self) -> str:
        return " × ".join(f.describe() for f in self.factors)


@dataclass(frozen=True)
class ColumnWeight(Weight):
    column: str

    @property
    def columns(self) -> frozenset[str]:
        return frozenset({self.column})

    def compute(self, table: pa.Table) -> np.ndarray:
        return _col(table, self.column)

    def describe(self) -> str:
        return f"Column({self.column})"


@dataclass(frozen=True)
class ConstantWeight(Weight):
    value: float = 1.0

    @property
    def columns(self) -> frozenset[str]:
        return frozenset()

    def compute(self, table: pa.Table) -> np.ndarray:
        return np.full(table.num_rows, float(self.value))

    def describe(self) -> str:
        return f"Constant({self.value:g})"


@dataclass(frozen=True)
class InverseVarianceWeight(Weight):
    """w = 1 / sigma^2. NaN or zero sigma follows *missing*; negative sigma raises."""

    err_column: str
    missing: Literal["unit", "zero", "raise"] = "unit"

    def __post_init__(self) -> None:
        if self.missing not in ("unit", "zero", "raise"):
            raise ValidationError(f"missing must be 'unit', 'zero' or 'raise', got {self.missing!r}")

    @property
    def columns(self) -> frozenset[str]:
        return frozenset({self.err_column})

    def compute(self, table: pa.Table) -> np.ndarray:
        sig = _col(table, self.err_column)
        neg = int(np.count_nonzero(sig < 0))
        if neg:
            raise ValidationError(f"{self.describe()}: {neg} negative error(s)")
        bad = np.isnan(sig) | (sig == 0)
        if self.missing == "raise" and bad.any():
            raise ValidationError(f"{self.describe()}: {int(bad.sum())} NaN or zero error(s)")
        out = np.empty_like(sig)
        out[~bad] = 1.0 / sig[~bad] ** 2
        out[bad] = 1.0 if self.missing == "unit" else 0.0
        return out

    def describe(self) -> str:
        return f"InverseVariance({self.err_column})"


@dataclass(frozen=True)
class FKPWeight(Weight):
    """w = 1 / (1 + nbar(z) P0). *nbar* and *P0* are user-supplied numbers.

    *nbar* is a callable of z, or the name of a column holding nbar per object.
    """

    nbar: Callable[[np.ndarray], np.ndarray] | str
    P0: float
    z_column: str = "z"

    def __post_init__(self) -> None:
        if not self.P0 > 0:
            raise ValidationError(f"P0 must be > 0, got {self.P0}")

    @property
    def columns(self) -> frozenset[str]:
        return frozenset({self.nbar if isinstance(self.nbar, str) else self.z_column})

    def compute(self, table: pa.Table) -> np.ndarray:
        if isinstance(self.nbar, str):
            nb = _col(table, self.nbar)
        else:
            nb = np.asarray(self.nbar(_col(table, self.z_column)), dtype=np.float64)
        return 1.0 / (1.0 + nb * self.P0)

    def describe(self) -> str:
        nb = self.nbar if isinstance(self.nbar, str) else getattr(
            self.nbar, "__name__", type(self.nbar).__name__)
        return f"FKP(P0={self.P0:g}, nbar={nb})"


@dataclass(frozen=True, eq=False)
class HealpixMapWeight(Weight):
    """Weight read from a HEALPix map at each object's pixel."""

    map: np.ndarray
    nside: int
    nest: bool = True
    nan: Literal["raise"] | float = "raise"
    ra_column: str = "ra"
    dec_column: str = "dec"

    def __post_init__(self) -> None:
        m = np.array(self.map, dtype=np.float64, copy=True)
        if m.ndim != 1 or m.size != 12 * self.nside**2:
            raise ValidationError(
                f"map has {m.size} pixels, expected 12*nside^2 = {12 * self.nside**2}")
        object.__setattr__(self, "map", m)
        if isinstance(self.nan, str) and self.nan != "raise":
            raise ValidationError(f"nan must be 'raise' or a float, got {self.nan!r}")

    @classmethod
    def from_fits(cls, path: str | Path, *, nest: bool = True, **kwargs) -> HealpixMapWeight:
        """Read a map (first column) in ordering *nest*; UNSEEN pixels become NaN."""
        m = np.asarray(hp.read_map(str(path), nest=nest), dtype=np.float64)
        m[m == hp.UNSEEN] = np.nan
        return cls(m, hp.npix2nside(m.size), nest=nest, **kwargs)

    @property
    def columns(self) -> frozenset[str]:
        return frozenset({self.ra_column, self.dec_column})

    def compute(self, table: pa.Table) -> np.ndarray:
        ra, dec = _col(table, self.ra_column), _col(table, self.dec_column)
        if self.nest:
            pix = radec_to_pix(self.nside, ra, dec)
        else:
            pix = radec_to_pix(self.nside, ra, dec, nest=False)
        out = self.map[pix].copy()
        nan = np.isnan(out)
        if nan.any():
            if self.nan == "raise":
                raise ValidationError(f"{self.describe()}: {int(nan.sum())} object(s) on NaN pixels")
            out[nan] = float(self.nan)
        return out

    def describe(self) -> str:
        return f"HealpixMap(nside={self.nside}, nest={self.nest})"


@dataclass(frozen=True)
class ShearWeight(Weight):
    """w = shape_weight / (R_eff^2 + sigma_e^2).

    metacal: R_eff = (R11 + R22)/2 + R_S. lensfit: R_eff = 1 + m_bias.
    sigma_e^2 is the mean of the squares of the two *sigma_e_columns* when given.
    """

    kind: Literal["metacal", "lensfit"]
    shape_weight_column: str = "shear_weight"
    sigma_e_columns: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("metacal", "lensfit"):
            raise ValidationError(f"kind must be 'metacal' or 'lensfit', got {self.kind!r}")

    @property
    def columns(self) -> frozenset[str]:
        cols = {self.shape_weight_column}
        cols |= {"R11", "R22", "R_S"} if self.kind == "metacal" else {"m_bias"}
        if self.sigma_e_columns is not None:
            cols |= set(self.sigma_e_columns)
        return frozenset(cols)

    def compute(self, table: pa.Table) -> np.ndarray:
        if self.kind == "metacal":
            r = 0.5 * (_col(table, "R11") + _col(table, "R22")) + _col(table, "R_S")
        else:
            r = 1.0 + _col(table, "m_bias")
        den = r**2
        if self.sigma_e_columns is not None:
            a, b = (_col(table, c) for c in self.sigma_e_columns)
            den = den + 0.5 * (a**2 + b**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _col(table, self.shape_weight_column) / den

    def describe(self) -> str:
        return f"Shear({self.kind})"


_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
        "<=": operator.le, ">": operator.gt, ">=": operator.ge}


@dataclass(frozen=True)
class QualityMaskWeight(Weight):
    """1.0 where ``column op threshold`` holds, else 0.0. NaN and null values never pass, whatever the op."""

    column: str
    op: Literal["==", "!=", "<", "<=", ">", ">="]
    threshold: float

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValidationError(f"op must be one of {sorted(_OPS)}, got {self.op!r}")

    @property
    def columns(self) -> frozenset[str]:
        return frozenset({self.column})

    def compute(self, table: pa.Table) -> np.ndarray:
        x = _col(table, self.column)
        with np.errstate(invalid="ignore"):
            return (_OPS[self.op](x, self.threshold) & ~np.isnan(x)).astype(np.float64)

    def describe(self) -> str:
        return f"QualityMask({self.column} {self.op} {self.threshold:g})"
