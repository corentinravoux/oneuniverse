"""Measurement products: point sets, field maps and sightlines.

A Product bundles a catalog (core columns only plus optional extras),
optional randoms, window and Nz. It carries no cosmological quantity: the
consumer chooses the cosmology.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from oneuniverse._core import healpix
from oneuniverse._core.errors import ValidationError
from oneuniverse.measure.atoms import Nz, Window

CATALOG_CORE = ("ra", "dec", "z", "weight", "region_id", "source_row", "partition", "offset")
RANDOMS_CORE = ("ra", "dec", "z", "weight", "region_id")
FORBIDDEN_COLUMNS = frozenset({"comoving_distance", "r_comoving", "dist_mpc", "dist_mpc_h", "chi"})
_FLOAT_COLS = ("ra", "dec", "z", "weight")
_INT_COLS = ("region_id", "source_row", "partition", "offset")


def _check_columns(table: pa.Table, required: tuple[str, ...], label: str) -> None:
    if not isinstance(table, pa.Table):
        raise ValidationError(f"{label} must be a pyarrow Table, got {type(table).__name__}")
    names = set(table.column_names)
    for bad in sorted(FORBIDDEN_COLUMNS & names):
        raise ValidationError(f"{label} column {bad!r} is a cosmology-dependent quantity; "
                              "products are cosmology-free")
    for col in required:
        if col not in names:
            raise ValidationError(f"{label} is missing required column {col!r}")
        t = table.schema.field(col).type
        ok = pa.types.is_floating(t) if col in _FLOAT_COLS else pa.types.is_integer(t)
        if not ok:
            want = "floating" if col in _FLOAT_COLS else "integer"
            raise ValidationError(f"{label} column {col!r} must be {want}, got {t}")


def _normalise_json(value, what: str):
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"{type(o).__name__} is not JSON serialisable")
    try:
        return json.loads(json.dumps(value, default=default, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{what} must be JSON-serialisable: {e}") from e


@dataclass(frozen=True, eq=False)
class Product:
    """Catalog + randoms + window + Nz + extras + provenance.

    extras: name -> ndarray or pa.Table. No length check is made: per-object
    arrays conventionally have first dimension == catalog rows, but global
    arrays (e.g. a per-bin vector) are allowed; the producer documents which.
    provenance is normalised to JSON-native types.
    """
    catalog: pa.Table
    randoms: pa.Table | None = None
    window: Window | None = None
    nz: Nz | None = None
    extras: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    kind: ClassVar[str] = "pointset"
    _allow_empty: ClassVar[bool] = False

    def __post_init__(self):
        _check_columns(self.catalog, CATALOG_CORE, "catalog")
        if self.catalog.num_rows == 0 and not self._allow_empty:
            raise ValidationError(
                "catalog is empty (0 rows after selection); refusing to build a Product")
        if self.randoms is not None:
            _check_columns(self.randoms, RANDOMS_CORE, "randoms")
            if self.randoms.num_rows == 0:
                raise ValidationError("randoms are empty (0 rows)")
        extras = {}
        for k, v in dict(self.extras).items():
            if not isinstance(k, str) or not k.isidentifier():
                raise ValidationError(f"extras key must be an identifier, got {k!r}")
            if k in FORBIDDEN_COLUMNS:
                raise ValidationError(f"extras key {k!r} is a cosmology-dependent quantity")
            if isinstance(v, np.ndarray):
                extras[k] = v.copy()
            elif isinstance(v, pa.Table):
                extras[k] = v
            else:
                raise TypeError(
                    f"extras[{k!r}] must be a numpy array or pyarrow Table, got {type(v).__name__}")
        object.__setattr__(self, "extras", extras)
        object.__setattr__(self, "provenance", _normalise_json(dict(self.provenance), "provenance"))

    @property
    def n(self) -> int:
        return self.catalog.num_rows

    def with_extra(self, name: str, value) -> "Product":
        import dataclasses
        return dataclasses.replace(self, extras={**self.extras, name: value})


@dataclass(frozen=True, eq=False, kw_only=True)
class FieldMap(Product):
    """Pixelised field (e.g. convergence map) with its mask.

    The catalog may be a 0-row placeholder but still carries CATALOG_CORE.
    """
    values: np.ndarray
    mask: np.ndarray
    nside: int
    nest: bool = True

    kind: ClassVar[str] = "fieldmap"
    _allow_empty: ClassVar[bool] = True

    def __post_init__(self):
        super().__post_init__()
        healpix._check_nside(self.nside)
        npix = 12 * self.nside ** 2
        for name in ("values", "mask"):
            a = np.array(getattr(self, name), dtype=np.float64)
            if a.shape != (npix,):
                raise ValidationError(f"{name} must have length {npix}, got shape {a.shape}")
            object.__setattr__(self, name, a)
        object.__setattr__(self, "nest", bool(self.nest))


_SIGHTLINE_LISTS = ("wavelength", "delta", "pixel_weight")


@dataclass(frozen=True, eq=False)
class Sightline(Product):
    """Forest sightlines: one row per line of sight with list columns."""

    kind: ClassVar[str] = "sightline"

    def __post_init__(self):
        super().__post_init__()
        names = set(self.catalog.column_names)
        missing = [c for c in ("sightline_id", *_SIGHTLINE_LISTS) if c not in names]
        if missing:
            raise ValidationError(f"sightline catalog is missing required columns {missing}")
        if not pa.types.is_integer(self.catalog.schema.field("sightline_id").type):
            raise ValidationError("sightline_id must be integer")
        for col in _SIGHTLINE_LISTS:
            t = self.catalog.schema.field(col).type
            if not ((pa.types.is_list(t) or pa.types.is_large_list(t))
                    and pa.types.is_floating(t.value_type)):
                raise ValidationError(f"sightline column {col!r} must be a list of floats, got {t}")

    @property
    def n_pixels(self) -> int:
        lengths = pc.list_value_length(self.catalog.column("delta"))
        return int(pc.sum(lengths).as_py() or 0)
