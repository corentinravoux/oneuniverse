"""OUF 3.0 column contracts, redshift-type registry and stable object ids."""
from __future__ import annotations

import hashlib
import re
from enum import Enum

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import SchemaError, ValidationError


class Geometry(str, Enum):
    """Row geometry of a dataset."""

    POINT = "point"
    SIGHTLINE = "sightline"
    LIGHTCURVE = "lightcurve"


# ---------------------------------------------------------------- z types
Z_TYPES: frozenset[str] = frozenset({"spec", "phot", "phot_pdf", "pv", "none"})
_registered_z_types: set[str] = set(Z_TYPES)
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def z_types() -> frozenset[str]:
    """All currently valid ``z_type`` values (built-in plus registered)."""
    return frozenset(_registered_z_types)


def register_z_type(name: str) -> None:
    """Add a redshift type; raises ValidationError if invalid or already known."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValidationError(
            f"z_type name must be a non-empty lowercase identifier, got {name!r}")
    if name in _registered_z_types:
        raise ValidationError(f"z_type {name!r} is already registered")
    _registered_z_types.add(name)


# ---------------------------------------------------------------- schemas
_Z_TYPE = pa.dictionary(pa.int8(), pa.string())
_SURVEY = pa.dictionary(pa.int16(), pa.string())

CORE_POINT = pa.schema([
    ("ra", pa.float64()), ("dec", pa.float64()), ("z", pa.float64()),
    ("z_type", _Z_TYPE), ("z_err", pa.float32()),
    ("object_id", pa.int64()), ("survey", _SURVEY),
    ("source_row", pa.int64()), ("hpix32", pa.int32()),
])

CORE_SIGHTLINE = pa.schema([
    ("sightline_id", pa.int64()), ("ra", pa.float64()), ("dec", pa.float64()),
    ("z", pa.float64()), ("survey", _SURVEY), ("source_row", pa.int64()),
    ("hpix32", pa.int32()),
    ("wavelength", pa.list_(pa.float32())), ("delta", pa.list_(pa.float32())),
    ("weight", pa.list_(pa.float32())),
])

CORE_LIGHTCURVE = pa.schema(list(CORE_POINT) + [
    ("mjd", pa.list_(pa.float64())), ("flux", pa.list_(pa.float32())),
    ("flux_err", pa.list_(pa.float32())),
    ("band", pa.list_(pa.dictionary(pa.int8(), pa.string()))),
])

_CORE = {
    Geometry.POINT: CORE_POINT,
    Geometry.SIGHTLINE: CORE_SIGHTLINE,
    Geometry.LIGHTCURVE: CORE_LIGHTCURVE,
}


def _f4(*names: str) -> list[tuple[str, pa.DataType]]:
    return [(n, pa.float32()) for n in names]


OPTIONAL_COLUMNS: dict[str, pa.Schema] = {
    "spectroscopic": pa.schema(_f4(
        "z_spec", "z_spec_err", "z_helio", "z_cmb", "cz_cmb",
        "w_fkp", "w_comp", "w_cp", "w_noz", "w_sys", "w_tot")),
    "photometric": pa.schema(_f4(
        "z_phot", "z_phot_err", "z_phot_l68", "z_phot_u68", "odds")),
    "peculiar_velocity": pa.schema(
        _f4("v_pec", "v_pec_err", "mu", "mu_err") + [("mu_method", pa.string())]),
    "snia": pa.schema(_f4(
        "z_cmb", "mu", "mu_err", "x1", "x1_err", "c", "c_err", "mb", "mb_err")),
    "shear": pa.schema(_f4(
        "e1", "e2", "e1_err", "e2_err", "R11", "R22", "R12", "R21", "R_S",
        "m_bias", "c1_bias", "c2_bias", "shear_weight")),
    "pdf": pa.schema(
        [("z_pdf_values", pa.list_(pa.float32())),
         ("z_pdf_sigma", pa.list_(pa.float32())),
         ("z_pdf_weights", pa.list_(pa.float32()))]
        + _f4("z_pdf_mean", "z_pdf_std", "z_pdf_median", "z_pdf_mode",
              "z_pdf_l68", "z_pdf_u68")),
}


def core_schema(geometry: Geometry) -> pa.Schema:
    """Core column schema of *geometry*."""
    return _CORE[Geometry(geometry)]


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t) \
        or pa.types.is_fixed_size_list(t)


def _is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def _is_dict_of_text(t: pa.DataType) -> bool:
    return pa.types.is_dictionary(t) and _is_text(t.value_type)


def _family_ok(expected: pa.DataType, actual: pa.DataType) -> bool:
    if pa.types.is_dictionary(expected):
        return _is_dict_of_text(actual) or _is_text(actual)
    if _is_list(expected):
        return _is_list(actual)
    if pa.types.is_floating(expected):
        return pa.types.is_floating(actual)
    if pa.types.is_integer(expected):
        return pa.types.is_integer(actual)
    if _is_text(expected):
        return _is_text(actual) or _is_dict_of_text(actual)
    return expected == actual


def validate_schema(schema: pa.Schema, geometry: Geometry) -> None:
    """Check *schema* against the core contract and typed optional columns.

    Raises SchemaError on a missing core column or a wrong type family.
    Extra columns are accepted; known optional names must have the right family.
    """
    core = core_schema(geometry)
    missing = [n for n in core.names if n not in schema.names]
    if missing:
        raise SchemaError(f"missing core column(s): {', '.join(missing)}")
    for f in core:
        actual = schema.field(f.name).type
        if not _family_ok(f.type, actual):
            raise SchemaError(
                f"column {f.name!r} has type {actual}, expected {f.type} family")
        if f.name in ("z_type", "survey") and pa.types.is_dictionary(actual) \
                and not _is_text(actual.value_type):
            raise SchemaError(
                f"column {f.name!r} dictionary values must be strings, "
                f"got {actual.value_type}")
    core_names = set(core.names)
    for group in OPTIONAL_COLUMNS.values():
        for f in group:
            if f.name in schema.names and f.name not in core_names:
                actual = schema.field(f.name).type
                if not _family_ok(f.type, actual):
                    raise SchemaError(
                        f"optional column {f.name!r} has type {actual}, "
                        f"expected {f.type} family")


def validate_z_type_values(array: pa.Array | pa.ChunkedArray) -> None:
    """Raise SchemaError naming the first ``z_type`` value outside ``z_types()``."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if pa.types.is_dictionary(array.type):
        # only used dictionary entries count
        values = array.dictionary
        present = {values[i].as_py() for i in set(array.indices.drop_null().to_pylist())}
    else:
        present = set(array.drop_null().unique().to_pylist())
    bad = sorted(v for v in present if v not in _registered_z_types)
    if bad:
        raise SchemaError(
            f"z_type value {bad[0]!r} not in registered z_types "
            f"{sorted(_registered_z_types)}")


# ---------------------------------------------------------------- ids
def stable_object_id(survey: str, source_row: np.ndarray) -> np.ndarray:
    """``(hash32(survey) << 32) | source_row`` as int64 (computed in uint64)."""
    rows = np.asarray(source_row)
    if rows.size and (rows.min() < 0 or rows.max() >= 2**32):
        raise ValidationError("source_row must be in [0, 2**32)")
    h = np.uint64(int(hashlib.sha256(survey.encode()).hexdigest()[:8], 16))
    out = (h << np.uint64(32)) | rows.astype(np.uint64)
    return out.view(np.int64)


def cast_to_core(table: pa.Table, geometry: Geometry) -> pa.Table:
    """Cast core columns to their contract types; other columns untouched."""
    core = core_schema(geometry)
    for f in core:
        if f.name in table.schema.names:
            i = table.schema.get_field_index(f.name)
            col = table.column(i)
            if col.type != f.type:
                if pa.types.is_dictionary(f.type) and not pa.types.is_dictionary(col.type):
                    col = col.dictionary_encode().cast(f.type)
                else:
                    col = col.cast(f.type)
            table = table.set_column(i, f, col)
    return table
