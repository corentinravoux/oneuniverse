"""
oneuniverse.data.schema
~~~~~~~~~~~~~~~~~~~~~~~
Column definitions for the oneuniverse standardized catalog format.

Every survey loader maps its native columns into this schema. Columns are
grouped into a *core* set (mandatory for all surveys) and *extension* sets
(spectroscopic, photometric, peculiar-velocity) that are survey-type
dependent.

The schema is defined as plain dataclasses — no heavy ORM, no metaclass
magic. A loader declares which groups it provides, and validation checks
that the mandatory columns are present with correct dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Column descriptor ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnDef:
    """Definition of a single catalog column."""

    name: str
    dtype: str  # numpy dtype string: "f8", "f4", "i8", "i1", "U32", …
    unit: str  # astropy-compatible unit string, "" if dimensionless
    description: str
    required: bool = True  # within its group


# ── Column groups ─────────────────────────────────────────────────────────

CORE_COLUMNS: Tuple[ColumnDef, ...] = (
    # Position — float64 for sub-arcsecond precision
    ColumnDef("ra", "f8", "deg", "Right ascension ICRS J2000"),
    ColumnDef("dec", "f8", "deg", "Declination ICRS J2000"),
    # Redshift
    ColumnDef("z", "f4", "", "Best available redshift"),
    ColumnDef("z_type", "i1", "", "Redshift type: 0=spec, 1=phot, 2=other", required=False),
    # Identifiers
    ColumnDef("galaxy_id", "i8", "", "Unique ID in oneuniverse"),
    ColumnDef("survey_id", "U32", "", "Source survey name"),
)

SPECTROSCOPIC_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("z_spec", "f4", "", "Spectroscopic redshift"),
    ColumnDef("z_spec_err", "f4", "", "Spectroscopic redshift uncertainty"),
    ColumnDef("z_helio", "f4", "", "Heliocentric redshift", required=False),
    ColumnDef("z_cmb", "f4", "", "CMB-frame redshift", required=False),
    ColumnDef("cz_cmb", "f4", "km/s", "CMB-frame recession velocity", required=False),
    ColumnDef("w_fkp", "f4", "", "FKP weight", required=False),
    ColumnDef("w_comp", "f4", "", "Angular completeness weight", required=False),
    ColumnDef("w_cp", "f4", "", "Fiber collision weight", required=False),
    ColumnDef("w_noz", "f4", "", "Redshift failure weight", required=False),
    ColumnDef("w_sys", "f4", "", "Imaging systematic weight", required=False),
    ColumnDef("w_tot", "f4", "", "Total analysis weight", required=False),
)

PHOTOMETRIC_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("z_phot", "f4", "", "Photometric redshift"),
    ColumnDef("z_phot_err", "f4", "", "Photo-z 1-sigma uncertainty"),
    ColumnDef("z_phot_l68", "f4", "", "Photo-z 68% lower bound", required=False),
    ColumnDef("z_phot_u68", "f4", "", "Photo-z 68% upper bound", required=False),
    ColumnDef("odds", "f4", "", "Photo-z quality (0-1)", required=False),
)

PV_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("v_pec", "f4", "km/s", "Peculiar velocity"),
    ColumnDef("v_pec_err", "f4", "km/s", "Peculiar velocity uncertainty"),
    ColumnDef("mu", "f4", "mag", "Distance modulus", required=False),
    ColumnDef("mu_err", "f4", "mag", "Distance modulus uncertainty", required=False),
    ColumnDef("mu_method", "U8", "", "TF/FP/SBF/SNIa", required=False),
)

# Lookup by group name
COLUMN_GROUPS: Dict[str, Tuple[ColumnDef, ...]] = {
    "core": CORE_COLUMNS,
    "spectroscopic": SPECTROSCOPIC_COLUMNS,
    "photometric": PHOTOMETRIC_COLUMNS,
    "peculiar_velocity": PV_COLUMNS,
}


# ── Validation ────────────────────────────────────────────────────────────


def get_required_columns(groups: List[str]) -> List[str]:
    """Return names of all required columns for the given groups."""
    required = []
    for group in groups:
        for col in COLUMN_GROUPS[group]:
            if col.required:
                required.append(col.name)
    return required


def get_all_columns(groups: List[str]) -> Dict[str, ColumnDef]:
    """Return a {name: ColumnDef} dict for all columns in the given groups."""
    columns = {}
    for group in groups:
        for col in COLUMN_GROUPS[group]:
            columns[col.name] = col
    return columns


def validate_dataframe(df, groups: List[str]) -> List[str]:
    """Check that *df* satisfies the schema for *groups*.

    Returns a list of warning strings (empty if valid). Raises ValueError
    if a required column is missing.
    """
    warnings = []
    all_cols = get_all_columns(groups)

    # Check required columns exist
    for name, coldef in all_cols.items():
        if coldef.required and name not in df.columns:
            raise ValueError(
                f"Required column '{name}' (group '{_group_of(name, groups)}') "
                f"is missing from the DataFrame."
            )

    # Check dtypes (soft warning, not error — numeric promotion is fine)
    for name in df.columns:
        if name in all_cols:
            expected = np.dtype(all_cols[name].dtype)
            actual = df[name].dtype
            # pandas stores strings as object dtype — skip for string columns
            if expected.kind == "U" and actual == np.dtype("O"):
                continue
            if not np.can_cast(actual, expected, casting="same_kind"):
                warnings.append(
                    f"Column '{name}': expected dtype compatible with "
                    f"{expected}, got {actual}"
                )

    return warnings


def _group_of(col_name: str, groups: List[str]) -> Optional[str]:
    """Return the group name that contains *col_name*."""
    for group in groups:
        for col in COLUMN_GROUPS[group]:
            if col.name == col_name:
                return group
    return None
