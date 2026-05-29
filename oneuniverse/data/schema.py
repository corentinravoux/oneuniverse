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
    # Phase 16 observational-metadata annotations. All optional. No
    # cosmology assumption — only what the survey publishes.
    frame: Optional[str] = None
    # One of: "heliocentric", "cmb", "lsr", "galactocentric", "icrs",
    # "galactic", "ecliptic", "observer", "AB", "Vega", … Loader-defined.
    epoch: Optional[float] = None
    # Decimal-year epoch for position columns (e.g. 2016.0 for GAIA DR3).
    wavelength_convention: Optional[str] = None
    # "vacuum" | "air" for spectral columns; None for non-spectral.
    nullable: bool = False
    # True if the column is allowed to contain NaN / missing rows.


# ── Column groups ─────────────────────────────────────────────────────────

CORE_COLUMNS: Tuple[ColumnDef, ...] = (
    # Position — float64 for sub-arcsecond precision
    ColumnDef("ra", "f8", "deg", "Right ascension ICRS J2000"),
    ColumnDef("dec", "f8", "deg", "Declination ICRS J2000"),
    # Redshift — required string tag: "spec" | "phot" | "pv" | "none"
    ColumnDef("z", "f4", "", "Best available redshift"),
    ColumnDef("z_type", "U8", "", "Redshift type: spec | phot | pv | none"),
    ColumnDef("z_err", "f4", "", "Best-redshift 1-sigma uncertainty"),
    # Identifiers
    ColumnDef("galaxy_id", "i8", "", "Unique ID in oneuniverse"),
    ColumnDef("survey_id", "U32", "", "Source survey name"),
    # Phase 21: optional survey-published composite ID preserved
    # verbatim (PLATE-MJD-FIBERID, KIDS_TILE+SeqNr, GAIA source_id, …).
    ColumnDef("composite_id", "U64", "",
              "Survey-published composite ID (PLATE-MJD-FIBERID, "
              "KIDS_TILE+SeqNr, GAIA source_id, …); optional",
              required=False),
    # Technical / bookkeeping
    ColumnDef("_original_row_index", "i8", "",
              "Row index in the original survey file (for audit/join-back)"),
    ColumnDef("_healpix32", "i4", "",
              "HEALPix NESTED index at NSIDE=32 (spatial partition key)"),
)

# Allowed tags for the required ``z_type`` CORE column.
Z_TYPE_VALUES: Tuple[str, ...] = ("spec", "phot", "phot_pdf", "pv", "none")

SPECTROSCOPIC_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("z_spec", "f4", "", "Spectroscopic redshift"),
    ColumnDef("z_spec_err", "f4", "", "Spectroscopic redshift uncertainty"),
    ColumnDef("z_helio", "f4", "", "Heliocentric redshift", required=False,
              frame="heliocentric"),
    ColumnDef("z_cmb", "f4", "", "CMB-frame redshift", required=False,
              frame="cmb"),
    ColumnDef("cz_cmb", "f4", "km/s", "CMB-frame recession velocity",
              required=False, frame="cmb"),
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

QSO_COLUMNS: Tuple[ColumnDef, ...] = (
    # Classification
    ColumnDef("is_qso", "i1", "", "QSO classification flag (1=confirmed QSO)"),
    ColumnDef("source_z", "U16", "", "Source of best redshift (VI, PIPE, etc.)", required=False),
    ColumnDef("z_pipe", "f8", "", "Pipeline redshift", required=False),
    ColumnDef("z_pca", "f8", "", "PCA redshift", required=False),
    ColumnDef("z_vi", "f8", "", "Visual inspection redshift", required=False),
    ColumnDef("z_conf", "i1", "", "Visual inspection confidence (0-3)", required=False),
    ColumnDef("zwarning", "i4", "", "Pipeline ZWARNING flag (0=good)", required=False),
    # BAL properties
    ColumnDef("bal_prob", "f4", "", "BAL probability (0-1, -1=not assessed)", required=False),
    ColumnDef("bi_civ", "f8", "km/s", "Balnicity index CIV", required=False),
    # Emission line redshifts
    ColumnDef("z_mgii", "f8", "", "MgII emission line redshift", required=False),
    ColumnDef("z_ciii", "f8", "", "CIII] emission line redshift", required=False),
    ColumnDef("z_civ", "f8", "", "CIV emission line redshift", required=False),
    ColumnDef("z_lya", "f8", "", "Lyman-alpha emission line redshift", required=False),
    # Photometry (ugriz PSF magnitudes)
    ColumnDef("psfmag_u", "f4", "mag", "PSF magnitude u-band", required=False),
    ColumnDef("psfmag_g", "f4", "mag", "PSF magnitude g-band", required=False),
    ColumnDef("psfmag_r", "f4", "mag", "PSF magnitude r-band", required=False),
    ColumnDef("psfmag_i", "f4", "mag", "PSF magnitude i-band", required=False),
    ColumnDef("psfmag_z", "f4", "mag", "PSF magnitude z-band", required=False),
    ColumnDef("extinction_u", "f4", "mag", "Galactic extinction u-band", required=False),
    ColumnDef("extinction_g", "f4", "mag", "Galactic extinction g-band", required=False),
    ColumnDef("extinction_r", "f4", "mag", "Galactic extinction r-band", required=False),
    ColumnDef("extinction_i", "f4", "mag", "Galactic extinction i-band", required=False),
    ColumnDef("extinction_z", "f4", "mag", "Galactic extinction z-band", required=False),
    # Observation metadata
    ColumnDef("plate", "i4", "", "SDSS plate number", required=False),
    ColumnDef("mjd", "i4", "d", "Modified Julian Date of observation", required=False),
    ColumnDef("fiberid", "i2", "", "SDSS fiber ID", required=False),
    ColumnDef("sn_median", "f4", "", "Median S/N per pixel", required=False),
    # DLA (stored as separate columns for up to N_DLA absorbers)
    ColumnDef("n_dla", "i1", "", "Number of DLA systems detected", required=False),
)

PROBABILISTIC_REDSHIFT_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef(
        "z_pdf_kind", "U8", "",
        "PDF parameterisation tag: interp | quant | mixmod",
    ),
    ColumnDef(
        "z_pdf_values", "f4", "",
        "PDF values: interp p(z), quant z(q), or mixmod component means",
    ),
    ColumnDef(
        "z_pdf_sigma", "f4", "",
        "mixmod only: component std devs", required=False,
    ),
    ColumnDef(
        "z_pdf_weights", "f4", "",
        "mixmod only: component weights", required=False,
    ),
    ColumnDef("z_pdf_mean", "f4", "", "PDF first moment", required=False),
    ColumnDef("z_pdf_std", "f4", "", "PDF standard deviation", required=False),
    ColumnDef("z_pdf_median", "f4", "", "PDF median", required=False),
    ColumnDef("z_pdf_mode", "f4", "", "PDF mode", required=False),
    ColumnDef("z_pdf_l68", "f4", "", "PDF 16th percentile", required=False),
    ColumnDef("z_pdf_u68", "f4", "", "PDF 84th percentile", required=False),
)


SHEAR_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("e1", "f4", "", "First shear component", required=False),
    ColumnDef("e2", "f4", "", "Second shear component", required=False),
    ColumnDef("e1_err", "f4", "", "1σ on e1", required=False),
    ColumnDef("e2_err", "f4", "", "1σ on e2", required=False),
    ColumnDef("R11", "f4", "", "Metacal response ∂e1/∂γ1", required=False),
    ColumnDef("R22", "f4", "", "Metacal response ∂e2/∂γ2", required=False),
    ColumnDef("R12", "f4", "", "Metacal off-diagonal response", required=False),
    ColumnDef("R21", "f4", "", "Metacal off-diagonal response", required=False),
    ColumnDef("R_S", "f4", "", "Selection response", required=False),
    ColumnDef("m_bias", "f4", "", "Multiplicative bias (lensfit)", required=False),
    ColumnDef("c1_bias", "f4", "", "Additive bias e1 (lensfit)", required=False),
    ColumnDef("c2_bias", "f4", "", "Additive bias e2 (lensfit)", required=False),
    ColumnDef("shear_weight", "f4", "", "Per-object shape weight", required=False),
)


SNIA_COLUMNS: Tuple[ColumnDef, ...] = (
    ColumnDef("z_cmb", "f4", "", "CMB-frame redshift", frame="cmb"),
    ColumnDef("mu", "f4", "mag", "Distance modulus"),
    ColumnDef("mu_err", "f4", "mag", "Distance modulus uncertainty"),
    ColumnDef("x1", "f4", "", "SALT2 stretch parameter", required=False),
    ColumnDef("x1_err", "f4", "", "SALT2 stretch uncertainty", required=False),
    ColumnDef("c", "f4", "", "SALT2 color parameter", required=False),
    ColumnDef("c_err", "f4", "", "SALT2 color uncertainty", required=False),
    ColumnDef("mb", "f4", "mag", "SALT2 peak B-band magnitude", required=False),
    ColumnDef("mb_err", "f4", "mag", "SALT2 peak B-band magnitude uncertainty", required=False),
)

# Lookup by group name
COLUMN_GROUPS: Dict[str, Tuple[ColumnDef, ...]] = {
    "core": CORE_COLUMNS,
    "spectroscopic": SPECTROSCOPIC_COLUMNS,
    "photometric": PHOTOMETRIC_COLUMNS,
    "peculiar_velocity": PV_COLUMNS,
    "qso": QSO_COLUMNS,
    "snia": SNIA_COLUMNS,
    "probabilistic_redshift": PROBABILISTIC_REDSHIFT_COLUMNS,
    "shear": SHEAR_COLUMNS,
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
