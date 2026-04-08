"""
oneuniverse.data.format_spec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Formal specification of the **oneuniverse file format** (OUF).

The format is designed to store any astronomical survey data in a
uniform Parquet-based layout regardless of data geometry.  Three
geometries are supported:

    POINT       One row per object (galaxy, QSO, SN).
                Standard catalog — scalar properties only.
                Example: eBOSS DR16Q, SDSS MGS, CosmicFlows-4.

    SIGHTLINE   One row per pixel/element in a 1-D spectrum or
                sightline, grouped by ``sightline_id``.  Each sightline
                has per-object metadata (ra, dec, z_source) plus
                per-pixel arrays (wavelength, flux, weight, …).
                Example: Lyman-alpha forest deltas, DESI coadded spectra.

    HEALPIX     One row per HEALPix pixel, representing a sky map at
                a fixed ``nside``.  No per-object redshift — the data
                is a 2-D projection on the sphere.
                Example: CMB lensing kappa maps, survey depth maps,
                photometric source density maps, DES/LSST image tile
                metadata catalogs.

All three geometries share the same on-disk layout:

::

    {survey_path}/oneuniverse/
    ├── manifest.json           ← metadata, geometry, linkback info
    ├── objects.parquet         ← object/sightline/tile metadata table
    │                              (always present; one row per logical entity)
    ├── part_0000.parquet       ← data partitions (POINT or SIGHTLINE pixels)
    ├── part_0001.parquet
    └── ...

For **POINT** geometry, the ``objects.parquet`` file is not written
separately — the ``part_*.parquet`` files contain everything (backward
compatible with the existing converter).

For **SIGHTLINE** geometry:
    - ``objects.parquet``: one row per sightline (ra, dec, z_source,
      sightline_id, n_pixels, mean_snr, …)
    - ``part_*.parquet``:  one row per pixel (sightline_id, loglam,
      delta, weight, …), partitioned by row count

For **HEALPIX** geometry:
    - ``objects.parquet``: one row per pixel (healpix_index, value,
      weight, …), partitioned by row count.  No separate part files
      needed unless the map is very large.

The ``manifest.json`` always declares the geometry, enabling any reader
to handle the data correctly without prior knowledge of the survey.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


# ── Data geometry ────────────────────────────────────────────────────────


class DataGeometry(str, Enum):
    """Geometry of the data stored in an oneuniverse directory.

    The geometry determines the structure of the Parquet files and which
    tables are present.
    """

    POINT = "point"
    """One row per object. Standard catalog (galaxies, QSOs, SNe).
    Columns: ra, dec, z, plus scalar properties.
    Tables: part_*.parquet only."""

    SIGHTLINE = "sightline"
    """One row per spectral pixel, grouped by sightline_id.
    Tables: objects.parquet (per-sightline metadata)
          + part_*.parquet (per-pixel data)."""

    HEALPIX = "healpix"
    """One row per HEALPix pixel. Sky maps and image tile catalogs.
    Tables: part_*.parquet (per-pixel data).
    No per-object redshift."""


# ── Format version ───────────────────────────────────────────────────────


# Current format version.  Bump when the manifest schema changes.
FORMAT_VERSION: str = "1.0.0"


# ── Per-geometry column requirements ─────────────────────────────────────


# Minimum columns that MUST be present in the objects/data tables
# for each geometry.  Surveys add their own columns on top.

POINT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "ra", "dec", "z", "galaxy_id", "survey_id",
)

SIGHTLINE_OBJECT_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "sightline_id",     # int64, unique per sightline
    "ra",               # float64, degrees (ICRS)
    "dec",              # float64, degrees (ICRS)
    "z_source",         # float32, source redshift (e.g. QSO redshift)
    "survey_id",        # string
    "n_pixels",         # int32, number of pixels in this sightline
)

SIGHTLINE_DATA_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "sightline_id",     # int64, foreign key to objects table
    "loglam",           # float32, log10(wavelength/Angstrom)
    "delta",            # float32, flux contrast δ_F = F/F̄ - 1
    "weight",           # float32, inverse variance weight
)

HEALPIX_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "healpix_index",    # int64, pixel index (NESTED or RING)
    "value",            # float32 or float64, the map quantity
)

GEOMETRY_COLUMNS: Dict[DataGeometry, Dict[str, Tuple[str, ...]]] = {
    DataGeometry.POINT: {
        "data": POINT_REQUIRED_COLUMNS,
    },
    DataGeometry.SIGHTLINE: {
        "objects": SIGHTLINE_OBJECT_REQUIRED_COLUMNS,
        "data": SIGHTLINE_DATA_REQUIRED_COLUMNS,
    },
    DataGeometry.HEALPIX: {
        "data": HEALPIX_REQUIRED_COLUMNS,
    },
}


# ── Partition norms ──────────────────────────────────────────────────────


# Row counts per partition file, tuned per geometry for ~20–30 MB files.
DEFAULT_PARTITION_ROWS: Dict[DataGeometry, int] = {
    DataGeometry.POINT:     200_000,    # ~25 scalar columns → ~20 MB
    DataGeometry.SIGHTLINE: 2_000_000,  # ~4 float columns per pixel → ~25 MB
    DataGeometry.HEALPIX:   500_000,    # ~2-3 float columns per pixel → ~10 MB
}

COMPRESSION: str = "zstd"

ONEUNIVERSE_SUBDIR: str = "oneuniverse"
MANIFEST_FILENAME: str = "manifest.json"
OBJECTS_FILENAME: str = "objects.parquet"
ORIGINAL_INDEX_COL: str = "_original_row_index"


# ── Validation helpers ───────────────────────────────────────────────────


def validate_manifest(manifest: Dict) -> List[str]:
    """Validate a manifest dict against the format spec.

    Returns a list of error strings (empty if valid).
    """
    errors = []

    # Required keys
    for key in ("format_version", "geometry", "survey_name", "n_rows",
                "partitions", "compression"):
        if key not in manifest:
            errors.append(f"Missing required key: '{key}'")

    # Geometry
    geo = manifest.get("geometry", "")
    valid_geos = [g.value for g in DataGeometry]
    if geo not in valid_geos:
        errors.append(f"Invalid geometry '{geo}'. Must be one of {valid_geos}")

    # Sightline-specific checks
    if geo == DataGeometry.SIGHTLINE.value:
        if not manifest.get("has_objects_table", False):
            errors.append("SIGHTLINE geometry requires has_objects_table=true")
        if manifest.get("n_sightlines", 0) <= 0:
            errors.append("SIGHTLINE geometry requires n_sightlines > 0")

    # HEALPix-specific checks
    if geo == DataGeometry.HEALPIX.value:
        nside = manifest.get("healpix_nside", 0)
        if nside <= 0:
            errors.append("HEALPIX geometry requires healpix_nside > 0")
        ordering = manifest.get("healpix_ordering", "")
        if ordering not in ("nested", "ring"):
            errors.append(f"healpix_ordering must be 'nested' or 'ring', got '{ordering}'")

    return errors


def validate_columns(
    df_columns: List[str],
    geometry: DataGeometry,
    table_type: str = "data",
) -> List[str]:
    """Check that a DataFrame has the required columns for its geometry.

    Parameters
    ----------
    df_columns : list[str]
        Column names of the DataFrame.
    geometry : DataGeometry
    table_type : str
        ``"data"`` for part_*.parquet, ``"objects"`` for objects.parquet.

    Returns
    -------
    List of missing required column names (empty if valid).
    """
    required = GEOMETRY_COLUMNS[geometry].get(table_type, ())
    return [c for c in required if c not in df_columns]
