"""
oneuniverse.data.converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert survey catalogs from their native format (FITS, CSV, …) into
the standardized **oneuniverse file format** (OUF).

See ``format_spec.py`` for the formal specification of the three
supported geometries (POINT, SIGHTLINE, HEALPIX).

Directory layout after conversion
----------------------------------
::

    {survey_path}/oneuniverse/
    ├── manifest.json               ← metadata + geometry + linkback
    ├── objects.parquet              ← per-object metadata (SIGHTLINE only)
    ├── part_0000.parquet
    ├── part_0001.parquet
    └── ...

Usage
-----
>>> from oneuniverse.data import convert_survey, set_data_root
>>> set_data_root("/data/surveys")
>>> convert_survey("eboss_qso", overwrite=True, qso_only=True)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.data.format_spec import (
    COMPRESSION,
    DEFAULT_PARTITION_ROWS,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    OBJECTS_FILENAME,
    ONEUNIVERSE_SUBDIR,
    ORIGINAL_INDEX_COL,
    DataGeometry,
    validate_columns,
)

logger = logging.getLogger(__name__)


# ── Convert: POINT geometry ──────────────────────────────────────────────


def convert_survey(
    survey_name: str,
    data_root: Optional[str | Path] = None,
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    overwrite: bool = False,
    output_dir: Optional[str | Path] = None,
    raw_path: Optional[str | Path] = None,
    **loader_kwargs: Any,
) -> Path:
    """Convert a registered survey to oneuniverse POINT format.

    Parameters
    ----------
    survey_name : str
        Registered survey name (e.g. ``"eboss_qso"``).
    data_root : str or Path or None
        Override data root (default: use global setting).
    partition_rows : int or None
        Rows per Parquet file. None = use geometry default (200K for POINT).
    compression : str
        Parquet compression codec (default: ``"zstd"``).
    overwrite : bool
        If True, remove existing oneuniverse/ dir before converting.
    **loader_kwargs
        Passed to the survey loader (e.g. ``qso_only=True``).

    Returns
    -------
    Path to the oneuniverse/ output directory.
    """
    from oneuniverse.data._config import resolve_survey_path, set_data_root
    from oneuniverse.data._registry import get_loader

    if data_root is not None:
        set_data_root(data_root)

    loader = get_loader(survey_name)
    config = loader.config

    if raw_path is not None:
        rp = Path(raw_path).expanduser().resolve()
        survey_path = rp.parent if rp.is_file() else rp
    else:
        survey_path = resolve_survey_path(
            config.survey_type, config.name, config.data_subpath,
        )
        if survey_path is None:
            raise FileNotFoundError(
                f"Cannot resolve data path for '{survey_name}'. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_root= or raw_path=."
            )

    out_base = Path(output_dir) if output_dir is not None else survey_path
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = _prepare_output_dir(out_base, overwrite)
    geometry = DataGeometry.POINT
    if partition_rows is None:
        partition_rows = DEFAULT_PARTITION_ROWS[geometry]

    # Load the full DataFrame via the loader
    logger.info("Loading %s via loader...", survey_name)
    if raw_path is not None:
        loader_kwargs.setdefault("data_path", survey_path)
    df = loader.load(validate=False, force_native=True, **loader_kwargs)

    # Add original row index for linkback
    df[ORIGINAL_INDEX_COL] = np.arange(len(df), dtype=np.int64)

    # Write partitions
    part_files = _write_partitions(df, out_dir, partition_rows, compression)

    # Write manifest
    manifest = {
        "format_version": FORMAT_VERSION,
        "oneuniverse_version": "0.1.0",
        "survey_name": config.name,
        "survey_type": config.survey_type,
        "geometry": geometry.value,
        "original_files": [config.data_filename],
        "original_format": config.data_format,
        "original_n_rows": _count_original_rows(survey_path, config),
        "n_rows": len(df),
        "n_objects": len(df),
        "n_partitions": len(part_files),
        "partition_rows": partition_rows,
        "compression": compression,
        "has_objects_table": False,
        "partitions": part_files,
        "data_columns": list(df.columns),
        "object_columns": [],
        "conversion_kwargs": loader_kwargs,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    _write_manifest(out_dir, manifest)
    _log_summary(out_dir, survey_path, config, manifest, part_files)

    return out_dir


# ── Convert: SIGHTLINE geometry ──────────────────────────────────────────


def convert_sightlines(
    objects_df: pd.DataFrame,
    data_df: pd.DataFrame,
    survey_path: Path,
    survey_name: str,
    survey_type: str = "lya_forest",
    original_files: Optional[List[str]] = None,
    original_format: str = "fits",
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    overwrite: bool = False,
    sightline_id_column: str = "sightline_id",
    **extra_manifest: Any,
) -> Path:
    """Convert sightline data (e.g. Lya forest) to oneuniverse format.

    This writes a two-table layout:
    - ``objects.parquet``: one row per sightline (ra, dec, z_source, …)
    - ``part_*.parquet``:  one row per pixel (sightline_id, loglam, delta, …)

    Parameters
    ----------
    objects_df : pd.DataFrame
        Per-sightline metadata. Must contain: sightline_id, ra, dec, z_source.
    data_df : pd.DataFrame
        Per-pixel data. Must contain: sightline_id, loglam, delta, weight.
    survey_path : Path
        Directory where ``oneuniverse/`` will be created.
    survey_name : str
        Identifier for this dataset.
    original_files : list[str] or None
        Filenames of the original data for linkback.
    partition_rows : int or None
        Rows per data partition. None = 2M (SIGHTLINE default).
    """
    geometry = DataGeometry.SIGHTLINE
    if partition_rows is None:
        partition_rows = DEFAULT_PARTITION_ROWS[geometry]

    # Validate required columns
    missing_obj = validate_columns(list(objects_df.columns), geometry, "objects")
    if missing_obj:
        raise ValueError(f"objects_df missing required columns: {missing_obj}")
    missing_data = validate_columns(list(data_df.columns), geometry, "data")
    if missing_data:
        raise ValueError(f"data_df missing required columns: {missing_data}")

    out_dir = _prepare_output_dir(survey_path, overwrite)

    # Write objects table
    _write_single_parquet(objects_df, out_dir / OBJECTS_FILENAME, compression)
    logger.info("  objects.parquet: %d sightlines", len(objects_df))

    # Write pixel data partitions
    part_files = _write_partitions(data_df, out_dir, partition_rows, compression)

    manifest = {
        "format_version": FORMAT_VERSION,
        "oneuniverse_version": "0.1.0",
        "survey_name": survey_name,
        "survey_type": survey_type,
        "geometry": geometry.value,
        "n_sightlines": len(objects_df),
        "sightline_id_column": sightline_id_column,
        "original_files": original_files or [],
        "original_format": original_format,
        "original_n_rows": -1,
        "n_rows": len(data_df),
        "n_objects": len(objects_df),
        "n_partitions": len(part_files),
        "partition_rows": partition_rows,
        "compression": compression,
        "has_objects_table": True,
        "partitions": part_files,
        "data_columns": list(data_df.columns),
        "object_columns": list(objects_df.columns),
        "conversion_kwargs": {},
        "created": datetime.now(timezone.utc).isoformat(),
        **extra_manifest,
    }

    _write_manifest(out_dir, manifest)
    logger.info(
        "SIGHTLINE conversion complete: %d sightlines, %d pixels → %d files",
        len(objects_df), len(data_df), len(part_files),
    )

    return out_dir


# ── Convert: HEALPIX geometry ────────────────────────────────────────────


def convert_healpix_map(
    data_df: pd.DataFrame,
    survey_path: Path,
    survey_name: str,
    survey_type: str = "map",
    nside: int = 256,
    ordering: str = "nested",
    coordsys: str = "icrs",
    original_files: Optional[List[str]] = None,
    original_format: str = "fits",
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    overwrite: bool = False,
    **extra_manifest: Any,
) -> Path:
    """Convert a HEALPix map to oneuniverse format.

    Parameters
    ----------
    data_df : pd.DataFrame
        Must contain: healpix_index, value. Optionally: weight, etc.
    survey_path : Path
        Directory where ``oneuniverse/`` will be created.
    nside : int
        HEALPix nside parameter.
    ordering : str
        "nested" or "ring".
    coordsys : str
        "icrs" or "galactic".
    """
    geometry = DataGeometry.HEALPIX
    if partition_rows is None:
        partition_rows = DEFAULT_PARTITION_ROWS[geometry]

    missing = validate_columns(list(data_df.columns), geometry, "data")
    if missing:
        raise ValueError(f"data_df missing required columns: {missing}")

    out_dir = _prepare_output_dir(survey_path, overwrite)
    part_files = _write_partitions(data_df, out_dir, partition_rows, compression)

    manifest = {
        "format_version": FORMAT_VERSION,
        "oneuniverse_version": "0.1.0",
        "survey_name": survey_name,
        "survey_type": survey_type,
        "geometry": geometry.value,
        "healpix_nside": nside,
        "healpix_ordering": ordering,
        "healpix_coordsys": coordsys,
        "original_files": original_files or [],
        "original_format": original_format,
        "original_n_rows": -1,
        "n_rows": len(data_df),
        "n_objects": len(data_df),
        "n_partitions": len(part_files),
        "partition_rows": partition_rows,
        "compression": compression,
        "has_objects_table": False,
        "partitions": part_files,
        "data_columns": list(data_df.columns),
        "object_columns": [],
        "conversion_kwargs": {},
        "created": datetime.now(timezone.utc).isoformat(),
        **extra_manifest,
    }

    _write_manifest(out_dir, manifest)
    logger.info(
        "HEALPIX conversion complete: nside=%d, %d pixels → %d files",
        nside, len(data_df), len(part_files),
    )

    return out_dir


# ── Reading ──────────────────────────────────────────────────────────────


def read_oneuniverse_parquet(
    survey_path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """Read data partitions from a converted oneuniverse directory.

    Works for any geometry — returns the part_*.parquet data.
    For SIGHTLINE geometry, use ``read_objects_table()`` to get
    the per-sightline metadata separately.
    """
    import pyarrow.parquet as pq

    ou_dir = survey_path / ONEUNIVERSE_SUBDIR
    manifest = _read_manifest(ou_dir)

    dfs = []
    for part_name in manifest["partitions"]:
        part_path = ou_dir / part_name
        table = pq.read_table(part_path, columns=columns, filters=filters)
        dfs.append(table.to_pandas())

    return pd.concat(dfs, ignore_index=True)


def read_objects_table(survey_path: Path) -> pd.DataFrame:
    """Read the objects.parquet table (SIGHTLINE geometry only).

    Returns one row per sightline with metadata (ra, dec, z_source, …).
    """
    import pyarrow.parquet as pq

    ou_dir = survey_path / ONEUNIVERSE_SUBDIR
    manifest = _read_manifest(ou_dir)

    if not manifest.get("has_objects_table", False):
        raise ValueError(
            f"No objects table for '{manifest.get('survey_name', '?')}' "
            f"(geometry={manifest.get('geometry', '?')}). "
            "Objects table is only present for SIGHTLINE geometry."
        )

    return pq.read_table(ou_dir / OBJECTS_FILENAME).to_pandas()


def fetch_original_columns(
    survey_path: Path,
    original_columns: List[str],
    row_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Fetch columns from the original file via linkback.

    Uses the ``_original_row_index`` column in the Parquet data to
    map back to rows in the original FITS/CSV file.
    """
    ou_dir = survey_path / ONEUNIVERSE_SUBDIR
    manifest = _read_manifest(ou_dir)

    original_files = manifest.get("original_files", [])
    if not original_files:
        # Backward compat with old manifests
        original_files = [manifest.get("original_file", "")]
    original_format = manifest["original_format"]
    original_file = survey_path / original_files[0]

    if original_format == "fits":
        return _fetch_from_fits(original_file, original_columns, row_indices)
    elif original_format == "csv":
        return _fetch_from_csv(original_file, original_columns, row_indices)
    else:
        raise NotImplementedError(
            f"Linkback not implemented for format '{original_format}'"
        )


# ── Introspection ────────────────────────────────────────────────────────


def get_manifest(survey_path: Path) -> Dict:
    """Read and return the manifest for a converted survey."""
    return _read_manifest(survey_path / ONEUNIVERSE_SUBDIR)


def is_converted(survey_path: Path) -> bool:
    """Check whether oneuniverse Parquet files exist for this survey."""
    ou_dir = survey_path / ONEUNIVERSE_SUBDIR
    return (ou_dir / MANIFEST_FILENAME).exists()


def get_geometry(survey_path: Path) -> DataGeometry:
    """Return the geometry of a converted survey."""
    manifest = get_manifest(survey_path)
    return DataGeometry(manifest["geometry"])


# ── Internal helpers ─────────────────────────────────────────────────────


def _prepare_output_dir(survey_path: Path, overwrite: bool) -> Path:
    """Create the oneuniverse/ output directory."""
    out_dir = survey_path / ONEUNIVERSE_SUBDIR
    if out_dir.exists():
        if overwrite:
            import shutil
            shutil.rmtree(out_dir)
            logger.info("Removed existing %s", out_dir)
        else:
            raise FileExistsError(
                f"{out_dir} already exists. Pass overwrite=True to replace."
            )
    out_dir.mkdir(parents=True)
    return out_dir


def _write_partitions(
    df: pd.DataFrame,
    out_dir: Path,
    partition_rows: int,
    compression: str,
) -> List[str]:
    """Write a DataFrame as fixed-size Parquet partitions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_total = len(df)
    n_parts = max(1, int(np.ceil(n_total / partition_rows)))

    part_files = []
    for i in range(n_parts):
        start = i * partition_rows
        end = min(start + partition_rows, n_total)
        chunk = df.iloc[start:end]

        part_name = f"part_{i:04d}.parquet"
        part_path = out_dir / part_name

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, part_path, compression=compression)
        part_files.append(part_name)

        size_mb = part_path.stat().st_size / 1e6
        logger.info(
            "  %s: rows %d–%d (%d rows, %.1f MB)",
            part_name, start, end - 1, end - start, size_mb,
        )

    return part_files


def _write_single_parquet(
    df: pd.DataFrame,
    filepath: Path,
    compression: str,
) -> None:
    """Write a single Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath, compression=compression)


def _write_manifest(out_dir: Path, manifest: Dict) -> None:
    """Write manifest.json."""
    manifest_path = out_dir / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _read_manifest(ou_dir: Path) -> Dict:
    """Read the manifest.json from an oneuniverse directory."""
    manifest_path = ou_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Run a convert function first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def _count_original_rows(survey_path: Path, config) -> int:
    """Count rows in the original file (without reading all data)."""
    original = survey_path / config.data_filename
    if not original.exists():
        return -1
    if config.data_format == "fits":
        import fitsio
        with fitsio.FITS(original) as f:
            return f[1].get_nrows()
    return -1


def _log_summary(out_dir, survey_path, config, manifest, part_files):
    """Log a human-readable summary of the conversion."""
    total_size = sum((out_dir / p).stat().st_size for p in part_files)
    original_path = survey_path / config.data_filename
    if original_path.exists():
        original_size = original_path.stat().st_size
        logger.info(
            "Conversion complete: %d rows → %d files (%.1f MB, %.1fx compression vs original %.1f MB)",
            manifest["n_rows"], len(part_files), total_size / 1e6,
            original_size / total_size, original_size / 1e6,
        )
    else:
        logger.info(
            "Conversion complete: %d rows → %d files (%.1f MB)",
            manifest["n_rows"], len(part_files), total_size / 1e6,
        )


def _fetch_from_fits(
    filepath: Path,
    columns: List[str],
    row_indices: Optional[np.ndarray],
) -> pd.DataFrame:
    """Read specific columns and rows from a FITS file."""
    import fitsio
    from oneuniverse.data._io import _fix_byteorder

    with fitsio.FITS(filepath) as f:
        hdu = f[1]
        data = {}
        for col in columns:
            if row_indices is not None:
                arr = hdu.read_column(col, rows=row_indices)
            else:
                arr = hdu.read_column(col)
            if arr.ndim == 1:
                if arr.dtype.kind in ("U", "S"):
                    arr = np.array(
                        [s.strip() if isinstance(s, str) else s for s in arr]
                    )
                else:
                    arr = _fix_byteorder(arr)
                data[col] = arr
            else:
                arr = _fix_byteorder(arr)
                data[col] = list(arr)
    return pd.DataFrame(data)


def _fetch_from_csv(
    filepath: Path,
    columns: List[str],
    row_indices: Optional[np.ndarray],
) -> pd.DataFrame:
    """Read specific columns and rows from a CSV file."""
    df = pd.read_csv(filepath, usecols=columns)
    if row_indices is not None:
        df = df.iloc[row_indices]
    return df
