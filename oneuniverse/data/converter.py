"""
oneuniverse.data.converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Convert survey catalogs from their native format (FITS, CSV, …) into
the standardized **oneuniverse file format** (OUF) v2.

See ``format_spec.py`` for the formal geometry specification
(POINT, SIGHTLINE, HEALPIX) and ``manifest.py`` for the typed
:class:`Manifest` dataclass that is the single source of truth for
every converted dataset on disk.

Directory layout after conversion::

    {survey_path}/oneuniverse/
    ├── manifest.json               ← typed Manifest (see manifest.py)
    ├── objects.parquet             ← per-object metadata (SIGHTLINE only)
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

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.data._hashing import hash_file
from oneuniverse.data.format_spec import (
    COMPRESSION,
    DEFAULT_PARTITION_ROWS,
    HEALPIX_PARTITION_NSIDE,
    HEALPIX_SUBDIR_FMT,
    MANIFEST_FILENAME,
    OBJECTS_FILENAME,
    ONEUNIVERSE_SUBDIR,
    ORIGINAL_INDEX_COL,
    DataGeometry,
    validate_columns,
)
from oneuniverse.data.manifest import (
    FORMAT_VERSION,
    SCHEMA_VERSION,
    ColumnSpec,
    LoaderSpec,
    Manifest,
    OriginalFileSpec,
    PartitionSpec,
    PartitionStats,
    PartitioningSpec,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)


# ── Unified writer ───────────────────────────────────────────────────────


def write_ouf_dataset(
    df: pd.DataFrame,
    out_dir: Path,
    survey_name: str,
    survey_type: str,
    geometry: DataGeometry,
    *,
    objects_df: Optional[pd.DataFrame] = None,
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    original_paths: Optional[List[Path]] = None,
    original_format: str = "fits",
    conversion_kwargs: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    partitioning: Optional[PartitioningSpec] = None,
    loader: Optional[LoaderSpec] = None,
    stats_builder=None,
) -> Manifest:
    """Write *df* as a complete OUF 2.0 dataset under *out_dir*.

    Writes Parquet partitions (and an optional ``objects.parquet``),
    content-hashes every file, and atomically writes a typed
    :class:`Manifest` describing the result.

    Parameters
    ----------
    df
        The data table (per-object for POINT, per-pixel for SIGHTLINE
        and HEALPIX).
    out_dir
        The ``{survey_path}/oneuniverse/`` directory. Must already exist
        and be empty.
    objects_df
        Only for SIGHTLINE geometry — one row per sightline.
    partition_rows
        Rows per Parquet partition. ``None`` = geometry default.
    original_paths
        Paths to original source files for audit linkback. Each gets
        hashed and recorded in :attr:`Manifest.original_files`.
    stats_builder
        Optional callable ``(chunk_df) -> PartitionStats``.

    Returns
    -------
    The :class:`Manifest` that was written.
    """
    out_dir = Path(out_dir)
    if partition_rows is None:
        partition_rows = DEFAULT_PARTITION_ROWS[geometry]
    conversion_kwargs = dict(conversion_kwargs or {})
    extra = dict(extra or {})
    loader = loader or LoaderSpec(name="unknown", version="0.0.0")

    # Column contract ----------------------------------------------------
    missing = validate_columns(list(df.columns), geometry, "data")
    if missing:
        raise ValueError(f"data_df missing required columns: {missing}")
    if geometry is DataGeometry.SIGHTLINE:
        if objects_df is None:
            raise ValueError("SIGHTLINE geometry requires objects_df")
        missing_obj = validate_columns(list(objects_df.columns), geometry, "objects")
        if missing_obj:
            raise ValueError(f"objects_df missing required columns: {missing_obj}")

    # Objects table (SIGHTLINE only) -------------------------------------
    if objects_df is not None:
        _write_single_parquet(objects_df, out_dir / OBJECTS_FILENAME, compression)
        logger.info("  objects.parquet: %d sightlines", len(objects_df))

    # Partitions ---------------------------------------------------------
    if geometry is DataGeometry.POINT:
        partitions = _write_partitions_by_healpix(
            df, out_dir, compression, stats_builder,
        )
        if partitioning is None:
            partitioning = PartitioningSpec(
                scheme="healpix",
                column="_healpix32",
                extra={"nside": HEALPIX_PARTITION_NSIDE, "nest": True},
            )
    else:
        partitions = _write_partitions(
            df, out_dir, partition_rows, compression, stats_builder,
        )

    # Original-file specs ------------------------------------------------
    original_files: List[OriginalFileSpec] = []
    for p in original_paths or []:
        p = Path(p)
        if not p.is_file():
            original_files.append(OriginalFileSpec(
                path=str(p.name), sha256="", n_rows=None,
                size_bytes=0, format=original_format,
            ))
            continue
        original_files.append(OriginalFileSpec(
            path=str(p.name),
            sha256=hash_file(p),
            n_rows=_count_rows(p, original_format),
            size_bytes=p.stat().st_size,
            format=original_format,
        ))

    # Schema from df dtypes ---------------------------------------------
    schema_cols = [
        ColumnSpec(name=str(c), dtype=str(df[c].dtype)) for c in df.columns
    ]

    manifest = Manifest(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=SCHEMA_VERSION,
        geometry=geometry,
        survey_name=survey_name,
        survey_type=survey_type,
        created_utc=datetime.now(timezone.utc).isoformat(),
        original_files=original_files,
        partitions=partitions,
        partitioning=partitioning,
        schema=schema_cols,
        conversion_kwargs=conversion_kwargs,
        loader=loader,
        extra=extra,
    )
    write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


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
    """Convert a registered survey to OUF 2.0 POINT format."""
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

    logger.info("Loading %s via loader...", survey_name)
    if raw_path is not None:
        loader_kwargs.setdefault("data_path", survey_path)
    df = loader.load(validate=False, force_native=True, **loader_kwargs)

    # Guarantee _original_row_index for linkback (CORE col).
    if ORIGINAL_INDEX_COL not in df.columns:
        df[ORIGINAL_INDEX_COL] = np.arange(len(df), dtype=np.int64)

    original_paths = []
    if config.data_filename:
        original_paths.append(survey_path / config.data_filename)

    def _point_stats(chunk: pd.DataFrame) -> PartitionStats:
        return PartitionStats(
            ra_min=float(chunk["ra"].min()) if "ra" in chunk else None,
            ra_max=float(chunk["ra"].max()) if "ra" in chunk else None,
            dec_min=float(chunk["dec"].min()) if "dec" in chunk else None,
            dec_max=float(chunk["dec"].max()) if "dec" in chunk else None,
            z_min=float(chunk["z"].min()) if "z" in chunk else None,
            z_max=float(chunk["z"].max()) if "z" in chunk else None,
        )

    manifest = write_ouf_dataset(
        df=df,
        out_dir=out_dir,
        survey_name=config.name,
        survey_type=config.survey_type,
        geometry=DataGeometry.POINT,
        partition_rows=partition_rows,
        compression=compression,
        original_paths=original_paths,
        original_format=config.data_format or "fits",
        conversion_kwargs=loader_kwargs,
        loader=LoaderSpec(name=survey_name, version="0.2.0"),
        stats_builder=_point_stats,
    )

    _log_summary(out_dir, survey_path, config, manifest)
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
    """Convert sightline data (e.g. Lya forest) to OUF 2.0."""
    out_dir = _prepare_output_dir(Path(survey_path), overwrite)
    original_paths = [Path(survey_path) / f for f in (original_files or [])]

    extra = {
        "n_sightlines": int(len(objects_df)),
        "sightline_id_column": sightline_id_column,
        "has_objects_table": True,
        "object_columns": list(objects_df.columns),
    }
    extra.update(extra_manifest)

    write_ouf_dataset(
        df=data_df,
        out_dir=out_dir,
        survey_name=survey_name,
        survey_type=survey_type,
        geometry=DataGeometry.SIGHTLINE,
        objects_df=objects_df,
        partition_rows=partition_rows,
        compression=compression,
        original_paths=original_paths,
        original_format=original_format,
        extra=extra,
        loader=LoaderSpec(name=survey_name, version="0.2.0"),
    )

    logger.info(
        "SIGHTLINE conversion complete: %d sightlines, %d pixels",
        len(objects_df), len(data_df),
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
    """Convert a HEALPix map to OUF 2.0."""
    out_dir = _prepare_output_dir(Path(survey_path), overwrite)
    original_paths = [Path(survey_path) / f for f in (original_files or [])]

    extra = {
        "healpix_nside": int(nside),
        "healpix_ordering": ordering,
        "healpix_coordsys": coordsys,
    }
    extra.update(extra_manifest)

    write_ouf_dataset(
        df=data_df,
        out_dir=out_dir,
        survey_name=survey_name,
        survey_type=survey_type,
        geometry=DataGeometry.HEALPIX,
        partition_rows=partition_rows,
        compression=compression,
        original_paths=original_paths,
        original_format=original_format,
        extra=extra,
        loader=LoaderSpec(name=survey_name, version="0.2.0"),
    )

    logger.info(
        "HEALPIX conversion complete: nside=%d, %d pixels", nside, len(data_df),
    )
    return out_dir


# ── Reading ──────────────────────────────────────────────────────────────


def read_oneuniverse_parquet(
    survey_path: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """Read data partitions from a converted oneuniverse directory."""
    import pyarrow.parquet as pq

    ou_dir = Path(survey_path) / ONEUNIVERSE_SUBDIR
    manifest = _load_manifest(ou_dir)

    dfs = []
    for part in manifest.partitions:
        part_path = ou_dir / part.name
        table = pq.read_table(part_path, columns=columns, filters=filters)
        dfs.append(table.to_pandas())

    return pd.concat(dfs, ignore_index=True)


def read_objects_table(survey_path: Path) -> pd.DataFrame:
    """Read the objects.parquet table (SIGHTLINE geometry only)."""
    import pyarrow.parquet as pq

    ou_dir = Path(survey_path) / ONEUNIVERSE_SUBDIR
    manifest = _load_manifest(ou_dir)

    if not manifest.extra.get("has_objects_table", False):
        raise ValueError(
            f"No objects table for '{manifest.survey_name}' "
            f"(geometry={manifest.geometry.value}). "
            "Objects table is only present for SIGHTLINE geometry."
        )

    return pq.read_table(ou_dir / OBJECTS_FILENAME).to_pandas()


def fetch_original_columns(
    survey_path: Path,
    original_columns: List[str],
    row_indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Fetch columns from the original file via linkback."""
    ou_dir = Path(survey_path) / ONEUNIVERSE_SUBDIR
    manifest = _load_manifest(ou_dir)

    if not manifest.original_files:
        raise ValueError(
            f"No original files recorded in manifest for "
            f"'{manifest.survey_name}'."
        )
    spec = manifest.original_files[0]
    original_file = Path(survey_path) / spec.path

    if spec.format == "fits":
        return _fetch_from_fits(original_file, original_columns, row_indices)
    if spec.format == "csv":
        return _fetch_from_csv(original_file, original_columns, row_indices)
    raise NotImplementedError(
        f"Linkback not implemented for format '{spec.format}'"
    )


# ── Introspection ────────────────────────────────────────────────────────


def get_manifest(survey_path: Path) -> Manifest:
    """Read and return the typed :class:`Manifest` for a converted dataset."""
    return _load_manifest(Path(survey_path) / ONEUNIVERSE_SUBDIR)


def is_converted(survey_path: Path) -> bool:
    """Check whether an OUF manifest exists for this survey."""
    return (Path(survey_path) / ONEUNIVERSE_SUBDIR / MANIFEST_FILENAME).exists()


def get_geometry(survey_path: Path) -> DataGeometry:
    """Return the geometry of a converted survey."""
    return get_manifest(survey_path).geometry


# ── Internal helpers ─────────────────────────────────────────────────────


def _prepare_output_dir(survey_path: Path, overwrite: bool) -> Path:
    """Create the oneuniverse/ output directory."""
    out_dir = Path(survey_path) / ONEUNIVERSE_SUBDIR
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
    stats_builder=None,
) -> List[PartitionSpec]:
    """Write *df* as fixed-size Parquet partitions + return typed specs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_total = len(df)
    n_parts = max(1, int(np.ceil(n_total / partition_rows)))
    specs: List[PartitionSpec] = []
    for i in range(n_parts):
        start = i * partition_rows
        end = min(start + partition_rows, n_total)
        chunk = df.iloc[start:end]

        part_name = f"part_{i:04d}.parquet"
        part_path = out_dir / part_name
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, part_path, compression=compression)

        stats = stats_builder(chunk) if stats_builder else PartitionStats()
        specs.append(PartitionSpec(
            name=part_name,
            n_rows=int(end - start),
            sha256=hash_file(part_path),
            size_bytes=part_path.stat().st_size,
            stats=stats,
        ))
        logger.info(
            "  %s: rows %d–%d (%d rows, %.1f MB)",
            part_name, start, end - 1, end - start,
            part_path.stat().st_size / 1e6,
        )
    return specs


def _write_partitions_by_healpix(
    df: pd.DataFrame,
    out_dir: Path,
    compression: str,
    stats_builder=None,
) -> List[PartitionSpec]:
    """Write *df* as one Parquet file per ``_healpix32`` cell.

    Layout: ``{out_dir}/data/healpix32={cell:05d}/part_0000.parquet``.
    ``PartitionSpec.healpix_cell`` records the cell id for later
    pruning.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if "_healpix32" not in df.columns:
        raise ValueError("POINT df missing required _healpix32 column")

    data_root = out_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    specs: List[PartitionSpec] = []
    for cell, chunk in df.groupby("_healpix32", sort=True):
        cell = int(cell)
        cell_dir = data_root / HEALPIX_SUBDIR_FMT.format(cell=cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        rel_name = f"data/{cell_dir.name}/part_0000.parquet"
        part_path = out_dir / rel_name
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, part_path, compression=compression)

        stats = stats_builder(chunk) if stats_builder else PartitionStats()
        specs.append(PartitionSpec(
            name=rel_name,
            n_rows=len(chunk),
            sha256=hash_file(part_path),
            size_bytes=part_path.stat().st_size,
            stats=stats,
            healpix_cell=cell,
        ))
        logger.info(
            "  %s: %d rows (%.1f MB)",
            rel_name, len(chunk), part_path.stat().st_size / 1e6,
        )
    return specs


def _write_single_parquet(
    df: pd.DataFrame, filepath: Path, compression: str,
) -> None:
    """Write a single Parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath, compression=compression)


def _load_manifest(ou_dir: Path) -> Manifest:
    """Read the typed Manifest from an oneuniverse directory."""
    manifest_path = ou_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found at {manifest_path}. Run a convert function first."
        )
    return read_manifest(manifest_path)


def _count_rows(path: Path, fmt: str) -> Optional[int]:
    """Count rows in an original source file without loading all data."""
    if fmt == "fits":
        try:
            import fitsio
            with fitsio.FITS(path) as f:
                return int(f[1].get_nrows())
        except Exception:
            return None
    if fmt == "csv":
        try:
            with open(path) as f:
                return sum(1 for _ in f) - 1  # minus header
        except Exception:
            return None
    return None


def _log_summary(out_dir, survey_path, config, manifest: Manifest):
    """Log a human-readable summary of the conversion."""
    total_size = sum(p.size_bytes for p in manifest.partitions)
    original_path = Path(survey_path) / (config.data_filename or "")
    if config.data_filename and original_path.exists():
        original_size = original_path.stat().st_size
        logger.info(
            "Conversion complete: %d rows → %d files (%.1f MB, "
            "%.1fx compression vs original %.1f MB)",
            manifest.n_rows, manifest.n_partitions, total_size / 1e6,
            original_size / max(total_size, 1), original_size / 1e6,
        )
    else:
        logger.info(
            "Conversion complete: %d rows → %d files (%.1f MB)",
            manifest.n_rows, manifest.n_partitions, total_size / 1e6,
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
