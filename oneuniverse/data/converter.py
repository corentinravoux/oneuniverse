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
>>> from oneuniverse.data import convert_survey
>>> convert_survey("eboss_qso", data_root="/data/surveys", overwrite=True, qso_only=True)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from oneuniverse.data.coordinate_spec import CoordinateSpec
    from oneuniverse.data.spectrum_spec import SpectrumSpec

import numpy as np
import pandas as pd

from oneuniverse.data._hashing import hash_file
from oneuniverse.data.format_spec import (
    COMPRESSION,
    DEFAULT_PARTITION_ROWS,
    HEALPIX_PARTITION_NSIDE,
    HEALPIX_SUBDIR_FMT,
    MANIFEST_FILENAME,
    MIN_ROWS_PER_PARTITION,
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
)
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.manifest import (
    read_manifest,
    write_manifest,
)
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity

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
    temporal: Optional[TemporalSpec] = None,
    validity: Optional[DatasetValidity] = None,
    pdf_spec: Optional[PdfSpec] = None,
    partition_nside: Optional[int] = None,
    coordinate: Optional["CoordinateSpec"] = None,
    spectrum: Optional["SpectrumSpec"] = None,
    column_dtypes: Optional[Dict[str, str]] = None,
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

    # Phase 16: validate z_type values against the runtime registry and
    # capture the observed set for the manifest. Fail loudly rather than
    # silently writing a manifest that breaks downstream.
    if "z_type" in df.columns:
        from oneuniverse.data.ztypes import assert_valid as _assert_z_types

        seen = {str(v) for v in df["z_type"].dropna().unique()}
        _assert_z_types(seen)
        observed_z_types = tuple(sorted(seen))
    else:
        observed_z_types = ()

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

    # Default stats builder: captures all available partition columns
    # (ra, dec, z, t_obs) so per-partition pruning can filter on any of
    # them without each caller hand-rolling a builder.
    if stats_builder is None:
        stats_builder = _default_stats_builder

    # Partitions ---------------------------------------------------------
    if geometry is DataGeometry.POINT:
        chosen_nside = (
            int(partition_nside) if partition_nside is not None
            else _auto_partition_nside(len(df))
        )
        partitions = _write_partitions_by_healpix(
            df, out_dir, compression, stats_builder, pdf_spec,
            partition_nside=chosen_nside,
            column_dtypes=column_dtypes,
        )
        if partitioning is None:
            partitioning = PartitioningSpec(
                scheme="healpix",
                column="_healpix32",
                extra={"nside": chosen_nside, "nest": True},
            )
    else:
        partitions = _write_partitions(
            df, out_dir, partition_rows, compression, stats_builder, pdf_spec,
            column_dtypes=column_dtypes,
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

    # Temporal auto-fill from df["t_obs"] (POINT geometry only) ---------
    if temporal is None and "t_obs" in df.columns and geometry is DataGeometry.POINT:
        temporal = TemporalSpec(
            t_min=float(df["t_obs"].min()),
            t_max=float(df["t_obs"].max()),
        )

    # Default validity: "as of this conversion, still current" ----------
    if validity is None:
        validity = DatasetValidity(
            valid_from_utc=datetime.now(timezone.utc).isoformat(),
        )

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
        temporal=temporal,
        validity=validity,
        pdf_spec=pdf_spec,
        coordinate=coordinate,
        spectrum=spectrum,
        observed_z_types=observed_z_types,
    )
    write_manifest(out_dir / MANIFEST_FILENAME, manifest)
    return manifest


# ── Convert: POINT geometry ──────────────────────────────────────────────


def convert_survey(
    survey_name: Optional[str] = None,
    data_root: Optional[str | Path] = None,
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    overwrite: bool = False,
    output_dir: Optional[str | Path] = None,
    raw_path: Optional[str | Path] = None,
    *,
    loader=None,
    partition_nside: Optional[int] = None,
    **loader_kwargs: Any,
) -> Path:
    """Convert a registered survey to OUF 2.0 POINT format.

    Either pass a registered ``survey_name`` (loader looked up in the
    ``@register`` registry) or an explicit ``loader=<BaseSurveyLoader>``
    instance for one-off / unregistered loaders. The instance form
    bypasses the registry; useful for tests and ad-hoc conversions.
    """
    from oneuniverse.data._config import resolve_survey_path
    from oneuniverse.data._registry import get_loader

    if loader is None and survey_name is None:
        raise TypeError(
            "convert_survey requires either survey_name= (registered) or "
            "loader=<BaseSurveyLoader instance>"
        )
    if loader is None:
        loader = get_loader(survey_name)
    config = loader.config
    survey_name = survey_name or config.name

    if raw_path is not None:
        rp = Path(raw_path).expanduser().resolve()
        survey_path = rp.parent if rp.is_file() else rp
    else:
        survey_path = resolve_survey_path(
            config.survey_type, config.name, config.data_subpath,
            data_root=Path(data_root) if data_root is not None else None,
        )
        if survey_path is None and config.data_filename:
            raise FileNotFoundError(
                f"Cannot resolve data path for '{survey_name}'. "
                "Set ONEUNIVERSE_DATA_ROOT or pass data_root= or raw_path=."
            )

    if output_dir is None and survey_path is None:
        raise TypeError(
            "convert_survey: pass output_dir= when survey_path cannot be resolved"
        )
    out_base = Path(output_dir) if output_dir is not None else survey_path
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = _prepare_output_dir(out_base, overwrite)

    logger.info("Loading %s via loader...", survey_name)
    # Pass survey_path unconditionally so the loader's _load_raw can find
    # its files without relying on the (removed) module-level data root.
    if survey_path is not None:
        loader_kwargs.setdefault("data_path", survey_path)
    df = loader.load(validate=False, force_native=True, **loader_kwargs)

    # Guarantee _original_row_index for linkback (CORE col).
    if ORIGINAL_INDEX_COL not in df.columns:
        df[ORIGINAL_INDEX_COL] = np.arange(len(df), dtype=np.int64)

    # Guarantee CORE `z_err`: promote the active redshift-group's error
    # column when loaders only populate it (common pattern: spectroscopic
    # group exposes `z_spec_err`, photometric exposes `z_phot_err`).
    if "z_err" not in df.columns:
        for src in ("z_spec_err", "z_phot_err"):
            if src in df.columns:
                df["z_err"] = df[src].astype(np.float32)
                break

    # Guarantee CORE/partition key `_healpix32`. Computed once here so
    # every POINT loader gets partitioning "for free".
    if "_healpix32" not in df.columns and {"ra", "dec"}.issubset(df.columns):
        import healpy as hp
        theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
        phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
        df["_healpix32"] = hp.ang2pix(
            HEALPIX_PARTITION_NSIDE, theta, phi, nest=True,
        ).astype(np.int32)

    original_paths = []
    if config.data_filename and survey_path is not None:
        original_paths.append(survey_path / config.data_filename)

    # Phase 16: pull observational metadata from the loader if declared.
    coord = loader.coordinate_spec()
    spec = loader.spectrum_spec()

    manifest = write_ouf_dataset(
        df=df,
        out_dir=out_dir,
        survey_name=config.name,
        survey_type=config.survey_type,
        geometry=DataGeometry.POINT,
        partition_rows=partition_rows,
        partition_nside=partition_nside,
        compression=compression,
        original_paths=original_paths,
        original_format=config.data_format or "fits",
        conversion_kwargs=loader_kwargs,
        loader=LoaderSpec(name=survey_name, version="0.2.0"),
        coordinate=coord,
        spectrum=spec,
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


def _default_stats_builder(chunk: pd.DataFrame) -> PartitionStats:
    def _minmax(col: str):
        if col not in chunk.columns:
            return None, None
        return float(chunk[col].min()), float(chunk[col].max())
    ra_lo, ra_hi = _minmax("ra")
    dec_lo, dec_hi = _minmax("dec")
    z_lo, z_hi = _minmax("z")
    t_lo, t_hi = _minmax("t_obs")
    return PartitionStats(
        ra_min=ra_lo, ra_max=ra_hi,
        dec_min=dec_lo, dec_max=dec_hi,
        z_min=z_lo, z_max=z_hi,
        t_min=t_lo, t_max=t_hi,
    )


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
    pdf_spec: Optional[PdfSpec] = None,
    column_dtypes: Optional[Dict[str, str]] = None,
) -> List[PartitionSpec]:
    """Write *df* as fixed-size Parquet partitions + return typed specs."""
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
        table = _chunk_to_table(chunk, pdf_spec, column_dtypes=column_dtypes)
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


def _auto_partition_nside(
    n_rows: int, min_rows: int = MIN_ROWS_PER_PARTITION,
) -> int:
    """Return the largest valid NSIDE (power of 2, ≤ HEALPIX_PARTITION_NSIDE)
    for which the *mean* rows-per-cell ≥ ``min_rows``. Floors at 1.
    """
    nside = HEALPIX_PARTITION_NSIDE
    while nside > 1:
        npix = 12 * nside * nside
        if n_rows >= min_rows * npix:
            return nside
        nside //= 2
    return 1


def _write_partitions_by_healpix(
    df: pd.DataFrame,
    out_dir: Path,
    compression: str,
    stats_builder=None,
    pdf_spec: Optional[PdfSpec] = None,
    partition_nside: int = HEALPIX_PARTITION_NSIDE,
    column_dtypes: Optional[Dict[str, str]] = None,
) -> List[PartitionSpec]:
    """Write *df* as one Parquet file per partition cell.

    Layout: ``{out_dir}/data/healpix32={cell:05d}/part_0000.parquet``.
    The cell id is at ``partition_nside`` (coarsened from the fine
    NSIDE=32 ``_healpix32`` column by right-shifting in NEST ordering).
    ``PartitionSpec.healpix_cell`` records the (coarse) cell id.
    """
    import pyarrow.parquet as pq

    if "_healpix32" not in df.columns:
        raise ValueError("POINT df missing required _healpix32 column")

    fine = HEALPIX_PARTITION_NSIDE
    if partition_nside > fine or fine % partition_nside != 0:
        raise ValueError(
            f"partition_nside={partition_nside} must be a power-of-2 divisor "
            f"of HEALPIX_PARTITION_NSIDE={fine}"
        )
    bits_to_drop = 2 * int(np.log2(fine // partition_nside))
    fine_cells = df["_healpix32"].to_numpy(dtype=np.int64)
    partition_cells = fine_cells >> bits_to_drop if bits_to_drop else fine_cells

    data_root = out_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    specs: List[PartitionSpec] = []
    df_with_pcell = df.assign(_partition_cell=partition_cells)
    for cell, chunk in df_with_pcell.groupby(
        "_partition_cell", sort=True, observed=False,
    ):
        cell = int(cell)
        cell_dir = data_root / HEALPIX_SUBDIR_FMT.format(cell=cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        rel_name = f"data/{cell_dir.name}/part_0000.parquet"
        part_path = out_dir / rel_name
        chunk = chunk.drop(columns=["_partition_cell"])
        table = _chunk_to_table(chunk, pdf_spec, column_dtypes=column_dtypes)
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


def _chunk_to_table(
    chunk: pd.DataFrame,
    pdf_spec: Optional[PdfSpec],
    *,
    column_dtypes: Optional[Dict[str, str]] = None,
):
    """Convert a DataFrame chunk to a pyarrow Table.

    Routing
    -------
    * Columns listed in ``column_dtypes`` are coerced according to the
      dtype mini-language in :mod:`oneuniverse.data.dtype_lang`
      (``f4[N]`` / ``i8[N]`` / ``list<f4>`` / ``large_list<f4>``).
    * PDF columns implied by ``pdf_spec`` are cast to
      ``FixedSizeList[float32, n_components]`` (Phase 10 behaviour).
    * Remaining columns fall through to :func:`pa.Table.from_pandas`.
    """
    import pyarrow as pa

    from oneuniverse.data.dtype_lang import parse_dtype

    column_dtypes = dict(column_dtypes or {})

    # Resolve PDF list columns first so they appear in ``list_cols`` like
    # any other variable-length payload.
    if pdf_spec is not None:
        n = int(pdf_spec.n_components)
        pdf_cols = ["z_pdf_values"]
        if pdf_spec.parameterisation == "mixmod":
            pdf_cols += ["z_pdf_sigma", "z_pdf_weights"]
        for c in pdf_cols:
            if c in chunk.columns:
                column_dtypes.setdefault(c, f"f4[{n}]")

    list_cols = [c for c in column_dtypes if c in chunk.columns]
    scalar = chunk.drop(columns=list_cols)
    table = pa.Table.from_pandas(scalar, preserve_index=False)

    for col in list_cols:
        spec = column_dtypes[col]
        target = parse_dtype(spec)
        if isinstance(target, pa.FixedSizeListType):
            n_target = target.list_size
            arr = np.stack(
                [
                    np.asarray(r, dtype=target.value_type.to_pandas_dtype())
                    for r in chunk[col].to_numpy()
                ]
            )
            if arr.shape[1] != n_target:
                raise ValueError(
                    f"column {col!r}: expected {n_target} components, "
                    f"got {arr.shape[1]}"
                )
            flat = pa.array(arr.reshape(-1), type=target.value_type)
            built = pa.FixedSizeListArray.from_arrays(flat, n_target)
        elif isinstance(target, (pa.ListType, pa.LargeListType)):
            built = pa.array(
                [list(r) for r in chunk[col].to_numpy()],
                type=target,
            )
        else:
            built = pa.array(chunk[col].to_numpy(), type=target)
        table = table.append_column(col, built)

    return table


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
    if survey_path is None or not config.data_filename:
        logger.info(
            "Conversion complete: %d rows → %d files (%.1f MB)",
            manifest.n_rows, manifest.n_partitions, total_size / 1e6,
        )
        return
    original_path = Path(survey_path) / config.data_filename
    if original_path.exists():
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
