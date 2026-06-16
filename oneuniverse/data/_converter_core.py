"""Shared OUF partition-writing engine: stats, partitioning, parquet writers.

Split out of converter.py (review S1). Pure helpers; no public API of its own.
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


def _default_stats_builder(
    chunk: pd.DataFrame,
    *,
    extra_columns: tuple = (),
) -> PartitionStats:
    def _minmax(col: str):
        if col not in chunk.columns:
            return None, None
        return float(chunk[col].min()), float(chunk[col].max())
    ra_lo, ra_hi = _minmax("ra")
    dec_lo, dec_hi = _minmax("dec")
    z_lo, z_hi = _minmax("z")
    t_lo, t_hi = _minmax("t_obs")
    er = {}
    for col in extra_columns:
        lo, hi = _minmax(col)
        if lo is not None:
            er[col] = (lo, hi)
    return PartitionStats(
        ra_min=ra_lo, ra_max=ra_hi,
        dec_min=dec_lo, dec_max=dec_hi,
        z_min=z_lo, z_max=z_hi,
        t_min=t_lo, t_max=t_hi,
        extra_ranges=er,
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
    # any other variable-length payload. Phase 18 splits the routing per
    # parameterisation: ``sample`` -> per-row ``list<f4>``; everything
    # else (interp / quant / mixmod / hist) -> fixed-size ``f4[N]``.
    if pdf_spec is not None:
        n = int(pdf_spec.n_components)
        param = pdf_spec.parameterisation
        if param == "sample":
            if pdf_spec.value_column in chunk.columns:
                column_dtypes.setdefault(pdf_spec.value_column, "list<f4>")
        else:
            pdf_cols = [pdf_spec.value_column]
            if param == "mixmod":
                pdf_cols += [pdf_spec.sigma_column, pdf_spec.weights_column]
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
