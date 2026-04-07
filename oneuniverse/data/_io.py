"""
oneuniverse.data._io
~~~~~~~~~~~~~~~~~~~~~
I/O utilities for reading survey catalogs in various formats.

Handles FITS (via fitsio), Parquet (via PyArrow), and CSV.
Provides automatic byte-order correction for FITS big-endian data.

Performance notes:
    - FITS: fitsio supports row-sliced and column-selective reads.
      For large files with row_filter, we do a two-pass approach:
      read the filter column first to get row indices, then read
      only matching rows for all other columns.
    - Parquet: PyArrow provides predicate pushdown and columnar reads.
      50x faster than FITS for selective reads on large catalogs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fix_byteorder(arr: np.ndarray) -> np.ndarray:
    """Convert big-endian (FITS) arrays to native byte order."""
    if arr.dtype.byteorder not in ("=", "<", "|"):
        return arr.astype(arr.dtype.newbyteorder("="))
    return arr


def read_fits(
    filepath: Path,
    columns: Optional[List[str]] = None,
    hdu: int = 1,
    row_filter: Optional[Dict[str, object]] = None,
) -> "tuple[pd.DataFrame, dict[str, np.ndarray]]":
    """Read a FITS binary table into a DataFrame.

    Uses a two-pass strategy when row_filter is set: first reads the
    filter column(s) to identify matching row indices, then reads only
    those rows for the remaining columns. This is much faster for
    large files with selective filters.

    Parameters
    ----------
    filepath : Path
        Path to the FITS file.
    columns : list[str] or None
        Columns to read. None = all columns.
    hdu : int
        HDU index (default 1 for first extension).
    row_filter : dict or None
        Simple equality filter, e.g. ``{"IS_QSO_FINAL": 1}``.

    Returns
    -------
    (pd.DataFrame, dict[str, np.ndarray])
        DataFrame of scalar columns, dict of multi-element array columns.
    """
    import fitsio

    with fitsio.FITS(filepath) as f:
        table_hdu = f[hdu]
        all_colnames = table_hdu.get_colnames()

        if columns is not None:
            missing = set(columns) - set(all_colnames)
            if missing:
                raise ValueError(
                    f"Columns not in FITS file: {sorted(missing)}. "
                    f"Available: {sorted(all_colnames)}"
                )
            read_cols = list(columns)
        else:
            read_cols = all_colnames

        # Two-pass row filtering: read filter columns first, get row indices
        rows = None
        if row_filter:
            mask = None
            for col_name, value in row_filter.items():
                arr = table_hdu.read_column(col_name)
                arr = _fix_byteorder(arr) if arr.dtype.kind not in ("U", "S") else arr
                col_mask = arr == value
                mask = col_mask if mask is None else (mask & col_mask)
            rows = np.where(mask)[0]
            logger.info(
                "Row filter: %d / %d rows selected (%.1f%%)",
                len(rows), table_hdu.get_nrows(),
                100 * len(rows) / table_hdu.get_nrows(),
            )

        # Read requested columns (only matching rows if filtered)
        data = {}
        array_cols = {}
        for col in read_cols:
            if rows is not None:
                arr = table_hdu.read_column(col, rows=rows)
            else:
                arr = table_hdu.read_column(col)

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
                array_cols[col] = arr

    return pd.DataFrame(data), array_cols


def read_parquet(
    filepath: Path,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame with optional predicate pushdown.

    Parameters
    ----------
    filepath : Path
        Path to the Parquet file.
    columns : list[str] or None
        Columns to read. None = all.
    filters : list or None
        PyArrow filter expressions for predicate pushdown,
        e.g. ``[("z", ">", 0.5), ("z", "<", 2.0)]``.

    Returns
    -------
    pd.DataFrame
    """
    import pyarrow.parquet as pq

    table = pq.read_table(filepath, columns=columns, filters=filters)
    return table.to_pandas()


def read_csv(
    filepath: Path,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    df = pd.read_csv(filepath, **kwargs)
    if columns is not None:
        df = df[list(columns)]
    return df


def write_parquet(
    df: pd.DataFrame,
    filepath: Path,
    compression: str = "zstd",
) -> None:
    """Write a DataFrame to Parquet with compression."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, filepath, compression=compression)
    logger.info("Wrote %d rows to %s (%.1f MB)", len(df), filepath,
                filepath.stat().st_size / 1e6)


def fits_column_info(filepath: Path, hdu: int = 1) -> List[Dict]:
    """Return metadata for all columns in a FITS binary table."""
    import fitsio

    info = []
    with fitsio.FITS(filepath) as f:
        table_hdu = f[hdu]
        colnames = table_hdu.get_colnames()
        sample = table_hdu.read(rows=range(1))
        for col in colnames:
            arr = sample[col]
            info.append({
                "name": col,
                "dtype": str(arr.dtype),
                "shape": arr.shape[1:] if arr.ndim > 1 else (),
                "ndim": arr.ndim,
            })
    return info
