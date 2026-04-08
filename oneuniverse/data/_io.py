"""
oneuniverse.data._io
~~~~~~~~~~~~~~~~~~~~~
Low-level I/O helpers shared by the converter and the survey loaders.

Currently provides:
    - ``read_fits``       : selective FITS reads with optional row filtering
    - ``_fix_byteorder``  : convert big-endian FITS arrays to native order
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

    Uses a two-pass strategy when ``row_filter`` is set: first reads the
    filter columns to identify matching row indices, then reads only those
    rows for the remaining columns.  This is much faster for large files
    with selective filters.

    Parameters
    ----------
    filepath : Path
        Path to the FITS file.
    columns : list[str] or None
        Columns to read.  ``None`` = all columns.
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

        # Two-pass row filtering: read filter columns first, get row indices.
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
