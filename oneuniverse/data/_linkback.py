"""Original-column linkback: fetch projected rows from FITS/CSV/parquet (review S1)."""
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

from oneuniverse.data._converter_core import _load_manifest


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
    if spec.format == "parquet":
        return _fetch_from_parquet(original_file, original_columns,
                                   row_indices)
    raise NotImplementedError(
        f"Linkback not implemented for format '{spec.format}'"
    )


def _fetch_from_parquet(
    filepath: Path,
    columns: List[str],
    row_indices: Optional[np.ndarray],
) -> pd.DataFrame:
    """Read specific columns and rows from a parquet original (review B6).

    Column projection happens at read time; the row take runs on the
    projected table only.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(filepath, columns=list(columns))
    if row_indices is not None:
        table = table.take(np.asarray(row_indices, dtype=np.int64))
    return table.to_pandas()


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
