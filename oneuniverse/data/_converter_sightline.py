"""OUF SIGHTLINE + HEALPIX-map writers (review S1)."""
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

from oneuniverse.data._converter_core import (
    _default_stats_builder,
    _prepare_output_dir,
    _write_partitions,
    _auto_partition_nside,
    _write_partitions_by_healpix,
    _chunk_to_table,
    _write_single_parquet,
    _load_manifest,
    _count_rows,
    _log_summary,
)
from oneuniverse.data._converter_point import write_ouf_dataset


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
