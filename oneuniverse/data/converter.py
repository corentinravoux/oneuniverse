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


# ── public façade: definitions live in sibling modules (S1 split) ──
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
from oneuniverse.data._converter_point import (  # noqa: F401
    write_ouf_dataset, convert_survey,
)
from oneuniverse.data._converter_sightline import (  # noqa: F401
    convert_sightlines, convert_healpix_map,
)
from oneuniverse.data._linkback import (  # noqa: F401
    fetch_original_columns, _fetch_from_parquet, _fetch_from_fits, _fetch_from_csv,
)


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


def get_manifest(survey_path: Path) -> Manifest:
    """Read and return the typed :class:`Manifest` for a converted dataset."""
    return _load_manifest(Path(survey_path) / ONEUNIVERSE_SUBDIR)


def is_converted(survey_path: Path) -> bool:
    """Check whether an OUF manifest exists for this survey."""
    return (Path(survey_path) / ONEUNIVERSE_SUBDIR / MANIFEST_FILENAME).exists()


def get_geometry(survey_path: Path) -> DataGeometry:
    """Return the geometry of a converted survey."""
    return get_manifest(survey_path).geometry
