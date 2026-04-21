"""Writer for OUF 2.1 LIGHTCURVE datasets.

A LIGHTCURVE dataset is the structural twin of SIGHTLINE: an
``objects.parquet`` table (one row per source) + epoch partitions
(``part_*.parquet``) sorted by ``(object_id, mjd)``. Time replaces
wavelength as the inner axis.

The writer enforces:
- all epoch ``object_id`` values exist in the objects table (no orphans);
- required columns per :data:`GEOMETRY_COLUMNS`;
- per-partition ``t_min`` / ``t_max`` on ``mjd`` for pushdown;
- dataset-level :class:`TemporalSpec` with ``time_column="mjd"``.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.data._atomic import atomic_write_bytes
from oneuniverse.data._hashing import hash_bytes
from oneuniverse.data.format_spec import (
    COMPRESSION,
    DEFAULT_PARTITION_ROWS,
    ONEUNIVERSE_SUBDIR,
    DataGeometry,
    validate_columns,
)
from oneuniverse.data.manifest import (
    ColumnSpec,
    LoaderSpec,
    Manifest,
    PartitionSpec,
    PartitionStats,
    write_manifest,
)
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


def _healpix32(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    import healpy as hp
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    return hp.ang2pix(32, theta, phi, nest=True).astype(np.int64)


def _check_no_orphan_epochs(
    objects: pd.DataFrame, epochs: pd.DataFrame,
) -> None:
    known = set(objects["object_id"].astype(np.int64).tolist())
    seen = set(epochs["object_id"].astype(np.int64).tolist())
    orphans = seen - known
    if orphans:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: {len(orphans)} epoch rows "
            f"refer to orphan object_id(s) not present in the objects "
            f"table: {sorted(list(orphans))[:5]}"
        )


def _build_objects_table(
    objects: pd.DataFrame, epochs: pd.DataFrame,
) -> pd.DataFrame:
    per_obj = epochs.groupby("object_id").agg(
        n_epochs=("mjd", "size"),
        mjd_min=("mjd", "min"),
        mjd_max=("mjd", "max"),
    ).reset_index()
    merged = objects.merge(per_obj, on="object_id", how="left")
    if merged["n_epochs"].isna().any():
        missing = merged.loc[merged["n_epochs"].isna(), "object_id"].tolist()
        raise ValueError(
            f"write_ouf_lightcurve_dataset: objects {missing!r} have no "
            f"epochs in the epoch table."
        )
    merged["n_epochs"] = merged["n_epochs"].astype("int32")
    if "_healpix32" not in merged.columns:
        merged["_healpix32"] = _healpix32(
            merged["ra"].to_numpy(), merged["dec"].to_numpy(),
        )
    missing_cols = validate_columns(
        list(merged.columns), DataGeometry.LIGHTCURVE, table_type="objects",
    )
    if missing_cols:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: objects table missing "
            f"required columns {missing_cols!r}"
        )
    return merged


def _partition(epochs: pd.DataFrame, rows: int) -> List[pd.DataFrame]:
    epochs = epochs.sort_values(["object_id", "mjd"]).reset_index(drop=True)
    return [epochs.iloc[s: s + rows].copy()
            for s in range(0, len(epochs), rows)]


def _schema(df: pd.DataFrame) -> List[ColumnSpec]:
    return [ColumnSpec(name=str(c), dtype=str(df[c].dtype))
            for c in df.columns]


def write_ouf_lightcurve_dataset(
    *,
    objects: pd.DataFrame,
    epochs: pd.DataFrame,
    survey_path: Union[str, Path],
    survey_name: str,
    survey_type: str,
    loader_name: str,
    loader_version: str,
    partition_rows: int = DEFAULT_PARTITION_ROWS[DataGeometry.LIGHTCURVE],
    conversion_kwargs: Optional[dict] = None,
    validity: Optional[DatasetValidity] = None,
) -> Path:
    """Write an OUF 2.1 LIGHTCURVE dataset.

    Layout: ``{survey_path}/oneuniverse/{manifest.json, objects.parquet,
    part_*.parquet}``. Epoch partitions are sorted by ``(object_id, mjd)``.
    """
    survey_path = Path(survey_path)
    ou = survey_path / ONEUNIVERSE_SUBDIR
    ou.mkdir(parents=True, exist_ok=True)

    _check_no_orphan_epochs(objects, epochs)

    missing = validate_columns(
        list(epochs.columns), DataGeometry.LIGHTCURVE, table_type="data",
    )
    if missing:
        raise ValueError(
            f"write_ouf_lightcurve_dataset: epoch table missing required "
            f"columns {missing!r}"
        )

    objects_final = _build_objects_table(objects, epochs)
    obj_buf = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pandas(objects_final, preserve_index=False),
        obj_buf, compression=COMPRESSION,
    )
    atomic_write_bytes(ou / "objects.parquet", obj_buf.getvalue().to_pybytes())

    parts = _partition(epochs, partition_rows)
    specs: List[PartitionSpec] = []
    for i, chunk in enumerate(parts):
        name = f"part_{i:04d}.parquet"
        buf = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pandas(chunk, preserve_index=False),
            buf, compression=COMPRESSION,
        )
        data = buf.getvalue().to_pybytes()
        atomic_write_bytes(ou / name, data)
        specs.append(PartitionSpec(
            name=name, n_rows=len(chunk),
            sha256=hash_bytes(data),
            size_bytes=(ou / name).stat().st_size,
            stats=PartitionStats(
                t_min=float(chunk["mjd"].min()),
                t_max=float(chunk["mjd"].max()),
            ),
        ))

    temporal = TemporalSpec(
        t_min=float(epochs["mjd"].min()),
        t_max=float(epochs["mjd"].max()),
        time_column="mjd",
    )
    if validity is None:
        validity = DatasetValidity(
            valid_from_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    manifest = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.LIGHTCURVE,
        survey_name=survey_name,
        survey_type=survey_type,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_files=[],
        partitions=specs,
        partitioning=None,
        schema=_schema(epochs),
        conversion_kwargs=dict(conversion_kwargs or {}),
        loader=LoaderSpec(name=loader_name, version=loader_version),
        extra={"n_objects": int(len(objects_final))},
        temporal=temporal,
        validity=validity,
    )
    write_manifest(ou / "manifest.json", manifest)
    return survey_path
