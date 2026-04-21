import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.manifest import (
    ColumnSpec, LoaderSpec, Manifest, OriginalFileSpec,
    PartitionSpec, PartitionStats, write_manifest,
)
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR
from oneuniverse.data.temporal import TemporalSpec
from oneuniverse.data.validity import DatasetValidity


_T0 = "2026-01-01T00:00:00+00:00"


def _make_df(n, seed, t0, t1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ra":  rng.uniform(0.0, 360.0, n),
        "dec": rng.uniform(-60.0, 60.0, n),
        "t_obs": rng.uniform(t0, t1, n),
    })


def _write_point(ou_dir: Path, name: str, df: pd.DataFrame) -> PartitionSpec:
    path = ou_dir / name
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return PartitionSpec(
        name=name, n_rows=len(df),
        sha256="0" * 16, size_bytes=path.stat().st_size,
        stats=PartitionStats(
            t_min=float(df["t_obs"].min()),
            t_max=float(df["t_obs"].max()),
        ),
    )


def _build(tmp_path: Path, partitions):
    survey_dir = tmp_path / "syn"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    specs = [_write_point(ou_dir, name, df) for name, df in partitions]
    t_min = min(p.stats.t_min for p in specs)
    t_max = max(p.stats.t_max for p in specs)
    m = Manifest(
        oneuniverse_format_version="2.1.0",
        oneuniverse_schema_version="2.1.0",
        geometry=DataGeometry.POINT,
        survey_name="syn", survey_type="transient",
        created_utc=_T0,
        original_files=[OriginalFileSpec(path="src.csv", sha256="0" * 16,
                                         n_rows=None, size_bytes=1, format="csv")],
        partitions=specs, partitioning=None,
        schema=[ColumnSpec(name=c, dtype=str(partitions[0][1][c].dtype))
                for c in partitions[0][1].columns],
        conversion_kwargs={},
        loader=LoaderSpec(name="syn", version="0"),
        temporal=TemporalSpec(t_min=t_min, t_max=t_max),
        validity=DatasetValidity(valid_from_utc=_T0),
    )
    write_manifest(ou_dir / "manifest.json", m)
    return survey_dir


def test_t_range_prunes_partition_by_stats(tmp_path):
    early = _make_df(5, 1, 58000.0, 58100.0)
    late = _make_df(5, 2, 60000.0, 60100.0)
    root = _build(tmp_path, [("part_0000.parquet", early),
                             ("part_0001.parquet", late)])
    view = DatasetView.from_path(root)
    kept = view._select_partitions(t_range=(59500.0, 61000.0))
    assert [p.name for p in kept] == ["part_0001.parquet"]


def test_t_range_pushdown_filters_rows(tmp_path):
    df = pd.DataFrame({
        "ra": [0.0, 0.0, 0.0],
        "dec": [0.0, 0.0, 0.0],
        "t_obs": [58000.0, 59000.0, 60000.0],
    })
    root = _build(tmp_path, [("part_0000.parquet", df)])
    view = DatasetView.from_path(root)
    out = view.read(columns=["t_obs"], t_range=(58500.0, 59500.0))
    assert list(out["t_obs"]) == [59000.0]


def test_t_range_none_returns_all(tmp_path):
    df = pd.DataFrame({"ra": [0.0, 0.0], "dec": [0.0, 0.0],
                       "t_obs": [58000.0, 60000.0]})
    root = _build(tmp_path, [("part_0000.parquet", df)])
    assert len(DatasetView.from_path(root).read(columns=["t_obs"])) == 2


def test_t_range_outside_any_partition_returns_empty(tmp_path):
    df = pd.DataFrame({"ra": [0.0], "dec": [0.0], "t_obs": [58000.0]})
    root = _build(tmp_path, [("part_0000.parquet", df)])
    out = DatasetView.from_path(root).read(
        columns=["t_obs"], t_range=(59000.0, 59500.0),
    )
    assert len(out) == 0


def _point_df_with_t_obs(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ra":  rng.uniform(0.0, 360.0, n),
        "dec": rng.uniform(-60.0, 60.0, n),
        "z":   rng.uniform(0.01, 0.3, n),
        "z_type": ["spec"] * n,
        "z_err":  rng.uniform(1e-4, 1e-3, n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"syn{i:05d}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
        "t_obs": rng.uniform(58000.0, 60000.0, n),
    })


def _point_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "ra": [0.0] * n, "dec": [0.0] * n, "z": [0.1] * n,
        "z_type": ["spec"] * n, "z_err": [0.01] * n,
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": [f"syn{i}" for i in range(n)],
        "_original_row_index": np.arange(n, dtype=np.int64),
    })


def _ensure_healpix(df: pd.DataFrame) -> pd.DataFrame:
    if "_healpix32" not in df.columns:
        df = df.copy()
        df["_healpix32"] = np.zeros(len(df), dtype=np.int64)
    return df


def test_converter_fills_temporal_when_t_obs_present(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = _ensure_healpix(_point_df_with_t_obs(500))
    survey_dir = tmp_path / "syn"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name="syn", survey_type="transient",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
    )
    m = read_manifest(ou_dir / "manifest.json")
    assert m.temporal is not None
    assert m.temporal.time_column == "t_obs"
    assert m.temporal.t_min == float(df["t_obs"].min())
    assert m.temporal.t_max == float(df["t_obs"].max())
    assert all(p.stats.t_min is not None for p in m.partitions)


def test_converter_accepts_validity_kwarg(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = _ensure_healpix(_point_df(1))
    v = DatasetValidity(valid_from_utc=_T0, version="dr17",
                        supersedes=("eboss_qso_dr16",))
    survey_dir = tmp_path / "with_v"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name="with_v", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
        validity=v,
    )
    m = read_manifest(ou_dir / "manifest.json")
    assert m.validity == v


def test_converter_fills_default_validity_when_absent(tmp_path):
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.manifest import read_manifest

    df = _ensure_healpix(_point_df(1))
    survey_dir = tmp_path / "no_v"
    ou_dir = survey_dir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df, ou_dir,
        survey_name="no_v", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="syn", version="0"),
    )
    m = read_manifest(ou_dir / "manifest.json")
    assert m.validity is not None
    assert m.validity.is_current()
