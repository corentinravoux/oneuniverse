"""Tests for :class:`oneuniverse.data.dataset_view.DatasetView`."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from oneuniverse.data import DatasetView, OneuniverseDatabase, write_ouf_dataset
from oneuniverse.data.format_spec import DataGeometry, ONEUNIVERSE_SUBDIR
from oneuniverse.data.manifest import PartitionStats


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_point_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    import healpy as hp
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    z = rng.uniform(0.01, 0.5, n)
    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": z,
        "z_type": ["spec"] * n,
        "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": "synth",
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": hp.ang2pix(
            32, np.radians(90.0 - dec), np.radians(ra), nest=True,
        ).astype(np.int32),
    })


def _point_stats(chunk: pd.DataFrame) -> PartitionStats:
    return PartitionStats(
        ra_min=float(chunk["ra"].min()),
        ra_max=float(chunk["ra"].max()),
        dec_min=float(chunk["dec"].min()),
        dec_max=float(chunk["dec"].max()),
        z_min=float(chunk["z"].min()),
        z_max=float(chunk["z"].max()),
    )


def _write(tmp: Path, df: pd.DataFrame, partition_rows: int = None,
           subdir: str = "synth"):
    """Write *df* as OUF 2.0 and return the ou_dir."""
    ou_dir = tmp / subdir / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df=df,
        out_dir=ou_dir,
        survey_name="synth",
        survey_type="test",
        geometry=DataGeometry.POINT,
        partition_rows=partition_rows,
        stats_builder=_point_stats,
    )
    return ou_dir


# ── Construction ─────────────────────────────────────────────────────────


class TestConstruction:
    def test_from_ou_dir(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(50))
        view = DatasetView.from_ou_dir(ou_dir)
        assert view.ou_dir == ou_dir
        assert view.manifest.n_rows == 50

    def test_from_path(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(50))
        view = DatasetView.from_path(ou_dir.parent)
        assert view.n_rows == 50

    def test_properties(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(123))
        view = DatasetView.from_ou_dir(ou_dir)
        assert view.n_rows == 123
        assert view.n_partitions >= 1
        assert view.geometry == DataGeometry.POINT
        assert view.survey_name == "synth"
        assert "ra" in view.columns
        assert "z_type" in view.columns


# ── Scan ─────────────────────────────────────────────────────────────────


class TestScan:
    def test_all_columns(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(80))
        tbl = DatasetView.from_ou_dir(ou_dir).scan()
        assert isinstance(tbl, pa.Table)
        assert tbl.num_rows == 80
        assert "ra" in tbl.column_names

    def test_column_projection(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(60))
        tbl = DatasetView.from_ou_dir(ou_dir).scan(columns=["ra", "dec"])
        assert tbl.column_names == ["ra", "dec"]
        assert tbl.num_rows == 60

    def test_user_filter_pushdown(self, tmp_path_clean):
        df = _make_point_df(200)
        ou_dir = _write(tmp_path_clean, df)
        expr = pc.field("z") >= 0.25
        tbl = DatasetView.from_ou_dir(ou_dir).scan(filter=expr)
        assert tbl.num_rows == int((df["z"] >= 0.25).sum())

    def test_z_range_filter(self, tmp_path_clean):
        df = _make_point_df(200)
        ou_dir = _write(tmp_path_clean, df)
        tbl = DatasetView.from_ou_dir(ou_dir).scan(z_range=(0.1, 0.3))
        assert tbl.num_rows == int(((df["z"] >= 0.1) & (df["z"] <= 0.3)).sum())

    def test_read_returns_dataframe(self, tmp_path_clean):
        ou_dir = _write(tmp_path_clean, _make_point_df(40))
        out = DatasetView.from_ou_dir(ou_dir).read(columns=["ra", "z"])
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["ra", "z"]
        assert len(out) == 40


# ── Partition pruning ────────────────────────────────────────────────────


class TestPartitionPruning:
    def test_multi_partition_layout(self, tmp_path_clean):
        # POINT geometry partitions one Parquet file per HEALPix cell.
        df = _make_point_df(1000)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        expected_cells = df["_healpix32"].nunique()
        assert view.n_partitions == expected_cells
        # Every partition records its cell id.
        assert all(p.healpix_cell is not None for p in view.manifest.partitions)

    def test_z_range_prunes_partitions(self, tmp_path_clean):
        df = _make_point_df(1000)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        # A narrow z slice removes at least one partition.
        kept = view._select_partitions(z_range=(0.45, 0.50))
        assert 0 <= len(kept) <= view.n_partitions

    def test_no_range_keeps_all(self, tmp_path_clean):
        df = _make_point_df(500)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        assert len(view._select_partitions()) == view.n_partitions

    def test_disjoint_range_empty(self, tmp_path_clean):
        df = _make_point_df(200)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        tbl = view.scan(z_range=(10.0, 20.0))  # outside any stats
        assert tbl.num_rows == 0


# ── Database integration ─────────────────────────────────────────────────


class TestDatabaseIntegration:
    def test_db_getitem_returns_view(self, tmp_path_clean):
        # Build a DB with one synth dataset.
        _write(tmp_path_clean, _make_point_df(75))
        db = OneuniverseDatabase.from_root(tmp_path_clean)
        name = next(iter(db))
        view = db[name]
        assert isinstance(view, DatasetView)
        assert view.n_rows == 75

    def test_db_view_method(self, tmp_path_clean):
        _write(tmp_path_clean, _make_point_df(50))
        db = OneuniverseDatabase.from_root(tmp_path_clean)
        name = next(iter(db))
        view = db.view(name)
        tbl = view.scan(columns=["ra", "dec"])
        assert tbl.num_rows == 50
        assert tbl.column_names == ["ra", "dec"]

    def test_db_unknown_name_raises(self, tmp_path_clean):
        _write(tmp_path_clean, _make_point_df(10))
        db = OneuniverseDatabase.from_root(tmp_path_clean)
        with pytest.raises(KeyError):
            db["nonexistent"]
