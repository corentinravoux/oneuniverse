"""Tests for ONEUID — universal cross-survey identifier and the optimised
load_universal path."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data import OneuniverseDatabase, convert_survey
from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import _REGISTRY
from oneuniverse.data.oneuid import (
    ONEUID_DIR,
    OneuidIndex,
    build_oneuid_index,
    load_oneuid_index,
    load_universal,
)


# ── Helpers ──────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _register_synth(name: str) -> str:
    class _Loader(BaseSurveyLoader):
        config = SurveyConfig(
            name=name,
            survey_type="test",
            description=name,
            data_subpath=f"test/{name}",
            data_filename="synth.csv",
            data_format="csv",
        )

        def _load_raw(self, data_path=None, **kwargs):
            return pd.read_csv(Path(data_path) / self.config.data_filename)

    _REGISTRY[name] = _Loader
    return name


def _make_csv(dir_: Path, ras, decs, zs, extra=None) -> Path:
    import healpy as hp
    dir_.mkdir(parents=True, exist_ok=True)
    n = len(ras)
    ras = np.asarray(ras)
    decs = np.asarray(decs)
    df = pd.DataFrame({
        "ra": ras,
        "dec": decs,
        "z": zs,
        "z_type": ["spec"] * n,
        "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": "synth",
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": hp.ang2pix(
            32, np.radians(90.0 - decs), np.radians(ras), nest=True,
        ).astype(np.int32),
    })
    if extra is not None:
        for k, v in extra.items():
            df[k] = v
    csv = dir_ / "synth.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.fixture
def two_overlapping_databases(tmp_path_clean):
    """Build a database with two POINT datasets that share 3/5 objects."""
    name_a = _register_synth("oneuid_a")
    name_b = _register_synth("oneuid_b")
    raw_a = tmp_path_clean / "raw_a"
    raw_b = tmp_path_clean / "raw_b"

    # Five truth objects.
    truth_ra  = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    truth_dec = np.array([-5.0, 0.0, 15.0, -20.0, 25.0])
    truth_z   = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

    # Survey A sees objects 0,1,2,3 with tiny perturbations.
    _make_csv(
        raw_a,
        truth_ra[:4] + 1e-5,
        truth_dec[:4] + 1e-5,
        truth_z[:4] + 1e-6,
        extra={"value": [100.0, 110.0, 120.0, 130.0],
               "value_err": [5.0, 5.0, 5.0, 5.0]},
    )
    # Survey B sees objects 1,2,4 with tiny perturbations.
    _make_csv(
        raw_b,
        truth_ra[[1, 2, 4]] + 2e-5,
        truth_dec[[1, 2, 4]] + 2e-5,
        truth_z[[1, 2, 4]] + 2e-6,
        extra={"value": [115.0, 125.0, 155.0],
               "value_err": [3.0, 3.0, 3.0]},
    )
    db_root = tmp_path_clean / "db"
    convert_survey(survey_name=name_a, raw_path=raw_a,
                   output_dir=db_root / f"test/{name_a}", overwrite=True)
    convert_survey(survey_name=name_b, raw_path=raw_b,
                   output_dir=db_root / f"test/{name_b}", overwrite=True)
    db = OneuniverseDatabase(db_root)
    yield db
    _REGISTRY.pop(name_a, None)
    _REGISTRY.pop(name_b, None)


# ── Build / persist ──────────────────────────────────────────────────────


class TestBuildOneuid:
    def test_build_returns_index(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        assert isinstance(idx, OneuidIndex)
        # 4 (A) + 3 (B) rows but 3 shared objects ⇒ 4 + 3 − 3 = 4 unique?
        # A: 0,1,2,3 ; B: 1,2,4 → unique = {0,1,2,3,4} = 5
        assert idx.n_unique == 5
        # multi-survey ones: 1, 2 ⇒ 2 ONEUIDs
        assert idx.n_multi == 2

    def test_index_persisted_to_disk(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        f = db.root / ONEUID_DIR / "default.parquet"
        assert f.is_file()
        mf = db.root / ONEUID_DIR / "default.manifest.json"
        assert mf.is_file()

    def test_load_oneuid_round_trip(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        loaded = db.load_oneuid()
        assert loaded.n_unique == idx.n_unique
        assert loaded.n_multi == idx.n_multi
        assert loaded.sky_tol_arcsec == 2.0
        assert loaded.dz_tol == 1e-3
        pd.testing.assert_frame_equal(
            loaded.table.sort_values(["oneuid", "dataset"]).reset_index(drop=True),
            idx.table.sort_values(["oneuid", "dataset"]).reset_index(drop=True),
            check_dtype=False,
        )

    def test_build_no_persist(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(persist=False, sky_tol_arcsec=2.0, dz_tol=1e-3)
        assert not (db.root / ONEUID_DIR / "default.parquet").exists()


# ── Index query API ──────────────────────────────────────────────────────


class TestIndexQueries:
    def test_of_returns_concurrences(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        # Find a multi-survey ONEUID
        multi = idx.multi_only()
        assert len(multi) == 4  # 2 ONEUIDs × 2 datasets
        first_uid = int(multi["oneuid"].iloc[0])
        rows = idx.of(first_uid)
        assert set(rows["dataset"]) == {"test_oneuid_a", "test_oneuid_b"}

    def test_lookup(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        # Each (dataset, row_index) pair has a unique ONEUID
        uid = idx.lookup("test_oneuid_a", 0)
        assert isinstance(uid, int)
        with pytest.raises(KeyError):
            idx.lookup("test_oneuid_a", 999)

    def test_load_missing_index_raises(self, two_overlapping_databases):
        db = two_overlapping_databases
        with pytest.raises(FileNotFoundError):
            db.load_oneuid()


# ── Compact dtypes (Phase 6 Task 2) ──────────────────────────────────────


class TestDtypeCompaction:
    def test_dataset_column_categorical(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        assert idx.table["dataset"].dtype.name == "category"
        assert idx.table["survey_type"].dtype.name == "category"
        assert idx.table["z_type"].dtype.name == "category"

    def test_row_index_int32_when_small(self, two_overlapping_databases):
        db = two_overlapping_databases
        idx = db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        assert idx.table["row_index"].dtype == np.int32

    def test_dtypes_survive_disk_roundtrip(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3, name="compacted")
        idx = db.load_oneuid(name="compacted")
        assert idx.table["dataset"].dtype.name == "category"
        assert idx.table["row_index"].dtype == np.int32


# ── Optimised load_universal ─────────────────────────────────────────────


class TestLoadUniversal:
    def test_returns_oneuid_dataset_columns(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        df = db.load_universal(columns=["ra", "dec", "z", "value", "value_err"])
        # 4 (A) + 3 (B) = 7 rows
        assert len(df) == 7
        assert set(df.columns) >= {"oneuid", "dataset", "ra", "dec", "z", "value"}

    def test_only_multi_filters(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        df = db.load_universal(columns=["value", "value_err"], only_multi=True)
        # 2 multi ONEUIDs × 2 datasets = 4 rows
        assert len(df) == 4
        assert df["oneuid"].nunique() == 2

    def test_dataset_filter(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        df = db.load_universal(
            columns=["value"], datasets=["test_oneuid_a"],
        )
        assert set(df["dataset"]) == {"test_oneuid_a"}
        assert len(df) == 4

    def test_oneuid_query_factory(self, two_overlapping_databases):
        from oneuniverse.data.oneuid import OneuidQuery
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        assert isinstance(q, OneuidQuery)
        assert q.index.n_unique == 5

    def test_combine_with_weight_subpackage(self, two_overlapping_databases):
        """Hand off the universal table to oneuniverse.weight.combine_weights."""
        from oneuniverse.weight import combine_weights
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        df = db.load_universal(columns=["value", "value_err"])
        df["variance"] = df["value_err"] ** 2
        out = combine_weights(
            df.rename(columns={"oneuid": "universal_id", "dataset": "survey"}),
            value_col="value", variance_col="variance",
            strategy="best_only",
        )
        # 5 unique universal objects
        assert len(out) == 5
        # For multi-survey ones, B's smaller error should win.
        multi = out.table[out.table["n_surveys"] == 2]
        assert (multi["variance"] == 9.0).all()  # 3² = 9


# ── Tiered query API: selectors × hydration levels ──────────────────────


class TestOneuidQuery:
    @pytest.fixture(autouse=True)
    def _setup(self, two_overlapping_databases):
        self.db = two_overlapping_databases
        self.db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        self.q = self.db.oneuid_query()

    # ---- selectors ----

    def test_from_id(self):
        # Pick a known oneuid from the index
        first_uid = int(self.q.index.table["oneuid"].iloc[0])
        out = self.q.from_id(first_uid)
        assert out.tolist() == [first_uid]

    def test_from_id_missing_raises(self):
        with pytest.raises(KeyError):
            self.q.from_id(99999)

    def test_from_ids_filters_unknown(self):
        present = int(self.q.index.table["oneuid"].iloc[0])
        out = self.q.from_ids([present, 99999])
        assert out.tolist() == [present]

    def test_from_foreign_ids(self):
        # row_index 0 in survey A → some oneuid
        uid = self.q.from_foreign_ids("test_oneuid_a", [0])
        assert uid.shape == (1,)
        # Round-trip through index
        rows = self.q.concurrences(int(uid[0]))
        assert (rows[(rows["dataset"] == "test_oneuid_a")
                     & (rows["row_index"] == 0)]).shape[0] == 1

    def test_from_cone_picks_one_object(self):
        # Truth object 0 is at (10.0, -5.0)
        uids = self.q.from_cone(ra=10.0, dec=-5.0, radius=0.001)
        assert len(uids) == 1

    def test_from_shell(self):
        uids = self.q.from_shell(z_min=0.05, z_max=0.25)
        # objects 0 and 1 (z = 0.10, 0.20)
        assert len(uids) == 2

    def test_from_skypatch(self):
        uids = self.q.from_skypatch(ra_min=0.0, ra_max=25.0,
                                    dec_min=-10.0, dec_max=5.0)
        # Truth obj 0 (10,-5) and 1 (20,0) — 2 ONEUIDs
        assert len(uids) == 2

    def test_from_selection_compose(self):
        from oneuniverse.data import Cone, Shell
        uids = self.q.from_selection([
            Cone(ra=20.0, dec=0.0, radius=0.5),
            Shell(0.15, 0.25),
        ])
        assert len(uids) == 1

    # ---- hydration levels ----

    def test_index_for_zero_io(self):
        first_uid = int(self.q.index.table["oneuid"].iloc[0])
        df = self.q.index_for([first_uid])
        # Index columns only
        assert set(df.columns) >= {"oneuid", "dataset", "row_index", "ra", "dec", "z"}
        # No 'value' column ⇒ no dataset I/O
        assert "value" not in df.columns

    def test_partial_for_pushdown(self):
        # Multi-survey ONEUID
        multi = self.q.index.multi_only()
        uid = int(multi["oneuid"].iloc[0])
        df = self.q.partial_for([uid], columns=["value", "value_err"])
        assert "value" in df.columns and "value_err" in df.columns
        # 2 datasets × 1 oneuid = 2 rows
        assert len(df) == 2
        # No collision on ra/dec/z
        assert (df.columns == "ra").sum() == 1

    def test_partial_for_dataset_filter(self):
        multi = self.q.index.multi_only()
        uid = int(multi["oneuid"].iloc[0])
        df = self.q.partial_for(
            [uid], columns=["value"], datasets=["test_oneuid_a"],
        )
        assert set(df["dataset"]) == {"test_oneuid_a"}

    def test_full_for_loads_all_columns(self):
        multi = self.q.index.multi_only()
        uid = int(multi["oneuid"].iloc[0])
        df = self.q.full_for([uid])
        # full row from each dataset; the converted Parquet has the
        # _original_row_index column too
        assert "_original_row_index" in df.columns
        assert len(df) == 2  # 2 datasets

    def test_chain_cone_then_partial(self):
        uids = self.q.from_cone(ra=20.0, dec=0.0, radius=0.5)
        df = self.q.partial_for(uids, columns=["value"])
        # 1 universal × 2 datasets = 2 rows
        assert len(df) == 2
        assert "value" in df.columns

    def test_index_for_empty(self):
        df = self.q.index_for([99999])
        assert len(df) == 0

    def test_partial_for_empty(self):
        df = self.q.partial_for([99999], columns=["value"])
        assert "value" in df.columns
        assert len(df) == 0
