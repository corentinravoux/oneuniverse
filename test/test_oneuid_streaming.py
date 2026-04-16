"""Phase 5 — streaming hydration: iter_partial + row-level pushdown."""

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


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _register_synth(name: str) -> str:
    class _Loader(BaseSurveyLoader):
        config = SurveyConfig(
            name=name, survey_type="test",
            description=name,
            data_subpath=f"test/{name}",
            data_filename="synth.csv", data_format="csv",
        )

        def _load_raw(self, data_path=None, **kwargs):
            return pd.read_csv(Path(data_path) / self.config.data_filename)

    _REGISTRY[name] = _Loader
    return name


def _make_csv(dir_: Path, ras, decs, zs, extra=None) -> Path:
    import healpy as hp
    dir_.mkdir(parents=True, exist_ok=True)
    n = len(ras)
    ras, decs = np.asarray(ras), np.asarray(decs)
    df = pd.DataFrame({
        "ra": ras, "dec": decs, "z": zs,
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
    name_a = _register_synth("stream_a")
    name_b = _register_synth("stream_b")
    raw_a = tmp_path_clean / "raw_a"
    raw_b = tmp_path_clean / "raw_b"
    truth_ra = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    truth_dec = np.array([-5.0, 0.0, 15.0, -20.0, 25.0])
    truth_z = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    _make_csv(
        raw_a,
        truth_ra[:4] + 1e-5, truth_dec[:4] + 1e-5, truth_z[:4] + 1e-6,
        extra={"value": [100.0, 110.0, 120.0, 130.0],
               "value_err": [5.0, 5.0, 5.0, 5.0]},
    )
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


# ── Task 1 — row-level pushdown ─────────────────────────────────────────


class TestRowLevelPushdown:
    def test_partial_for_bypasses_full_file_reads(
        self, two_overlapping_databases, monkeypatch,
    ):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uid = int(q.index.table["oneuid"].iloc[0])

        from oneuniverse.data import oneuid as mod
        calls = []
        real = mod.read_oneuniverse_parquet

        def _spy(*a, **kw):
            calls.append((a, kw))
            return real(*a, **kw)

        monkeypatch.setattr(mod, "read_oneuniverse_parquet", _spy)
        df = q.partial_for([uid], columns=["value"])
        assert calls == []
        assert len(df) >= 1
        assert "value" in df.columns

    def test_partial_for_preserves_row_order(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uids = q.index.table["oneuid"].unique().tolist()
        df = q.partial_for(uids, columns=["value"])
        # Each (dataset, row_index) should have been correctly aligned:
        # reloading the raw dataset and cross-checking value column.
        for ds, grp in df.groupby("dataset", sort=False):
            view = db[ds]
            raw = view.scan(columns=["value", "_original_row_index"]).to_pandas()
            raw_map = dict(zip(raw["_original_row_index"], raw["value"]))
            for row_idx, got in zip(grp["row_index"], grp["value"]):
                assert np.isclose(raw_map[row_idx], got)


# ── Task 2 — iter_partial generator contract ─────────────────────────────


class TestIterPartial:
    def test_yields_batches(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uids = q.index.table["oneuid"].unique().tolist()
        batches = list(q.iter_partial(uids, columns=["value"], batch_size=2))
        # 5 unique ONEUIDs, batch_size=2 → 3 batches
        assert len(batches) == 3
        assert all(isinstance(b, pd.DataFrame) for b in batches)
        assert all("value" in b.columns for b in batches)
        seen: set = set()
        for b in batches:
            this = set(b["oneuid"])
            assert this.isdisjoint(seen)
            seen |= this
        assert seen == set(uids)

    def test_matches_partial_for(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uids = q.index.table["oneuid"].unique().tolist()

        streamed = pd.concat(
            list(q.iter_partial(
                uids, columns=["value", "value_err"], batch_size=2,
            )),
            ignore_index=True,
        )
        materialised = q.partial_for(uids, columns=["value", "value_err"])
        key = ["oneuid", "dataset", "row_index"]
        streamed_sorted = streamed.sort_values(key).reset_index(drop=True)
        mat_sorted = materialised.sort_values(key).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            streamed_sorted, mat_sorted, check_dtype=False,
        )

    def test_default_batch_size_single_batch(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uids = q.index.table["oneuid"].unique().tolist()
        batches = list(q.iter_partial(uids, columns=["value"]))
        assert len(batches) == 1

    def test_empty_iter(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        assert list(q.iter_partial([], columns=["value"])) == []

    def test_batch_size_validation(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        with pytest.raises(ValueError, match="batch_size"):
            list(q.iter_partial([0], columns=["value"], batch_size=0))


# ── Task 3 — batch memory bound ─────────────────────────────────────────


class TestBatchBound:
    def test_batch_bound(self, two_overlapping_databases):
        db = two_overlapping_databases
        db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
        q = db.oneuid_query()
        uids = q.index.table["oneuid"].unique().tolist()
        n_ds = q.index.table["dataset"].nunique()
        for b in q.iter_partial(uids, columns=["value"], batch_size=2):
            assert len(b) <= 2 * n_ds
