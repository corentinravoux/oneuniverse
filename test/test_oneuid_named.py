"""Named ONEUID indices, audit columns, and restrict_to (Phase 4)."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
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
    list_oneuids,
    load_oneuid_index,
)
from oneuniverse.data.oneuid_rules import CrossMatchRules


# ── Fixtures (tiny synth datasets with two surveys sharing 3/5 rows) ────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _register_synth(name: str) -> str:
    class _Loader(BaseSurveyLoader):
        config = SurveyConfig(
            name=name, survey_type="spectroscopic",
            description=name,
            data_subpath=f"spectroscopic/{name}",
            data_filename="synth.csv", data_format="csv",
        )

        def _load_raw(self, data_path=None, **kwargs):
            return pd.read_csv(Path(data_path) / self.config.data_filename)

    _REGISTRY[name] = _Loader
    return name


def _make_csv(dir_: Path, ras, decs, zs) -> Path:
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
    csv = dir_ / "synth.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.fixture
def db_two(tmp_path_clean):
    tag = uuid.uuid4().hex[:8]
    raw_name_a = f"a{tag}"
    raw_name_b = f"b{tag}"
    _register_synth(raw_name_a)
    _register_synth(raw_name_b)
    # Dataset names inside the DB (derived from relpath by OneuniverseDatabase)
    ds_a = f"spectroscopic_{raw_name_a}"
    ds_b = f"spectroscopic_{raw_name_b}"
    raw_a = tmp_path_clean / "raw_a"
    raw_b = tmp_path_clean / "raw_b"

    truth_ra = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    truth_dec = np.array([-5.0, 0.0, 15.0, -20.0, 25.0])
    truth_z = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

    _make_csv(raw_a,
              truth_ra[:4] + 1e-5, truth_dec[:4] + 1e-5, truth_z[:4] + 1e-6)
    _make_csv(raw_b,
              truth_ra[[1, 2, 4]] + 2e-5,
              truth_dec[[1, 2, 4]] + 2e-5,
              truth_z[[1, 2, 4]] + 2e-6)

    db_root = tmp_path_clean / "db"
    convert_survey(survey_name=raw_name_a, raw_path=raw_a,
                   output_dir=db_root / f"spectroscopic/{raw_name_a}",
                   overwrite=True)
    convert_survey(survey_name=raw_name_b, raw_path=raw_b,
                   output_dir=db_root / f"spectroscopic/{raw_name_b}",
                   overwrite=True)
    db = OneuniverseDatabase(db_root)
    db._ds_a = ds_a  # expose for tests
    db._ds_b = ds_b
    yield db
    _REGISTRY.pop(raw_name_a, None)
    _REGISTRY.pop(raw_name_b, None)


# ── Audit columns on the built index ─────────────────────────────────────


class TestAuditColumns:
    def test_z_type_column_present(self, db_two):
        idx = db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
        )
        assert "z_type" in idx.table.columns
        assert set(idx.table["z_type"].unique()) <= {"spec", "phot", "none", "pv"}

    def test_survey_type_column_present(self, db_two):
        idx = db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
        )
        assert "survey_type" in idx.table.columns
        assert set(idx.table["survey_type"].unique()) == {"spectroscopic"}


# ── restrict_to ─────────────────────────────────────────────────────────


class TestRestrictTo:
    def test_filters_datasets(self, db_two):
        idx = db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
        )
        sub = idx.restrict_to([db_two._ds_a])
        assert set(sub.table["dataset"].unique()) == {db_two._ds_a}

    def test_reindexes_contiguously(self, db_two):
        idx = db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
        )
        sub = idx.restrict_to([db_two._ds_a])
        assert int(sub.table["oneuid"].min()) == 0
        assert int(sub.table["oneuid"].max()) == sub.n_unique - 1

    def test_preserves_rules(self, db_two):
        rules = CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3)
        idx = db_two.build_oneuid(rules=rules)
        sub = idx.restrict_to([db_two._ds_b])
        assert sub.rules is not None and sub.rules.hash() == rules.hash()

    def test_empty_restriction(self, db_two):
        idx = db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
        )
        sub = idx.restrict_to(["does_not_exist"])
        assert sub.n_unique == 0
        assert sub.table.empty


# ── Named indices, disk layout, rehydration ─────────────────────────────


class TestNamedIndices:
    def test_writes_parquet_and_manifest(self, db_two):
        db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
            name="default",
        )
        assert (db_two.root / ONEUID_DIR / "default.parquet").is_file()
        assert (db_two.root / ONEUID_DIR / "default.manifest.json").is_file()

    def test_multiple_named_indices_coexist(self, db_two):
        db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
            name="default",
        )
        db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=0.5, dz_tol_default=1e-3),
            name="tight",
        )
        names = set(db_two.list_oneuids())
        assert {"default", "tight"} <= names

    def test_list_oneuids_empty(self, db_two):
        assert db_two.list_oneuids() == []

    def test_load_roundtrip_preserves_rules(self, db_two):
        rules = CrossMatchRules(
            sky_tol_arcsec=2.0,
            dz_tol_default=1e-3,
            dz_tol_by_ztype={("spec", "phot"): 5e-2},
        )
        built = db_two.build_oneuid(rules=rules, name="mix")
        loaded = db_two.load_oneuid(name="mix")
        assert loaded.rules is not None
        assert loaded.rules.hash() == rules.hash()
        assert loaded.n_unique == built.n_unique
        assert loaded.n_multi == built.n_multi

    def test_manifest_records_rules_hash(self, db_two):
        rules = CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3)
        db_two.build_oneuid(rules=rules, name="m1")
        mf = db_two.root / ONEUID_DIR / "m1.manifest.json"
        data = json.loads(mf.read_text())
        assert data["rules"]["hash"] == rules.hash()
        assert data["name"] == "m1"
        assert set(data["datasets"]) == {db_two._ds_a, db_two._ds_b}

    def test_load_unknown_name_raises(self, db_two):
        db_two.build_oneuid(
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
            name="default",
        )
        with pytest.raises(FileNotFoundError):
            db_two.load_oneuid(name="does_not_exist")


# ── Dataset subsetting at build time ────────────────────────────────────


class TestBuildDatasetSubset:
    def test_build_with_subset_only_uses_requested(self, db_two):
        idx = db_two.build_oneuid(
            datasets=[db_two._ds_a],
            rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
            name="only_a",
        )
        assert set(idx.table["dataset"].unique()) == {db_two._ds_a}

    def test_build_unknown_dataset_raises(self, db_two):
        with pytest.raises(KeyError):
            db_two.build_oneuid(
                datasets=["nope"],
                rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
            )
