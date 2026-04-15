"""
Tests for OneuniverseDatabase, convert_survey output_dir / raw_path,
force_native, columns subsetting on converted data, and config-driven build.

All tests use the dummy loader or synthetic data — no external files required.
"""

from __future__ import annotations

import shutil
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data import (
    OneuniverseDatabase,
    convert_survey,
    get_manifest,
    is_converted,
)
from oneuniverse.data.converter import read_oneuniverse_parquet
from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data._registry import _REGISTRY, register
from oneuniverse.data.config_loader import parse_config
from oneuniverse.data.format_spec import ONEUNIVERSE_SUBDIR


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def synth_point_loader():
    """Register a transient POINT loader reading a synthetic CSV."""
    name = "synth_point_tmp"

    class _SynthLoader(BaseSurveyLoader):
        config = SurveyConfig(
            name=name,
            survey_type="test",
            description="synthetic point catalog",
            data_subpath=f"test/{name}",
            data_filename="synth.csv",
            data_format="csv",
        )

        def _load_raw(self, data_path=None, **kwargs):
            csv_path = Path(data_path) / self.config.data_filename
            return pd.read_csv(csv_path)

    _REGISTRY[name] = _SynthLoader
    yield name, _SynthLoader
    _REGISTRY.pop(name, None)


def _make_synth_csv(dir_: Path, n: int = 100) -> Path:
    import healpy as hp
    dir_.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    df = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": rng.uniform(0.01, 0.5, n),
        "z_type": ["spec"] * n,
        "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": "synth",
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": hp.ang2pix(
            32, np.radians(90.0 - dec), np.radians(ra), nest=True,
        ).astype(np.int32),
    })
    csv = dir_ / "synth.csv"
    df.to_csv(csv, index=False)
    return csv


# ── convert_survey(output_dir, raw_path) ────────────────────────────────


class TestConvertSurveyNewParams:
    def test_output_dir_places_files_elsewhere(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        out_dir = tmp_path_clean / "db"
        _make_synth_csv(raw_dir)

        result = convert_survey(
            survey_name=name,
            raw_path=raw_dir,
            output_dir=out_dir,
            overwrite=True,
        )
        assert result == out_dir / ONEUNIVERSE_SUBDIR
        assert is_converted(out_dir)
        # Raw directory untouched
        assert not (raw_dir / ONEUNIVERSE_SUBDIR).exists()

    def test_raw_path_accepts_file_or_dir(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        csv = _make_synth_csv(raw_dir)
        out_dir = tmp_path_clean / "db"

        # Pass the file itself
        convert_survey(
            survey_name=name,
            raw_path=csv,
            output_dir=out_dir,
            overwrite=True,
        )
        df = read_oneuniverse_parquet(out_dir)
        assert len(df) == 100
        assert "_original_row_index" in df.columns

    def test_manifest_records_geometry_and_rows(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir, n=250)
        out_dir = tmp_path_clean / "db"
        convert_survey(
            survey_name=name, raw_path=raw_dir,
            output_dir=out_dir, overwrite=True,
        )
        m = get_manifest(out_dir)
        assert m["geometry"] == "point"
        assert m["n_rows"] == 250
        assert m["has_objects_table"] is False


# ── BaseSurveyLoader: columns, force_native ─────────────────────────────


class TestLoaderColumnsAndForceNative:
    def test_columns_subset_on_converted_parquet(
        self, tmp_path_clean, synth_point_loader
    ):
        name, loader_cls = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir)
        out_dir = tmp_path_clean / "db"
        convert_survey(
            survey_name=name, raw_path=raw_dir,
            output_dir=out_dir, overwrite=True,
        )

        loader = loader_cls()
        df = loader.load(
            data_path=out_dir,
            columns=["ra", "dec"],
            validate=False,
        )
        assert list(df.columns) == ["ra", "dec"]

    def test_force_native_bypasses_parquet(
        self, tmp_path_clean, synth_point_loader
    ):
        name, loader_cls = synth_point_loader
        # Raw dir *contains* the CSV AND an oneuniverse/ dir with different data.
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir, n=100)

        # Convert from raw_dir into raw_dir itself to get an oneuniverse/ subdir
        convert_survey(
            survey_name=name, raw_path=raw_dir,
            output_dir=raw_dir, overwrite=True,
        )
        assert is_converted(raw_dir)

        loader = loader_cls()
        df_parq = loader.load(data_path=raw_dir, validate=False)
        df_native = loader.load(
            data_path=raw_dir, validate=False, force_native=True,
        )
        # OUF 2.0 requires _original_row_index on both paths (CORE col).
        assert "_original_row_index" in df_parq.columns
        assert "_original_row_index" in df_native.columns
        assert len(df_parq) == len(df_native) == 100


# ── OneuniverseDatabase ──────────────────────────────────────────────────


class TestOneuniverseDatabase:
    def test_scan_discovers_datasets(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir)
        db_root = tmp_path_clean / "db"
        (db_root / "test" / "synth").mkdir(parents=True)
        convert_survey(
            survey_name=name, raw_path=raw_dir,
            output_dir=db_root / "test" / "synth",
            overwrite=True,
        )

        db = OneuniverseDatabase(db_root)
        assert len(db) == 1
        assert "test_synth" in db
        assert db.types() == ["test"]

        df = db.load("test_synth", columns=["ra", "dec", "z"])
        assert df.shape == (100, 3)

    def test_build_mirrors_tree(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_root = tmp_path_clean / "raws"
        (raw_root / f"test/{name}").mkdir(parents=True)
        _make_synth_csv(raw_root / f"test/{name}")
        db_root = tmp_path_clean / "db"

        db = OneuniverseDatabase.build(raw_root, db_root, overwrite=True)
        assert len(db) >= 1
        assert any(name in n for n in db)
        # Database tree mirrors the raw subpath
        assert (db_root / f"test/{name}" / ONEUNIVERSE_SUBDIR).is_dir()

    def test_get_manifest_and_path(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_root = tmp_path_clean / "raws"
        (raw_root / f"test/{name}").mkdir(parents=True)
        _make_synth_csv(raw_root / f"test/{name}")
        db = OneuniverseDatabase.build(
            raw_root, tmp_path_clean / "db", overwrite=True,
        )
        key = next(iter(db))
        m = db.get_manifest(key)
        assert "geometry" in m
        assert db.get_path(key).is_dir()


# ── Config-driven build ─────────────────────────────────────────────────


class TestFromConfig:
    def test_parse_config_reads_sections(self, tmp_path_clean):
        cfg_path = tmp_path_clean / "oneuniverse.ini"
        cfg_path.write_text(textwrap.dedent(f"""
            [database]
            root = {tmp_path_clean / "db"}
            overwrite = true

            [dataset a]
            loader   = synth_point_tmp
            raw_path = /nowhere/a.csv
            some_int = 42
            some_bool = true
        """))
        db_settings, datasets = parse_config(cfg_path)
        assert db_settings["overwrite"] is True
        assert len(datasets) == 1
        assert datasets[0]["loader"] == "synth_point_tmp"
        assert datasets[0]["kwargs"] == {"some_int": 42, "some_bool": True}

    def test_parse_config_missing_database_section_raises(self, tmp_path_clean):
        p = tmp_path_clean / "bad.ini"
        p.write_text("[dataset x]\nloader = foo\n")
        with pytest.raises(ValueError, match="database"):
            parse_config(p)

    def test_from_config_builds_database(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir)
        csv = raw_dir / "synth.csv"
        db_root = tmp_path_clean / "db"
        cfg = tmp_path_clean / "oneuniverse.ini"
        cfg.write_text(textwrap.dedent(f"""
            [database]
            root = {db_root}
            overwrite = true

            [dataset synth]
            loader   = {name}
            raw_path = {csv}
            output_subpath = mytype/myname
        """))
        db = OneuniverseDatabase.from_config(cfg)
        assert "mytype_myname" in db
        df = db.load("mytype_myname", columns=["ra", "dec"])
        assert df.shape == (100, 2)

    def test_from_config_skip_flag(
        self, tmp_path_clean, synth_point_loader
    ):
        name, _ = synth_point_loader
        raw_dir = tmp_path_clean / "raw"
        _make_synth_csv(raw_dir)
        db_root = tmp_path_clean / "db"
        cfg = tmp_path_clean / "oneuniverse.ini"
        cfg.write_text(textwrap.dedent(f"""
            [database]
            root = {db_root}

            [dataset a]
            loader   = {name}
            raw_path = {raw_dir / "synth.csv"}
            skip     = true
        """))
        db = OneuniverseDatabase.from_config(cfg)
        assert len(db) == 0
