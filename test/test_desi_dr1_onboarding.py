"""End-to-end onboarding tests for DESI DR1 QSO.

Each test builds a synthetic DR1-shaped FITS catalog and runs it through
one more stage of the Pillar-1 pipeline.  Failures here drive the
Phase-9 fragility audit (plans/2026-04-23-phase9-fragilities.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits  # noqa: E402

from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (  # noqa: E402
    DESIQSOLoader,
)


def test_loader_reads_fake_dr1(tmp_path):
    write_fake_desi_dr1_fits(tmp_path, n_rows=500, seed=1)
    df = DESIQSOLoader()._load_raw(data_path=tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert {"ra", "dec", "z", "z_spec_err", "zwarning"}.issubset(df.columns)
    assert (df["zwarning"] == 0).all()
    assert (df["z"] > 0).all()


def test_cone_query_prunes_partitions(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=5000, seed=3)

    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.selection import Cone

    ou_dir = tmp_path / "db" / "desi_qso"
    convert_survey(
        "desi_qso", raw_path=raw_dir, output_dir=ou_dir, overwrite=True,
    )

    view = DatasetView.from_path(ou_dir)
    center_ra, center_dec, radius = 180.0, 20.0, 5.0
    cone = Cone(ra=center_ra, dec=center_dec, radius=radius)
    df = view.read(cone=cone, columns=["ra", "dec"])
    assert len(df) > 0

    import numpy as np
    ra1 = np.radians(df["ra"].to_numpy())
    dec1 = np.radians(df["dec"].to_numpy())
    ra2 = np.radians(center_ra)
    dec2 = np.radians(center_dec)
    cos_d = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    sep_deg = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    assert (sep_deg <= radius + 1e-6).all()


def test_convert_and_reread(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=1000, seed=2)

    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.dataset_view import DatasetView

    ou_dir = tmp_path / "db" / "desi_qso"
    convert_survey(
        "desi_qso",
        raw_path=raw_dir,
        output_dir=ou_dir,
        overwrite=True,
    )

    view = DatasetView.from_path(ou_dir)
    df = view.read(columns=["ra", "dec", "z", "z_type"])
    assert len(df) > 0
    assert set(df["z_type"].unique()) <= {"spec"}
    assert df["ra"].between(0, 360).all()


def test_oneuid_single_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=1200, seed=4)

    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules

    db_root = tmp_path / "db"
    db_root.mkdir()
    convert_survey(
        "desi_qso",
        raw_path=raw_dir,
        output_dir=db_root / "desi_qso",
        overwrite=True,
    )

    db = OneuniverseDatabase(db_root)
    db.build_oneuid(
        datasets=["desi_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")
    assert idx.table["oneuid"].nunique() == len(idx.table)
    assert set(idx.table["dataset"].unique()) == {"desi_qso"}


def test_weighted_catalog_defaults(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=800, seed=5)

    import numpy as np
    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    from oneuniverse.combine import WeightedCatalog

    db_root = tmp_path / "db"
    db_root.mkdir()
    convert_survey(
        "desi_qso",
        raw_path=raw_dir,
        output_dir=db_root / "desi_qso",
        overwrite=True,
    )
    db = OneuniverseDatabase(db_root)
    db.build_oneuid(
        datasets=["desi_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")

    wc = WeightedCatalog.from_oneuid(idx, db)
    wc.fill_defaults(db)

    assert "desi_qso" in repr(wc)
    w = wc.total_weight("desi_qso")
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    assert len(w) == len(wc.catalogs["desi_qso"])
