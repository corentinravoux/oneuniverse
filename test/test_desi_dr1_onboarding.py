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
