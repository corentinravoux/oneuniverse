"""Diagnostic figures for the DR1 QSO onboarding pipeline.

Writes PNGs under ``test/test_output/phase9_*``. Fails only if the
file is not produced or is suspiciously small.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits  # noqa: E402

from oneuniverse.data.converter import convert_survey  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402


OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_desi_dr1_sky_and_z(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fake_desi_dr1_fits(raw, n_rows=5000, seed=9)
    ou = tmp_path / "db" / "desi_qso"
    convert_survey(
        "desi_qso", raw_path=raw, output_dir=ou, overwrite=True,
    )
    df = DatasetView.from_path(ou).read(columns=["ra", "dec", "z"])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
    axa.scatter(df["ra"], df["dec"], s=2, alpha=0.3)
    axa.set_xlabel("RA [deg]")
    axa.set_ylabel("Dec [deg]")
    axa.set_title(f"DESI DR1 QSO (fake) sky — n={len(df)}")
    axb.hist(df["z"], bins=60, color="C3", alpha=0.8)
    axb.set_xlabel("z")
    axb.set_ylabel("N")
    axb.set_title("redshift histogram")
    fig.tight_layout()
    out = OUT / "phase9_desi_dr1_sky_z.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
