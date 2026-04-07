"""
eBOSS DR16Q loader tests.

These tests require the actual FITS data file. They are skipped if
the data is not available (CI-safe).
"""

import os
from pathlib import Path

import numpy as np
import pytest

DATA_ROOT = "/home/ravoux/Documents/Science/Cosmography/oneuniverse_data"
FITS_FILE = Path(DATA_ROOT) / "spectroscopic/eboss/qso/DR16Q_Superset_v3.fits"

has_data = FITS_FILE.exists()
skip_no_data = pytest.mark.skipif(not has_data, reason="eBOSS data not available")


@pytest.fixture(scope="module")
def setup_data_root():
    from oneuniverse.data import set_data_root
    set_data_root(DATA_ROOT)


@skip_no_data
class TestEbossQSOLoader:

    @pytest.fixture(autouse=True)
    def _setup(self, setup_data_root):
        pass

    def test_load_qso_only(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", validate=False)
        assert len(df) > 900_000
        assert all(df["is_qso"] == 1)

    def test_load_full_superset(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", qso_only=False, validate=False)
        assert len(df) > 1_400_000
        assert set(df["is_qso"].unique()) == {-2, 0, 1, 2}

    def test_z_range_filter(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", z_min=2.0, z_max=3.0, validate=False)
        assert all(df["z"] >= 2.0)
        assert all(df["z"] <= 3.0)
        assert len(df) > 100_000

    def test_columns_present(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", validate=False)
        expected = [
            "ra", "dec", "z", "is_qso", "source_z", "z_pipe", "z_pca",
            "zwarning", "psfmag_r", "extinction_r", "n_dla",
            "plate", "mjd", "fiberid", "survey_id", "galaxy_id",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_photometry_bands(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", validate=False)
        for band in "ugriz":
            assert f"psfmag_{band}" in df.columns
            assert f"extinction_{band}" in df.columns
        # Magnitudes should be reasonable (not all zero or NaN)
        assert df["psfmag_r"].median() > 15
        assert df["psfmag_r"].median() < 25

    def test_cone_selection(self):
        from oneuniverse.data import Cone, load_catalog
        df = load_catalog(
            "eboss_qso",
            selection=Cone(ra=185, dec=15, radius=2),
            validate=False,
        )
        assert len(df) > 0
        assert len(df) < 10_000

    def test_column_subset(self):
        from oneuniverse.data import load_catalog
        df = load_catalog(
            "eboss_qso",
            columns=["ra", "dec", "z"],
            validate=False,
        )
        assert list(df.columns) == ["ra", "dec", "z"]

    def test_dla_count(self):
        from oneuniverse.data import load_catalog
        df = load_catalog("eboss_qso", validate=False)
        assert "n_dla" in df.columns
        assert df["n_dla"].max() <= 5
        assert df["n_dla"].min() >= 0
        assert (df["n_dla"] > 0).sum() > 10_000

    def test_config_metadata(self):
        from oneuniverse.data import get_survey_config
        cfg = get_survey_config("eboss_qso")
        assert cfg.data_subpath == "spectroscopic/eboss/qso"
        assert cfg.data_filename == "DR16Q_Superset_v3.fits"
        assert "Lyke" in cfg.reference
