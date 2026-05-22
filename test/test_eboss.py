"""eBOSS DR16Q loader tests.

These tests require the actual FITS data file. They are skipped if
the data is not available (CI-safe).

Phase 14 T2: session-scope shared `eboss_default_df` fixture lets all
the "default-load + column inspection" tests share one ~31s load,
saving ~90s of suite time on machines with the data.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_ROOT = "/home/ravoux/Documents/Science/Cosmography/oneuniverse_data"
FITS_FILE = Path(DATA_ROOT) / "spectroscopic/eboss/qso/DR16Q_Superset_v3.fits"

has_data = FITS_FILE.exists()
skip_no_data = pytest.mark.skipif(not has_data, reason="eBOSS data not available")


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def setup_data_root(monkeypatch_module):
    # Phase 12 removed set_data_root(); plumb the data root via env var.
    monkeypatch_module.setenv("ONEUNIVERSE_DATA_ROOT", str(DATA_ROOT))


@pytest.fixture(scope="module")
def eboss_default_df(setup_data_root):
    """Load eBOSS DR16Q once with default kwargs; share across tests."""
    if not has_data:
        pytest.skip("eBOSS data not available")
    from oneuniverse.data import load_catalog
    return load_catalog("eboss_qso", validate=False)


@skip_no_data
class TestEbossQSOLoader:

    @pytest.fixture(autouse=True)
    def _setup(self, setup_data_root):
        pass

    # ── Tests reusing the shared default-load df ──────────────────────────

    def test_load_qso_only(self, eboss_default_df):
        df = eboss_default_df
        assert len(df) > 900_000
        assert all(df["is_qso"] == 1)

    def test_columns_present(self, eboss_default_df):
        df = eboss_default_df
        expected = [
            "ra", "dec", "z", "is_qso", "source_z", "z_pipe", "z_pca",
            "zwarning", "psfmag_r", "extinction_r", "n_dla",
            "plate", "mjd", "fiberid", "survey_id", "galaxy_id",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_photometry_bands(self, eboss_default_df):
        df = eboss_default_df
        for band in "ugriz":
            assert f"psfmag_{band}" in df.columns
            assert f"extinction_{band}" in df.columns
        assert df["psfmag_r"].median() > 15
        assert df["psfmag_r"].median() < 25

    def test_dla_count(self, eboss_default_df):
        df = eboss_default_df
        assert "n_dla" in df.columns
        assert df["n_dla"].max() <= 5
        assert df["n_dla"].min() >= 0
        assert (df["n_dla"] > 0).sum() > 10_000

    # ── Tests exercising loader-side filters (one load each) ──────────────

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

    # ── Pure-metadata tests (no load) ────────────────────────────────────

    def test_config_metadata(self):
        from oneuniverse.data import get_survey_config
        cfg = get_survey_config("eboss_qso")
        assert cfg.data_subpath == "spectroscopic/eboss/qso"
        assert cfg.data_filename == "DR16Q_Superset_v3.fits"
        assert "Lyke" in cfg.reference
