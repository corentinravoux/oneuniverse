"""
Core tests for oneuniverse — registry, schema, selections, dummy loader.

These tests require NO external data files (dummy loader is in-memory).
"""

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data import (
    Cone,
    Shell,
    SkyPatch,
    get_survey_config,
    list_survey_types,
    list_surveys,
    load_catalog,
)
from oneuniverse.data._registry import get_loader
from oneuniverse.data.schema import (
    COLUMN_GROUPS,
    get_all_columns,
    get_required_columns,
    validate_dataframe,
)


# ── Registry ──────────────────────────────────────────────────────────────


class TestRegistry:
    def test_list_surveys_returns_dict(self):
        surveys = list_surveys()
        assert isinstance(surveys, dict)
        assert "dummy" in surveys

    def test_list_surveys_filter_by_type(self):
        spectro = list_surveys(survey_type="spectroscopic")
        assert "eboss_qso" in spectro
        assert "dummy" not in spectro

    def test_list_survey_types(self):
        types = list_survey_types()
        assert "test" in types
        assert "spectroscopic" in types

    def test_get_survey_config(self):
        cfg = get_survey_config("dummy")
        assert cfg.name == "dummy"
        assert cfg.survey_type == "test"
        assert cfg.n_objects_approx == 5_000

    def test_get_loader_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown survey"):
            get_loader("nonexistent_survey")

    def test_eboss_qso_registered(self):
        cfg = get_survey_config("eboss_qso")
        assert cfg.survey_type == "spectroscopic"
        assert "qso" in cfg.column_groups
        assert cfg.data_format == "fits"
        assert cfg.data_subpath == "spectroscopic/eboss/qso"


# ── Schema ────────────────────────────────────────────────────────────────


class TestSchema:
    def test_column_groups_exist(self):
        for group in ["core", "spectroscopic", "photometric",
                       "peculiar_velocity", "qso", "snia"]:
            assert group in COLUMN_GROUPS

    def test_get_required_columns_core(self):
        required = get_required_columns(["core"])
        assert "ra" in required
        assert "dec" in required
        assert "z" in required
        assert "galaxy_id" in required

    def test_get_all_columns_qso(self):
        cols = get_all_columns(["core", "qso"])
        assert "is_qso" in cols
        assert "psfmag_r" in cols
        assert "z_lya" in cols
        assert "n_dla" in cols

    def test_validate_dataframe_pass(self):
        df = pd.DataFrame({
            "ra": [1.0, 2.0],
            "dec": [3.0, 4.0],
            "z": np.array([0.1, 0.2], dtype=np.float32),
            "galaxy_id": np.array([0, 1], dtype=np.int64),
            "survey_id": ["test", "test"],
        })
        warnings = validate_dataframe(df, ["core"])
        assert len(warnings) == 0

    def test_validate_dataframe_missing_required(self):
        df = pd.DataFrame({"ra": [1.0], "dec": [2.0]})
        with pytest.raises(ValueError, match="Required column 'z'"):
            validate_dataframe(df, ["core"])


# ── Selections ────────────────────────────────────────────────────────────


class TestSelections:
    def test_cone_center(self):
        cone = Cone(ra=180, dec=0, radius=1)
        ra = np.array([180.0, 180.5, 181.5])
        dec = np.array([0.0, 0.0, 0.0])
        z = np.array([0.1, 0.1, 0.1])
        mask = cone.mask(ra, dec, z)
        assert mask[0] and mask[1] and not mask[2]

    def test_shell(self):
        shell = Shell(z_min=0.5, z_max=1.5)
        z = np.array([0.3, 0.7, 1.0, 1.8])
        mask = shell.mask(np.zeros(4), np.zeros(4), z)
        np.testing.assert_array_equal(mask, [False, True, True, False])

    def test_skypatch_wrap(self):
        patch = SkyPatch(ra_min=350, ra_max=10, dec_min=-10, dec_max=10)
        ra = np.array([355, 5, 180])
        dec = np.array([0, 0, 0])
        z = np.array([0.1, 0.1, 0.1])
        mask = patch.mask(ra, dec, z)
        assert mask[0] and mask[1] and not mask[2]


# ── Dummy Loader ──────────────────────────────────────────────────────────


class TestDummyLoader:
    def test_load_basic(self):
        df = load_catalog("dummy")
        assert len(df) == 5000
        assert "ra" in df.columns
        assert "z" in df.columns
        assert "v_pec" in df.columns

    def test_load_reproducible(self):
        df1 = load_catalog("dummy")
        df2 = load_catalog("dummy")
        pd.testing.assert_frame_equal(df1, df2)

    def test_load_with_cone(self):
        df = load_catalog("dummy", selection=Cone(ra=180, dec=0, radius=10))
        assert len(df) < 5000
        assert len(df) > 0

    def test_load_with_shell(self):
        df = load_catalog("dummy", selection=Shell(0.05, 0.10))
        assert all(df["z"] >= 0.05)
        assert all(df["z"] <= 0.10)

    def test_load_column_subset(self):
        df = load_catalog("dummy", columns=["ra", "dec", "z"])
        assert list(df.columns) == ["ra", "dec", "z"]

    def test_load_invalid_column(self):
        with pytest.raises(ValueError, match="not in catalog"):
            load_catalog("dummy", columns=["ra", "nonexistent"])

    def test_info(self):
        loader = get_loader("dummy")
        info_str = loader.info()
        assert "dummy" in info_str
        assert "test" in info_str

    def test_coordinate_ranges(self):
        df = load_catalog("dummy")
        assert df["ra"].min() >= 0
        assert df["ra"].max() <= 360
        assert df["dec"].min() >= -90
        assert df["dec"].max() <= 90

    def test_custom_n_galaxies(self):
        df = load_catalog("dummy", n_galaxies=100, validate=False)
        assert len(df) == 100
