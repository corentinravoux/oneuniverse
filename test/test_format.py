"""
Tests for the oneuniverse file format specification and converter.

Tests all three geometries (POINT, SIGHTLINE, HEALPIX) using synthetic data.
No external data files required.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import (
    convert_healpix_map,
    convert_sightlines,
    get_manifest,
    get_geometry,
    is_converted,
    read_objects_table,
    read_oneuniverse_parquet,
)
from oneuniverse.data.format_spec import (
    FORMAT_VERSION,
    DataGeometry,
    validate_columns,
)


@pytest.fixture
def tmp_path_clean():
    """Provide a fresh temp directory, cleaned up after the test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── Format spec ──────────────────────────────────────────────────────────


class TestFormatSpec:
    def test_geometry_enum(self):
        assert DataGeometry.POINT.value == "point"
        assert DataGeometry.SIGHTLINE.value == "sightline"
        assert DataGeometry.HEALPIX.value == "healpix"

    def test_validate_columns_point(self):
        missing = validate_columns(
            [
                "ra", "dec", "z", "z_type", "z_err",
                "galaxy_id", "survey_id",
                "_original_row_index", "_healpix32",
            ],
            DataGeometry.POINT,
        )
        assert missing == []

    def test_validate_columns_point_missing(self):
        missing = validate_columns(["ra", "dec"], DataGeometry.POINT)
        assert "z" in missing
        assert "galaxy_id" in missing
        assert "z_type" in missing
        assert "_healpix32" in missing

    def test_validate_columns_sightline_objects(self):
        cols = ["sightline_id", "ra", "dec", "z_source", "survey_id", "n_pixels"]
        missing = validate_columns(cols, DataGeometry.SIGHTLINE, "objects")
        assert missing == []

    def test_validate_columns_sightline_data(self):
        cols = ["sightline_id", "loglam", "delta", "weight"]
        missing = validate_columns(cols, DataGeometry.SIGHTLINE, "data")
        assert missing == []

    def test_validate_columns_healpix(self):
        missing = validate_columns(["healpix_index", "value"], DataGeometry.HEALPIX)
        assert missing == []


# ── SIGHTLINE converter ─────────────────────────────────────────────────


class TestSightlineConverter:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path_clean):
        self.tmp = tmp_path_clean
        rng = np.random.default_rng(42)
        n_sl = 50
        n_pix = 200

        self.objects = pd.DataFrame({
            "sightline_id": np.arange(n_sl, dtype=np.int64),
            "ra": rng.uniform(0, 360, n_sl),
            "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n_sl))),
            "z_source": rng.uniform(2.1, 3.5, n_sl).astype(np.float32),
            "survey_id": "test_lya",
            "n_pixels": np.full(n_sl, n_pix, dtype=np.int32),
        })
        self.data = pd.DataFrame({
            "sightline_id": np.repeat(np.arange(n_sl, dtype=np.int64), n_pix),
            "loglam": np.tile(
                np.linspace(3.5, 3.75, n_pix).astype(np.float32), n_sl
            ),
            "delta": rng.normal(0, 0.2, n_sl * n_pix).astype(np.float32),
            "weight": rng.uniform(0.5, 5.0, n_sl * n_pix).astype(np.float32),
        })

    def test_convert_and_read(self):
        convert_sightlines(
            self.objects, self.data, self.tmp, "test_lya", overwrite=True,
        )
        assert is_converted(self.tmp)
        assert get_geometry(self.tmp) == DataGeometry.SIGHTLINE

        df_obj = read_objects_table(self.tmp)
        assert len(df_obj) == 50
        assert "z_source" in df_obj.columns

        df_data = read_oneuniverse_parquet(self.tmp)
        assert len(df_data) == 50 * 200
        assert "delta" in df_data.columns

    def test_manifest_fields(self):
        convert_sightlines(
            self.objects, self.data, self.tmp, "test_lya", overwrite=True,
        )
        m = get_manifest(self.tmp)
        assert m.geometry.value == "sightline"
        assert m.extra["n_sightlines"] == 50
        assert m.extra["has_objects_table"] is True
        assert m.oneuniverse_format_version == FORMAT_VERSION

    def test_missing_columns_raises(self):
        bad_data = self.data.drop(columns=["delta"])
        with pytest.raises(ValueError, match="missing required columns"):
            convert_sightlines(
                self.objects, bad_data, self.tmp, "test", overwrite=True,
            )

    def test_overwrite(self):
        convert_sightlines(
            self.objects, self.data, self.tmp, "test_lya", overwrite=True,
        )
        convert_sightlines(
            self.objects, self.data, self.tmp, "test_lya", overwrite=True,
        )
        assert is_converted(self.tmp)


# ── HEALPIX converter ────────────────────────────────────────────────────


class TestHealpixConverter:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path_clean):
        self.tmp = tmp_path_clean
        nside = 16
        npix = 12 * nside**2
        rng = np.random.default_rng(42)
        self.data = pd.DataFrame({
            "healpix_index": np.arange(npix, dtype=np.int64),
            "value": rng.uniform(22.0, 25.0, npix).astype(np.float32),
        })
        self.nside = nside

    def test_convert_and_read(self):
        convert_healpix_map(
            self.data, self.tmp, "test_map", nside=self.nside, overwrite=True,
        )
        assert is_converted(self.tmp)
        assert get_geometry(self.tmp) == DataGeometry.HEALPIX

        df = read_oneuniverse_parquet(self.tmp)
        assert len(df) == 12 * self.nside**2
        assert "healpix_index" in df.columns
        assert "value" in df.columns

    def test_manifest_fields(self):
        convert_healpix_map(
            self.data, self.tmp, "test_map",
            nside=self.nside, ordering="ring", overwrite=True,
        )
        m = get_manifest(self.tmp)
        assert m.geometry.value == "healpix"
        assert m.extra["healpix_nside"] == self.nside
        assert m.extra["healpix_ordering"] == "ring"

    def test_missing_columns_raises(self):
        bad = self.data.drop(columns=["value"])
        with pytest.raises(ValueError, match="missing required columns"):
            convert_healpix_map(bad, self.tmp, "test", nside=16, overwrite=True)

    def test_objects_table_not_available(self):
        convert_healpix_map(
            self.data, self.tmp, "test_map", nside=self.nside, overwrite=True,
        )
        with pytest.raises(ValueError, match="No objects table"):
            read_objects_table(self.tmp)
