"""Tests for HEALPix spatial partitioning (Phase 3)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from oneuniverse.data import (
    Cone,
    DatasetView,
    SkyPatch,
    write_ouf_dataset,
)
from oneuniverse.data.format_spec import (
    HEALPIX_PARTITION_NEST,
    HEALPIX_PARTITION_NSIDE,
    ONEUNIVERSE_SUBDIR,
    DataGeometry,
)
from oneuniverse.data.manifest import PartitionStats


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_path_clean():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_point_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": rng.uniform(0.01, 0.5, n),
        "z_type": ["spec"] * n,
        "z_err": np.full(n, 1e-4, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": "synth",
        "_original_row_index": np.arange(n, dtype=np.int64),
        "_healpix32": hp.ang2pix(
            HEALPIX_PARTITION_NSIDE,
            np.radians(90.0 - dec),
            np.radians(ra),
            nest=HEALPIX_PARTITION_NEST,
        ).astype(np.int32),
    })


def _point_stats(chunk: pd.DataFrame) -> PartitionStats:
    return PartitionStats(
        ra_min=float(chunk["ra"].min()),
        ra_max=float(chunk["ra"].max()),
        dec_min=float(chunk["dec"].min()),
        dec_max=float(chunk["dec"].max()),
        z_min=float(chunk["z"].min()),
        z_max=float(chunk["z"].max()),
    )


def _write(tmp: Path, df: pd.DataFrame) -> Path:
    ou_dir = tmp / "synth" / ONEUNIVERSE_SUBDIR
    ou_dir.mkdir(parents=True, exist_ok=True)
    write_ouf_dataset(
        df=df,
        out_dir=ou_dir,
        survey_name="synth",
        survey_type="test",
        geometry=DataGeometry.POINT,
        stats_builder=_point_stats,
    )
    return ou_dir


# ── Converter layout ─────────────────────────────────────────────────────


class TestConverterLayout:
    def test_writes_one_file_per_cell(self, tmp_path_clean):
        df = _make_point_df(500)
        ou_dir = _write(tmp_path_clean, df)
        data_dir = ou_dir / "data"
        assert data_dir.is_dir()
        cell_dirs = sorted(data_dir.glob("healpix32=*"))
        observed = {int(d.name.split("=")[1]) for d in cell_dirs}
        expected = {int(c) for c in df["_healpix32"].unique()}
        assert observed == expected

    def test_partition_records_cell_id(self, tmp_path_clean):
        df = _make_point_df(200)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        cells_in_manifest = {
            p.healpix_cell for p in view.manifest.partitions
        }
        assert None not in cells_in_manifest
        assert cells_in_manifest == {int(c) for c in df["_healpix32"].unique()}

    def test_partitioning_spec_on_manifest(self, tmp_path_clean):
        df = _make_point_df(100)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        spec = view.manifest.partitioning
        assert spec is not None
        assert spec.scheme == "healpix"
        assert spec.column == "_healpix32"
        assert spec.extra["nside"] == HEALPIX_PARTITION_NSIDE


# ── Selection.healpix_cells ──────────────────────────────────────────────


class TestSelectionHealpix:
    def test_cone_cells_include_centre(self):
        c = Cone(ra=180.0, dec=0.0, radius=1.0)
        cells = c.healpix_cells(HEALPIX_PARTITION_NSIDE, nest=True)
        centre = hp.ang2pix(
            HEALPIX_PARTITION_NSIDE,
            np.radians(90.0),
            np.radians(180.0),
            nest=True,
        )
        assert int(centre) in set(cells.tolist())

    def test_cone_cells_sensible_count(self):
        # NSIDE=32 pixel ~1.8°; a 5° radius should cover a handful of cells.
        c = Cone(ra=45.0, dec=30.0, radius=5.0)
        cells = c.healpix_cells(HEALPIX_PARTITION_NSIDE, nest=True)
        assert 1 <= len(cells) < 200

    def test_skypatch_cells_nonempty(self):
        s = SkyPatch(ra_min=10, ra_max=20, dec_min=-5, dec_max=5)
        cells = s.healpix_cells(HEALPIX_PARTITION_NSIDE, nest=True)
        assert len(cells) > 0

    def test_skypatch_wraparound(self):
        wrap = SkyPatch(ra_min=350, ra_max=10, dec_min=-1, dec_max=1)
        straight = SkyPatch(ra_min=0, ra_max=10, dec_min=-1, dec_max=1)
        w = set(wrap.healpix_cells(HEALPIX_PARTITION_NSIDE).tolist())
        s = set(straight.healpix_cells(HEALPIX_PARTITION_NSIDE).tolist())
        # wrap should be a strict superset (covers 0-10 AND 350-360)
        assert s.issubset(w)


# ── DatasetView cell pruning ─────────────────────────────────────────────


class TestDatasetViewPruning:
    def test_cone_prunes_partitions(self, tmp_path_clean):
        df = _make_point_df(2000, seed=1)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        r0 = df.iloc[0]
        cone = Cone(ra=float(r0["ra"]), dec=float(r0["dec"]), radius=2.0)
        kept = view._select_partitions(
            healpix_cells=cone.healpix_cells(HEALPIX_PARTITION_NSIDE),
        )
        assert 0 < len(kept) < view.n_partitions

    def test_cone_scan_returns_only_in_cone_rows(self, tmp_path_clean):
        df = _make_point_df(2000, seed=2)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        r0 = df.iloc[0]
        cone = Cone(ra=float(r0["ra"]), dec=float(r0["dec"]), radius=2.0)
        tbl = view.scan(cone=cone)
        assert isinstance(tbl, pa.Table)
        assert tbl.num_rows >= 1
        # Every returned row must actually be within the cone.
        ras = np.array(tbl["ra"].to_pylist())
        decs = np.array(tbl["dec"].to_pylist())
        sep = _angular_sep(cone.ra, cone.dec, ras, decs)
        assert np.all(sep <= cone.radius + 1e-9)

    def test_skypatch_scan(self, tmp_path_clean):
        df = _make_point_df(3000, seed=3)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        patch = SkyPatch(ra_min=30, ra_max=60, dec_min=-10, dec_max=10)
        tbl = view.scan(skypatch=patch)
        ras = np.array(tbl["ra"].to_pylist())
        decs = np.array(tbl["dec"].to_pylist())
        assert np.all((ras >= 30) & (ras <= 60))
        assert np.all((decs >= -10) & (decs <= 10))
        # And matches a brute-force count from the raw DataFrame.
        expected = ((df["ra"] >= 30) & (df["ra"] <= 60)
                    & (df["dec"] >= -10) & (df["dec"] <= 10)).sum()
        assert tbl.num_rows == int(expected)

    def test_explicit_cells_kwarg(self, tmp_path_clean):
        df = _make_point_df(500, seed=4)
        ou_dir = _write(tmp_path_clean, df)
        view = DatasetView.from_ou_dir(ou_dir)
        # Pick one partition's cell, ask for it explicitly.
        first_cell = view.manifest.partitions[0].healpix_cell
        tbl = view.scan(healpix_cells=[first_cell])
        got_cells = {int(c) for c in tbl["_healpix32"].to_pylist()}
        assert got_cells == {int(first_cell)}


# ── Helpers ──────────────────────────────────────────────────────────────


def _angular_sep(ra1, dec1, ra2, dec2):
    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    dra = ra2 - ra1
    num = np.sqrt(
        (np.cos(dec2) * np.sin(dra)) ** 2
        + (np.cos(dec1) * np.sin(dec2)
           - np.sin(dec1) * np.cos(dec2) * np.cos(dra)) ** 2
    )
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.degrees(np.arctan2(num, den))
