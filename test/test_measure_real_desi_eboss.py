"""Real-data validation — the measure layer on actual DESI DR1 + eBOSS DR16Q.

Loads the real QSO catalogs via the registered P1 loaders, builds an OUF POINT
view, and runs `build_galaxy_clustering` end-to-end, asserting the resulting
MeasurementSet is physically sane (real footprint, real n(z), randoms inside
the footprint, jackknife regions). Skipped automatically when the data files
are absent (CI-safe), exactly like `test_eboss.py`.
"""
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights import ConstantWeight
from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.measure import MeasurementSet, build_galaxy_clustering

DATA_ROOT = "/home/ravoux/Documents/Science/Cosmography/oneuniverse_data"
EBOSS = Path(DATA_ROOT) / "spectroscopic/eboss/qso/DR16Q_Superset_v3.fits"
DESI = Path(DATA_ROOT) / "spectroscopic/desi/dr1/qso/QSO_full.dat.fits"

skip_no_data = pytest.mark.skipif(
    not (EBOSS.exists() and DESI.exists()),
    reason="real DESI DR1 + eBOSS DR16Q data not available")


def _load(survey: str, zlo: float, zhi: float, cap: int) -> pd.DataFrame:
    os.environ["ONEUNIVERSE_DATA_ROOT"] = DATA_ROOT
    from oneuniverse.data import load_catalog
    df = load_catalog(survey, validate=False)
    df = df[(df["z"] >= zlo) & (df["z"] <= zhi)].dropna(subset=["ra", "dec", "z"])
    if len(df) > cap:
        df = df.sample(cap, random_state=0)
    return df.reset_index(drop=True)


def _to_ouf_view(df: pd.DataFrame, tmp: Path, name: str) -> DatasetView:
    n = len(df)
    ra = df["ra"].to_numpy(float); dec = df["dec"].to_numpy(float)
    out = pd.DataFrame({
        "ra": ra, "dec": dec, "z": df["z"].to_numpy(float),
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    od = tmp / name / "oneuniverse"
    write_ouf_dataset(df=out, out_dir=od, survey_name=name,
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(od.parent)


@skip_no_data
@pytest.mark.parametrize("survey,zlo,zhi", [("eboss_qso", 0.8, 2.2),
                                            ("desi_qso", 0.8, 2.2)])
def test_real_qso_clustering_measurement_set(tmp_path, survey, zlo, zhi):
    df = _load(survey, zlo, zhi, cap=60_000)
    assert len(df) > 5_000, f"{survey}: too few QSOs in z-range"
    view = _to_ouf_view(df, tmp_path, survey)

    ms = build_galaxy_clustering(
        view, tracer="qso", z_range=(zlo, zhi), weights=[ConstantWeight(1.0)],
        nz_edges=np.linspace(zlo - 0.1, zhi + 0.1, 30),
        nside_window=64, nside_region=8,
        randoms="generate", n_randoms=4 * len(df), seed=1)

    assert isinstance(ms, MeasurementSet)
    ps = ms.products["qso"]
    # real footprint: covers some sky but not all
    assert 0.0 < ps.window.covered_fraction() < 1.0
    # real n(z): non-empty and peaks inside the QSO redshift band
    assert ps.nz.counts.sum() > 0
    zpeak = ps.nz.centers()[int(ps.nz.counts.argmax())]
    assert zlo - 0.1 <= zpeak <= zhi + 0.1
    # generated randoms land inside the real footprint
    assert ps.window.contains(ps.randoms["ra"].to_numpy(),
                              ps.randoms["dec"].to_numpy()).all()
    # jackknife regions span the footprint
    assert len(np.unique(ps.region_map)) > 1
    ms.check_invariants()
    assert ms.summary()["cosmology_free"] is True
