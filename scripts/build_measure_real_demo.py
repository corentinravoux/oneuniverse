#!/usr/bin/env python3
"""measure real-data demo — eBOSS DR16Q QSO clustering MeasurementSet figure.

Runs only when the real eBOSS FITS is present. Renders the real footprint
(data vs generated randoms) + the real n(z). CI-safe (no-op without data).
"""
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd

DATA_ROOT = "/home/ravoux/Documents/Science/Cosmography/oneuniverse_data"
EBOSS = Path(DATA_ROOT) / "spectroscopic/eboss/qso/DR16Q_Superset_v3.fits"
FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_real_eboss.png")


def main():
    if not EBOSS.exists():
        print("eBOSS data absent — skipping real demo"); return
    os.environ["ONEUNIVERSE_DATA_ROOT"] = DATA_ROOT
    from oneuniverse.combine.weights import ConstantWeight
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data import load_catalog
    from oneuniverse.measure import build_galaxy_clustering

    df = load_catalog("eboss_qso", validate=False)
    df = df[(df["z"] >= 0.8) & (df["z"] <= 2.2)].dropna(subset=["ra", "dec", "z"])
    if len(df) > 80_000:
        df = df.sample(80_000, random_state=0)
    df = df.reset_index(drop=True); n = len(df)
    ra = df["ra"].to_numpy(float); dec = df["dec"].to_numpy(float)
    out = pd.DataFrame({"ra": ra, "dec": dec, "z": df["z"].to_numpy(float),
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4")})
    tmp = Path(tempfile.mkdtemp()) / "eboss" / "oneuniverse"
    write_ouf_dataset(df=out, out_dir=tmp, survey_name="eboss_real",
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name="eboss_real", version="0"))
    view = DatasetView.from_path(tmp.parent)
    ms = build_galaxy_clustering(view, tracer="qso", z_range=(0.8, 2.2),
        weights=[ConstantWeight(1.0)], nz_edges=np.linspace(0.7, 2.3, 33),
        nside_window=64, nside_region=16, randoms="generate",
        n_randoms=3 * n, seed=1)
    ps = ms.products["qso"]; cat, rnd, nz = ps.catalog, ps.randoms, ps.nz

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(rnd["ra"], rnd["dec"], s=.3, c=".75", label="randoms")
    ax[0].scatter(cat["ra"], cat["dec"], s=.6, c="C0", label="eBOSS QSO")
    ax[0].set_xlabel("RA [deg]"); ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title(f"real eBOSS DR16Q footprint (N={n})"); ax[0].legend(markerscale=8)
    ax[1].plot(nz.centers(), nz.pdf(), lw=2)
    ax[1].set_xlabel("z"); ax[1].set_ylabel("n(z)")
    ax[1].set_title("real eBOSS QSO redshift selection")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("covered_fraction:", round(ps.window.covered_fraction(), 4),
          "| regions:", len(np.unique(ps.region_map)), "| figure:", FIG)


if __name__ == "__main__":
    main()
