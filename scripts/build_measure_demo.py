#!/usr/bin/env python3
"""measure demo — galaxy-clustering MeasurementSet diagnostic figure.

Builds a MeasurementSet from a synthetic OUF POINT view and renders a 3-panel
diagnostic: (a) data vs generated randoms on-sky inside the footprint,
(b) weighted n(z) of data vs randoms, (c) the shared HEALPix region map.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure import build_galaxy_clustering

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402

FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_galaxy_clustering.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    view = synthetic_point_view(tmp, n=8000, seed=8)
    ms = build_galaxy_clustering(
        view, tracer="gal", z_range=(0.2, 0.9),
        weights=[FKPWeight(nbar=lambda z: np.full_like(z, 1e-3), P0=1e4),
                 ColumnWeight("weight_comp")],
        nside_window=64, nside_region=8,
        nz_edges=np.linspace(0.0, 1.2, 30),
        randoms="generate", n_randoms=40000, seed=1)
    ps = ms.products["gal"]
    cat, rnd, nz = ps.catalog, ps.randoms, ps.nz

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(rnd["ra"], rnd["dec"], s=1, c="0.7", label="randoms")
    ax[0].scatter(cat["ra"], cat["dec"], s=3, c="C0", label="data")
    ax[0].set_xlabel("RA [deg]"); ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title("footprint: data vs randoms"); ax[0].legend(markerscale=4)

    ctr = nz.centers()
    ax[1].plot(ctr, nz.pdf(), "C0-", lw=2, label="data n(z) (weighted)")
    rh, _ = np.histogram(rnd["z"], bins=nz.edges, density=True)
    ax[1].plot(ctr, rh, "C3--", lw=2, label="randoms n(z)")
    ax[1].set_xlabel("z"); ax[1].set_ylabel("n(z)")
    ax[1].set_title("radial selection match"); ax[1].legend()

    sc = ax[2].scatter(cat["ra"], cat["dec"], s=4, c=cat["region_id"],
                       cmap="tab20")
    ax[2].set_xlabel("RA [deg]"); ax[2].set_ylabel("Dec [deg]")
    ax[2].set_title(f"shared region map (NSIDE={ps.metadata.nside_region}, "
                    f"{cat['region_id'].nunique()} regions)")
    plt.colorbar(sc, ax=ax[2], fraction=0.046, label="region_id")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("randoms_source:", ps.provenance.randoms_source,
          "| nz_method:", ps.provenance.nz_method,
          "| weight_recipe:", ps.provenance.weight_recipe)
    print("figure:", FIG)


if __name__ == "__main__":
    main()
