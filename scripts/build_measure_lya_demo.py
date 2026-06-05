#!/usr/bin/env python3
"""measure Lyα demo — (a) example δ_F(λ) sightlines, (b) LOS sky by region."""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneuniverse.measure.lya import build_lya

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from fixtures.measure_ouf import synthetic_sightline_view  # noqa: E402

FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_lya.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    view = synthetic_sightline_view(tmp, n_los=200, n_pix=60, seed=2)
    ms = build_lya(view, statistic="p1d", nside_region=8)
    sl = ms.products["lya"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i in range(6):
        ax[0].plot(sl.delta[i] + i * 1.2, lw=0.8)
    ax[0].set_xlabel("pixel (λ)"); ax[0].set_ylabel("δ_F + offset")
    ax[0].set_title("6 example Lyα sightlines (δ_F)")

    sc = ax[1].scatter(sl.los["ra"], sl.los["dec"], s=10,
                       c=sl.los["region_id"], cmap="tab20")
    ax[1].set_xlabel("RA [deg]"); ax[1].set_ylabel("Dec [deg]")
    ax[1].set_title(f"{sl.n_sightlines} LOS (region NSIDE={sl.metadata.nside_region})")
    plt.colorbar(sc, ax=ax[1], fraction=0.046, label="region_id")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("n_sightlines:", sl.n_sightlines, "| statistic:", ms.spec.statistic,
          "| figure:", FIG)


if __name__ == "__main__":
    main()
