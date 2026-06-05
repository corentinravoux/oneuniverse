#!/usr/bin/env python3
"""measure PV/SN demo — (a) PV sky coloured by v_pec, (b) SN Hubble μ vs z."""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneuniverse.measure.pvsn import build_peculiar_velocity, build_sn_hubble

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from fixtures.measure_ouf import synthetic_pv_view, synthetic_sn_view  # noqa: E402

FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_pv_sn.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    pv = build_peculiar_velocity(synthetic_pv_view(tmp, n=3000, seed=3),
                                 z_range=(0.0, 0.1), nside_region=8)
    sview, n = synthetic_sn_view(tmp, n=400, seed=4)
    sn = build_sn_hubble(sview, z_range=(0.0, 1.5), nside_region=4)
    pcat = pv.products["pv"].catalog
    scat = sn.products["sn"].catalog

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sc = ax[0].scatter(pcat["ra"], pcat["dec"], s=5, c=pcat["v_pec"],
                       cmap="coolwarm", vmin=-600, vmax=600)
    ax[0].set_xlabel("RA [deg]"); ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title("peculiar velocities (v_pec)")
    plt.colorbar(sc, ax=ax[0], fraction=0.046, label="v_pec [km/s]")

    ax[1].errorbar(scat["z"], scat["mu"], yerr=scat["mu_err"], fmt=".",
                   ms=4, alpha=0.5, elinewidth=0.5)
    ax[1].set_xlabel("z"); ax[1].set_ylabel("distance modulus μ")
    ax[1].set_title("SN Ia Hubble diagram")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("PV family:", pv.spec.estimator_family,
          "| SN statistic:", sn.spec.statistic, "| figure:", FIG)


if __name__ == "__main__":
    main()
