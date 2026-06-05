#!/usr/bin/env python3
"""measure WL demo — cosmic-shear MeasurementSet diagnostic figure.

(a) ellipticity whisker map, (b) per-bin tomographic n(z), (c) region map.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oneuniverse.measure.lensing import build_cosmic_shear

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402

FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_weak_lensing.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    view = synthetic_shear_view(tmp, n=6000, seed=3, kind="metacal",
                                with_pdf=True, n_tomo=3)
    ms = build_cosmic_shear(view, tracer="src", kind="metacal",
                            tomo_column="tomo_bin",
                            z_grid=np.linspace(0, 2, 61), nside_region=8)
    ps = ms.products["src"]
    cat = ps.catalog

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    s = cat.iloc[::20]
    emag = np.hypot(s["e1"], s["e2"]); ang = 0.5 * np.arctan2(s["e2"], s["e1"])
    ax[0].quiver(s["ra"], s["dec"], emag * np.cos(ang), emag * np.sin(ang),
                 headwidth=0, headlength=0, pivot="mid", scale=8, width=0.003)
    ax[0].set_xlabel("RA [deg]"); ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title("shear ellipticity whiskers")

    for b, nz in sorted(ps.nz.items()):
        ax[1].plot(nz.centers(), nz.pdf(), lw=2, label=f"bin {b}")
    ax[1].set_xlabel("z"); ax[1].set_ylabel("n(z)")
    ax[1].set_title("tomographic n(z) (photo-z stack)"); ax[1].legend()

    sc = ax[2].scatter(cat["ra"], cat["dec"], s=3, c=cat["region_id"],
                       cmap="tab20")
    ax[2].set_xlabel("RA [deg]"); ax[2].set_ylabel("Dec [deg]")
    ax[2].set_title(f"region map (NSIDE={ps.metadata.nside_region})")
    plt.colorbar(sc, ax=ax[2], fraction=0.046, label="region_id")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("weight_recipe:", ps.provenance.weight_recipe,
          "| n_tomo:", len(ps.nz), "| figure:", FIG)


if __name__ == "__main__":
    main()
