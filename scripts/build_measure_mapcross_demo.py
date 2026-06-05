#!/usr/bin/env python3
"""measure map×catalog demo — HEALPix map (mollview) + galaxy footprint overlay."""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

from oneuniverse.measure.fieldmap import fieldmap_from_healpix
from oneuniverse.measure.mapcross import build_map_cross

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))
from fixtures.measure_ouf import synthetic_healpix_map, synthetic_point_view  # noqa: E402

FIG = (Path(__file__).resolve().parent.parent / "test" / "test_output"
       / "measure_map_cross.png")


def main():
    tmp = Path(tempfile.mkdtemp())
    gview = synthetic_point_view(tmp, n=6000, seed=3, name="gal")
    vals, mask = synthetic_healpix_map(nside=64, seed=4)
    fm = fieldmap_from_healpix(vals, mask=mask, nside=64, dataset_id="cmbk")
    ms = build_map_cross(gview, fm, nside_region=8, z_range=(0.1, 1.0))
    gcat = ms.products["gal"].catalog

    masked = np.where(fm.mask, fm.values, hp.UNSEEN)
    fig = plt.figure(figsize=(11, 5))
    hp.mollview(masked, nest=True, fig=fig.number, title="FieldMap (κ) + galaxy LOS",
                cmap="coolwarm", min=-3, max=3, hold=True)
    hp.projscatter(np.radians(90.0 - gcat["dec"].to_numpy()),
                   np.radians(gcat["ra"].to_numpy()), s=1, c="k", alpha=0.4)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110)
    print("pairs:", ms.spec.pairs, "| statistic:", ms.spec.statistic,
          "| figure:", FIG)


if __name__ == "__main__":
    main()
