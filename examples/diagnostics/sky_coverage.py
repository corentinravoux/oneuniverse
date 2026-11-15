"""Sky footprint of a dataset: HEALPix density map and stored partitions.

Usage: python examples/diagnostics/sky_coverage.py [--out DIR] [--data-root DB --dataset NAME]
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, pick, save, setup_style, suptitle, synthetic_database

from oneuniverse.data import Database


def main() -> None:
    args = parse_args(__doc__)
    setup_style()
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(args.data_root) if args.data_root else synthetic_database(
            Path(tmp), n=40_000, ra_range=(110.0, 250.0), dec_range=(-15.0, 55.0))
        name = pick(db, args.dataset, 1)[0]
        view = db[name]
        t = view.table(columns=["ra", "dec", "z"])
        ra, dec, z = (t[c].to_numpy() for c in ("ra", "dec", "z"))
        cells = view.cells_for()
        nside = 32
        pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
        dens = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
        dens[dens == 0] = np.nan
        cell_map = np.full(hp.nside2npix(view.nside), np.nan)
        cell_map[cells] = np.bincount(
            hp.ang2pix(view.nside, ra, dec, lonlat=True, nest=True),
            minlength=hp.nside2npix(view.nside))[cells]

        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(2, 2, 1)
        sc = ax.scatter(ra, dec, c=z, s=1, rasterized=True, cmap="viridis")
        ax.set(xlabel="ra [deg]", ylabel="dec [deg]", title=f"{name}: objects ({len(ra):,})")
        fig.colorbar(sc, ax=ax, label="z")
        ax = fig.add_subplot(2, 2, 2)
        hist = ax.hist2d(ra, dec, bins=60, cmap="magma")
        ax.set(xlabel="ra [deg]", ylabel="dec [deg]", title="2D density")
        fig.colorbar(hist[3], ax=ax, label="counts / bin")
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(z, bins=50, color="C0")
        ax.set(xlabel="z", ylabel="objects", title="redshift distribution")
        hp.mollview(dens, nest=True, title=f"density, nside={nside}", fig=fig.number,
                    sub=(2, 2, 4), cmap="magma", cbar=True, notext=True)
        hp.projscatter(*np.array(hp.pix2ang(view.nside, cells, lonlat=True, nest=True)),
                       lonlat=True, s=6, c="cyan")
        suptitle(fig, f"Sky coverage of {name}: {len(cells)} of {hp.nside2npix(view.nside)} "
                 f"partition cells at nside={view.nside} (cyan)", args)
        fig.subplots_adjust(hspace=0.45, wspace=0.25, top=0.9)
        save(fig, args.out, "sky_coverage")


if __name__ == "__main__":
    main()
