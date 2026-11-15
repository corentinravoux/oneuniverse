"""MeasurementSet overview: point set and randoms on the sky, n(z), jackknife regions, window.

Usage: python examples/diagnostics/measurement_set_overview.py [--out DIR]
                                                               [--data-root DB --dataset NAME]
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, pick, point_table, save, setup_style, suptitle, write_points

from oneuniverse.data import Database, DatasetView
from oneuniverse.measure import galaxy_clustering


def main() -> None:
    args = parse_args(__doc__)
    setup_style()
    with tempfile.TemporaryDirectory() as tmp:
        if args.data_root:
            db = Database(args.data_root)
            view = db[pick(db, args.dataset, 1)[0]]
            zr = (0.2, 1.0)
        else:
            write_points(Path(tmp), "gal", point_table(30_000, 0, ra_range=(110.0, 170.0),
                                                       dec_range=(-5.0, 35.0), z_range=(0.1, 1.2)))
            view = DatasetView.from_path(Path(tmp) / "gal")
            zr = (0.2, 1.0)
        ms = galaxy_clustering(view, z=zr, nz_edges=np.linspace(zr[0], zr[1], 17), regions=12,
                               random_factor=5)
        prod = ms.products["gal"]
        cat, rnd, nz = prod.catalog, prod.randoms, prod.nz
        reg = ms.regions
        ra_c, dec_c, rid = (cat[c].to_numpy() for c in ("ra", "dec", "region_id"))
        ra_r, dec_r = rnd["ra"].to_numpy(), rnd["dec"].to_numpy()

        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax[0, 0].scatter(ra_c, dec_c, s=1, c="C0", rasterized=True)
        ax[0, 0].set(xlabel="ra [deg]", ylabel="dec [deg]", title=f"catalog ({cat.num_rows:,})")
        ax[0, 1].scatter(ra_r, dec_r, s=0.5, c="C1", rasterized=True)
        ax[0, 1].set(xlabel="ra [deg]", ylabel="dec [deg]", title=f"randoms ({rnd.num_rows:,})")
        sc = ax[0, 2].scatter(ra_c, dec_c, s=1, c=rid, cmap="tab20", rasterized=True)
        ax[0, 2].set(xlabel="ra [deg]", ylabel="dec [deg]",
                     title=f"{reg.n_regions} jackknife regions ({reg.method})")
        fig.colorbar(sc, ax=ax[0, 2], label="region_id")
        ax[1, 0].stairs(nz.counts / nz.counts.sum() / np.diff(nz.edges), nz.edges, label="catalog n(z)")
        ax[1, 0].hist(rnd["z"].to_numpy(), bins=nz.edges, density=True, histtype="step",
                      color="C1", label="randoms z")
        ax[1, 0].set(xlabel="z", ylabel="normalised n(z)", title=f"n(z) [{nz.method}]")
        ax[1, 0].legend()
        ax[1, 1].hist(rid, bins=np.arange(reg.n_regions + 1) - 0.5, color="C2", label="catalog")
        counts_r = np.bincount(rnd["region_id"].to_numpy(), minlength=reg.n_regions)
        ax[1, 1].step(np.arange(reg.n_regions), counts_r / (rnd.num_rows / cat.num_rows),
                      where="mid", c="k", label="randoms (scaled)")
        ax[1, 1].set(xlabel="region_id", ylabel="objects", title="counts per region")
        ax[1, 1].legend()
        ax[1, 2].remove()
        win = prod.window
        m = np.where(win.mask > 0, win.mask, hp.UNSEEN)
        hp.mollview(m, nest=win.nest, fig=fig.number, sub=(2, 3, 6), notext=True, cbar=False,
                    title=f"window (nside={win.nside}, {win.area_sr * (180 / np.pi) ** 2:.0f} deg$^2$)")
        suptitle(fig, f"MeasurementSet '{ms.spec.name}': tracers {list(ms.spec.tracers)}, "
                 f"statistics {list(ms.spec.statistics)}", args)
        fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92)
        save(fig, args.out, "measurement_set_overview")


if __name__ == "__main__":
    main()
