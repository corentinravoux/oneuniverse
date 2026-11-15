"""Cone query pruning: partitions read versus total, with the selected cells highlighted.

Usage: python examples/diagnostics/partition_pruning.py [--out DIR] [--data-root DB --dataset NAME]
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, pick, point_table, save, setup_style, suptitle, write_points

from oneuniverse.data import Cone, Database, DatasetView


def main() -> None:
    args = parse_args(__doc__)
    setup_style()
    with tempfile.TemporaryDirectory() as tmp:
        if args.data_root:
            db = Database(args.data_root)
            view = db[pick(db, args.dataset, 1)[0]]
        else:
            write_points(Path(tmp), "survey", point_table(60_000, 0, ra_range=(100.0, 250.0),
                                                          dec_range=(-30.0, 60.0)),
                         partition_nside=8)
            view = DatasetView.from_path(Path(tmp) / "survey")
        cells = view.cells_for()
        nside = view.nside
        ra_c = float(np.median(view.table(columns=["ra"])["ra"].to_numpy()))
        dec_c = float(np.median(view.table(columns=["dec"])["dec"].to_numpy()))
        cone = Cone(ra_c, dec_c, 8.0)
        hit = view.cells_for(cone=cone)
        sub = view.table(columns=["ra", "dec"], cone=cone)

        radii = np.array([1, 2, 4, 8, 16, 32, 64.0])
        read = [len(view.cells_for(cone=Cone(ra_c, dec_c, r))) for r in radii]
        rows = [view.table(columns=["ra"], cone=Cone(ra_c, dec_c, r)).num_rows for r in radii]

        state = np.full(hp.nside2npix(nside), np.nan)
        state[cells] = 0.0
        state[hit] = 1.0
        fig = plt.figure(figsize=(13, 6.5))
        hp.mollview(state, nest=True, fig=fig.number, sub=(1, 2, 1), cbar=False, notext=True,
                    title=f"nside={nside}: {len(hit)} of {len(cells)} cells read "
                          f"(red); grey = pruned", cmap="coolwarm", min=-0.3, max=1.3)
        hp.projscatter(ra_c, dec_c, lonlat=True, c="k", marker="+", s=80)
        ax = fig.add_subplot(2, 2, 2)
        ax.scatter(*view.table(columns=["ra", "dec"]).to_pandas().to_numpy().T, s=0.3,
                   c="0.7", rasterized=True, label="all objects")
        ax.scatter(sub["ra"].to_numpy(), sub["dec"].to_numpy(), s=1.5, c="C3", rasterized=True,
                   label=f"cone r=8 deg ({sub.num_rows:,})")
        ax.set(xlabel="ra [deg]", ylabel="dec [deg]", title="cone selection")
        ax.legend(loc="upper right", markerscale=6, fontsize=8)
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(radii, 100 * np.array(read) / len(cells), "o-", label="cells read / total")
        ax.plot(radii, 100 * np.array(rows) / view.n_rows, "s--", label="rows returned / total")
        ax.set(xscale="log", xlabel="cone radius [deg]", ylabel="percent",
               title="pruning versus radius")
        ax.legend()
        suptitle(fig, f"Partition pruning of {view.n_rows:,} objects", args)
        fig.subplots_adjust(hspace=0.45, wspace=0.2, top=0.88)
        save(fig, args.out, "partition_pruning")


if __name__ == "__main__":
    main()
