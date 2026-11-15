"""ONEUID cross-match of two surveys: separation and delta-z histograms of matched pairs.

Note: with --data-root the ONEUID index "diagnostic" is written into that database.

Usage: python examples/diagnostics/crossmatch_separations.py [--out DIR]
                                                              [--data-root DB --dataset A --dataset B]
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, pick, save, setup_style, synthetic_database

from oneuniverse.data import CrossMatchRules, Database


def main() -> None:
    args = parse_args(__doc__)
    setup_style()
    rules = CrossMatchRules(sky_tol_arcsec=1.0, dz_tol=5e-3)
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(args.data_root) if args.data_root else synthetic_database(Path(tmp))
        names = pick(db, args.dataset, 2)
        idx = db.build_oneuid(names, rules=rules, name="diagnostic")
        multi = idx.multi_only()
        oid = multi["oneuid"].to_numpy()
        ds = multi["dataset"].to_numpy(zero_copy_only=False)
        ra, dec, z = (multi[c].to_numpy() for c in ("ra", "dec", "z"))
        first = ds == names[0]
        # pairs: one row of each dataset per oneuid (first/second occurrence)
        a = {o: i for o, i in zip(oid[first], np.flatnonzero(first), strict=True)}
        sep, dz, zz = [], [], []
        for i in np.flatnonzero(~first):
            j = a.get(oid[i])
            if j is None:
                continue
            dra = (ra[i] - ra[j]) * np.cos(np.radians(0.5 * (dec[i] + dec[j])))
            sep.append(3600 * np.hypot(dra, dec[i] - dec[j]))
            dz.append(z[i] - z[j])
            zz.append(z[j])
        sep, dz, zz = map(np.asarray, (sep, dz, zz))

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].hist(sep, bins=60, color="C0")
        ax[0, 0].axvline(rules.sky_tol_arcsec, c="k", ls="--", label="sky_tol")
        ax[0, 0].set(xlabel="separation [arcsec]", ylabel="pairs", title="matched-pair separation")
        ax[0, 0].legend()
        ax[0, 1].hist(dz, bins=60, color="C1")
        for s in (-1, 1):
            ax[0, 1].axvline(s * rules.dz_tol, c="k", ls="--")
        ax[0, 1].set(xlabel=r"$\Delta z$", ylabel="pairs", title=r"matched-pair $\Delta z$")
        ax[1, 0].scatter(sep, dz, s=2, alpha=0.4, rasterized=True)
        ax[1, 0].set(xlabel="separation [arcsec]", ylabel=r"$\Delta z$", title="joint")
        counts = np.bincount(np.unique(oid, return_counts=True)[1])
        counts[1] = idx.n_objects - idx.n_multi
        ax[1, 1].bar(np.arange(1, len(counts)), counts[1:], color="C2")
        ax[1, 1].set_xticks(np.arange(1, len(counts)))
        ax[1, 1].set(xlabel="members per ONEUID", ylabel="objects", yscale="log",
                     title=f"{idx.n_objects:,} objects, {idx.n_multi:,} multi-survey")
        fig.suptitle(f"Cross-match {names[0]} x {names[1]} ({len(sep):,} pairs)")
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
        save(fig, args.out, "crossmatch_separations")


if __name__ == "__main__":
    main()
