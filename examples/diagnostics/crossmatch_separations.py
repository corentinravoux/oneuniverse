"""Cross-match of two surveys (match_pair + components, nothing persisted): separation and delta-z histograms of matched pairs.

Read-only: nothing is written under --data-root (no ONEUID index is built).

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
from oneuniverse.data.crossmatch import components, match_pair


def main() -> None:
    args = parse_args(__doc__)
    setup_style()
    rules = CrossMatchRules(sky_tol_arcsec=1.0, dz_tol=5e-3)
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(args.data_root) if args.data_root else synthetic_database(Path(tmp))
        names = pick(db, args.dataset, 2)
        cols = ["ra", "dec", "z", "z_type", "hpix32"]
        ta, tb = (db[n].table(columns=cols) for n in names)
        pairs = match_pair(ta, tb, rules)
        sep, dz = pairs.sep_arcsec, pairs.dz
        na, nb = ta.num_rows, tb.num_rows
        label = components(na + nb, pairs.left, pairs.right + na, sep,
                           survey_of=np.repeat(np.array([0, 1], np.int16), [na, nb]),
                           allow_same_survey_merge=rules.allow_same_survey_merge)
        members = np.bincount(label)
        counts = np.bincount(members)
        n_objects, n_multi = len(members), int((members >= 2).sum())

        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        ax[0, 0].hist(sep, bins=60, color="C0")
        ax[0, 0].axvline(rules.sky_tol_arcsec, c="k", ls="--", label="sky_tol")
        ax[0, 0].set(xlabel="separation [arcsec]", ylabel="pairs", title="matched-pair separation")
        ax[0, 0].legend()
        ax[0, 1].hist(dz, bins=60, color="C1")
        ax[0, 1].axvline(rules.dz_tol, c="k", ls="--", label="dz_tol")
        ax[0, 1].legend()
        ax[0, 1].set(xlabel=r"$|\Delta z|$", ylabel="pairs", title=r"matched-pair $|\Delta z|$")
        ax[1, 0].scatter(sep, dz, s=2, alpha=0.4, rasterized=True)
        ax[1, 0].set(xlabel="separation [arcsec]", ylabel=r"$|\Delta z|$", title="joint")
        ax[1, 1].bar(np.arange(1, len(counts)), counts[1:], color="C2")
        ax[1, 1].set_xticks(np.arange(1, len(counts)))
        ax[1, 1].set(xlabel="members per ONEUID", ylabel="objects", yscale="log",
                     title=f"{n_objects:,} objects, {n_multi:,} multi-survey")
        fig.suptitle(f"Cross-match {names[0]} x {names[1]} ({len(sep):,} pairs)")
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92)
        save(fig, args.out, "crossmatch_separations")


if __name__ == "__main__":
    main()
