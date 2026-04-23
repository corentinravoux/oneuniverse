#!/usr/bin/env python
"""End-to-end onboarding driver for DESI DR1 QSO.

Usage
-----
    python scripts/onboard_desi_dr1_qso.py \\
        --raw-root /path/to/raw \\
        --db-root  /path/to/oneuniverse/db

When --raw-root does not contain QSO_full.dat.fits and
--fake-if-missing is set, a synthetic DR1-shaped FITS is written first
(test/fixtures/desi_dr1_like.py).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("onboard_desi_dr1_qso")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--db-root", type=Path, required=True)
    p.add_argument("--fake-if-missing", action="store_true")
    p.add_argument("--fake-n-rows", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from oneuniverse.combine import WeightedCatalog
    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import (
        DESIQSOLoader,
    )

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.db_root.mkdir(parents=True, exist_ok=True)

    fits_path = args.raw_root / DESIQSOLoader.config.data_filename
    if not fits_path.exists():
        if not args.fake_if_missing:
            LOG.error("FITS not found: %s", fits_path)
            return 2
        LOG.warning("Generating synthetic fixture at %s", fits_path)
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "test"),
        )
        from fixtures.desi_dr1_like import write_fake_desi_dr1_fits
        write_fake_desi_dr1_fits(
            args.raw_root, n_rows=args.fake_n_rows, seed=args.seed,
        )

    ou = args.db_root / "desi_qso"
    LOG.info("Converting -> %s", ou)
    convert_survey(
        "desi_qso",
        raw_path=args.raw_root,
        output_dir=ou,
        overwrite=True,
    )

    db = OneuniverseDatabase(args.db_root)
    LOG.info("Building ONEUID index 'default' ...")
    db.build_oneuid(
        datasets=["desi_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")
    LOG.info(
        "ONEUID: %d rows / %d unique / %d multi",
        len(idx.table), idx.n_unique, idx.n_multi,
    )

    wc = WeightedCatalog.from_oneuid(idx, db)
    wc.fill_defaults(db, skip_unknown=True)
    for survey in wc.catalogs:
        w = wc.total_weight(survey)
        LOG.info(
            "WeightedCatalog[%s]: n=%d mean(w)=%.3g",
            survey, len(w), float(w.mean()),
        )

    LOG.info("DONE — database root: %s", args.db_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
