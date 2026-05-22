"""Pin pandas observed=… behaviour on the OneuidQuery hydration path.

The default of `observed=False` is deprecated and will flip in a future
pandas — pinning the explicit kwarg keeps our semantics stable across
upgrades.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.desi_dr1_like import write_fake_desi_dr1_fits  # noqa: E402


def test_iter_partial_no_observed_futurewarning(tmp_path):
    from oneuniverse.combine import WeightedCatalog
    from oneuniverse.data.converter import convert_survey
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_fake_desi_dr1_fits(raw_dir, n_rows=300, seed=0)

    db_root = tmp_path / "db"
    db_root.mkdir()
    convert_survey(
        "desi_qso", raw_path=raw_dir,
        output_dir=db_root / "desi_qso", overwrite=True,
    )
    db = OneuniverseDatabase(db_root)
    db.build_oneuid(
        datasets=["desi_qso"],
        rules=CrossMatchRules(sky_tol_arcsec=0.5),
        name="default",
    )
    idx = db.load_oneuid("default")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Exercise the hydration path that contains the groupby.
        wc = WeightedCatalog.from_oneuid(idx, db)
        wc.fill_defaults(db)
        _ = wc.total_weight("desi_qso")
        # Also exercise OneuidQuery hydration directly via partial_for.
        from oneuniverse.data.oneuid import OneuidQuery
        q = OneuidQuery(db, index=idx)
        oneuids = idx.table["oneuid"].iloc[:10].tolist()
        _ = q.partial_for(oneuids, columns=["ra", "dec"])

    bad = [
        w for w in caught
        if issubclass(w.category, FutureWarning)
        and "observed=" in str(w.message)
    ]
    assert bad == [], [str(b.message) for b in bad]
