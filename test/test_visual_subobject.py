"""Diagnostic skymap of hosts + SNe with a line per sub-object link.
Skipped when matplotlib is unavailable. Output lands under
``test/test_output/``."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.database import OneuniverseDatabase  # noqa: E402
from oneuniverse.data.oneuid_rules import CrossMatchRules  # noqa: E402
from oneuniverse.data.subobject_rules import SubobjectRules  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_subobject_build import (  # noqa: E402
    _synthetic_host_catalog, _synthetic_sn_catalog,
)


OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_subobject_link_skymap(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    _synthetic_host_catalog(root, "hosts", n_host=30, seed=1)
    from oneuniverse.data.dataset_view import DatasetView
    host_df = DatasetView.from_path(root / "hosts").read(
        columns=["ra", "dec", "z"]
    )
    _synthetic_sn_catalog(root, "sne", host_df, seed=1)

    db = OneuniverseDatabase(root)
    db.build_oneuid(
        datasets=["hosts", "sne"],
        rules=CrossMatchRules(sky_tol_arcsec=0.05),
        name="default",
    )
    db.build_subobject_links(
        rules=SubobjectRules(
            parent_survey_type="spectroscopic",
            child_survey_type="transient",
            sky_tol_arcsec=2.0, dz_tol=1e-2,
        ),
        parent_datasets=["hosts"],
        child_datasets=["sne"],
        name="sne_in_hosts",
    )
    links = db.load_subobject_links("sne_in_hosts")

    oneuid_tbl = db.load_oneuid("default").table
    host_idx = oneuid_tbl[oneuid_tbl["dataset"] == "hosts"].set_index("oneuid")
    sn_idx = oneuid_tbl[oneuid_tbl["dataset"] == "sne"].set_index("oneuid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        host_idx["ra"], host_idx["dec"],
        marker="o", s=40, facecolors="none", edgecolors="C0",
        label=f"hosts (n={len(host_idx)})",
    )
    ax.scatter(
        sn_idx["ra"], sn_idx["dec"],
        marker="*", s=30, c="C3",
        label=f"SNe (n={len(sn_idx)})",
    )
    for _, row in links.table.iterrows():
        p = host_idx.loc[int(row["parent_oneuid"])]
        c = sn_idx.loc[int(row["child_oneuid"])]
        ax.plot([p.ra, c.ra], [p.dec, c.dec], "-", lw=0.5,
                alpha=max(0.15, float(row["confidence"])),
                color="k")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"Sub-object links: hosts -> SNe  "
        f"(n_links={len(links)}, mean sep="
        f"{links.table['sky_sep_arcsec'].mean():.2f}\")"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    outfile = OUTPUT_DIR / "subobject_host_sn_skymap.png"
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    assert outfile.exists() and outfile.stat().st_size > 10_000
