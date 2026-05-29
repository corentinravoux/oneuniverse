"""Phase 20 visual diagnostic — map-based host association."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.subobject_map import (  # noqa: E402
    build_subobject_links_to_map,
)

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def _gaussian_map(nside: int, ra: float, dec: float, sigma_deg: float):
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    target = hp.ang2vec(theta, phi)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix), nest=True))
    cos_sep = vecs.T @ target
    sep_rad = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    m = np.exp(-0.5 * (sep_rad / np.radians(sigma_deg)) ** 2)
    m /= m.sum()
    return m.astype("f4")


def test_phase20_visual(tmp_path):
    nside = 64
    rng = np.random.default_rng(0)
    n_parents = 1000
    parents = pd.DataFrame({
        "oneuid": np.arange(n_parents, dtype="i8"),
        "ra":  rng.uniform(0.0, 60.0, n_parents).astype("f8"),
        "dec": rng.uniform(-15.0, 15.0, n_parents).astype("f8"),
    })

    events = pd.DataFrame({
        "oneuid": np.array([1000, 1001], dtype="i8"),
        "skymap": [
            _gaussian_map(nside, 20.0, 0.0, sigma_deg=3.0),
            _gaussian_map(nside, 40.0, 5.0, sigma_deg=4.0),
        ],
    })

    links = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=1e-5,
    )
    df = links.table

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].scatter(parents["ra"], parents["dec"], s=2, color="0.7",
                  label="parents (all)")
    matched_ids = set(df["parent_oneuid"].astype("int64").tolist())
    matched_mask = parents["oneuid"].isin(matched_ids)
    ax[0].scatter(parents.loc[matched_mask, "ra"],
                  parents.loc[matched_mask, "dec"],
                  s=6, color="tab:red", label="matched")
    ax[0].set_xlabel("RA [deg]")
    ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title("Parents vs map matches")
    ax[0].legend()

    ax[1].hist(df["confidence"], bins=40, color="tab:blue", alpha=0.8)
    ax[1].set_xlabel("confidence (pixel probability)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Match confidence distribution")

    parent_lookup = parents.set_index("oneuid")["ra"]
    for evt_id in events["oneuid"]:
        sel = df["child_oneuid"] == int(evt_id)
        parent_ids = df.loc[sel, "parent_oneuid"].astype("int64").to_numpy()
        if parent_ids.size == 0:
            continue
        ras = parent_lookup.loc[parent_ids].to_numpy()
        ax[2].scatter(ras, df.loc[sel, "confidence"].to_numpy(),
                      s=4, alpha=0.5, label=f"event {int(evt_id)}")
    ax[2].set_xlabel("parent RA [deg]")
    ax[2].set_ylabel("confidence")
    ax[2].legend()
    ax[2].set_title("Per-event RA × confidence")

    fig.tight_layout()
    out_png = OUT / "phase20_map_based_subobject.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
