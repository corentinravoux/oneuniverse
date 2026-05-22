"""Diagnostic mollview: systematic map × point catalogue overlay."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import healpy as hp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.healpix_maps import make_systematic_map  # noqa: E402

from oneuniverse.combine.weights.hpmap import HealpixMapWeight  # noqa: E402


OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase11_hpmap_overlay():
    nside = 32
    m = make_systematic_map(nside, seed=7)
    n = 2000
    rng = np.random.default_rng(0)
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2 * np.pi, size=n)
    dec = np.degrees(np.arcsin(u))
    ra = np.degrees(phi)
    df = pd.DataFrame({"ra": ra, "dec": dec})

    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    vals = w(df)

    fig = plt.figure(figsize=(10, 5))
    hp.mollview(
        m, nest=True, title="Synthetic systematic map", fig=fig,
        sub=121, cmap="viridis",
    )
    ax = fig.add_subplot(122)
    sc = ax.scatter(df["ra"], df["dec"], c=vals, s=2, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="HealpixMapWeight value")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title("Points coloured by pixel weight")
    # mollview + add_subplot mix incompatible with tight_layout — use
    # explicit subplots_adjust instead.
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.10, wspace=0.25,
    )
    out_png = OUT / "phase11_hpmap_overlay.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 10_000
