"""Phase 22 visual diagnostic — synthetic IFU cube + GW skymap."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase22_visual(tmp_path):
    # Synthetic IFU cube
    nra, ndec, nchan = 24, 24, 30
    rng = np.random.default_rng(0)
    grid_ra = np.arange(nra)[:, None, None]
    grid_dec = np.arange(ndec)[None, :, None]
    grid_ch = np.arange(nchan)[None, None, :]
    blob = np.exp(
        -0.5 * ((grid_ra - nra / 2) ** 2 + (grid_dec - ndec / 2) ** 2) / 30.0
    )
    line = np.exp(-0.5 * ((grid_ch - nchan / 2) ** 2) / 4.0)
    cube = (blob * line + 0.1 * rng.normal(size=(nra, ndec, nchan))).astype("f4")

    # Synthetic GW skymap
    nside = 64
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    centre = hp.ang2vec(20.0, 0.0, lonlat=True)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix), nest=True))
    cos_sep = vecs.T @ centre
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    prob = np.exp(-0.5 * (sep / np.radians(5.0)) ** 2)
    prob /= prob.sum()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].imshow(cube.sum(axis=2).T, origin="lower", cmap="viridis")
    ax[0].set_xlabel("RA pixel")
    ax[0].set_ylabel("Dec pixel")
    ax[0].set_title("IFU cube — collapsed flux")

    ax[1].plot(cube.sum(axis=(0, 1)) / cube.sum(), lw=1.0)
    ax[1].set_xlabel("channel")
    ax[1].set_ylabel("relative flux")
    ax[1].set_title("Spectral axis (line at mid-channel)")

    sc = ax[2].scatter(lon, lat, c=prob, s=2, cmap="magma")
    ax[2].set_xlabel("lon [deg]")
    ax[2].set_ylabel("lat [deg]")
    ax[2].set_title("GW probability skymap")
    plt.colorbar(sc, ax=ax[2], label="P(pixel)")

    fig.tight_layout()
    out_png = OUT / "phase22_cube_and_gwskymap.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
