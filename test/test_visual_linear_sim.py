"""Phase S3 — diagnostic figure for the dummy linear simulation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.linear.gaussian_field import (  # noqa: E402
    generate_density_field,
)
from oneuniverse.simulation.linear.halos import find_peaks  # noqa: E402
from oneuniverse.simulation.linear.power_spectrum import linear_power  # noqa: E402
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_linear_sim_visual():
    c = CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        t_cmb=2.7255,
    )
    box, n = 256.0, 64
    field = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=42)
    pos, _ = zeldovich_particles(c, box_size=box, n_grid=n, z=0.0, seed=42)
    halos = find_peaks(field, box_size=box, threshold=1.5)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    k = np.logspace(-2.5, 0.5, 200)
    for z, style in ((0.0, "-"), (1.0, "--")):
        ax[0].loglog(k, linear_power(k, c, z=z), style, label=f"z={z}")
    ax[0].set_xlabel("k [h/Mpc]")
    ax[0].set_ylabel("P(k) [(Mpc/h)$^3$]")
    ax[0].set_title("Eisenstein-Hu linear P(k)")
    ax[0].legend()

    proj = field.sum(axis=2)
    im = ax[1].imshow(proj.T, origin="lower", extent=(0, box, 0, box),
                      cmap="magma")
    ax[1].set_xlabel("x [Mpc/h]")
    ax[1].set_ylabel("y [Mpc/h]")
    ax[1].set_title("density field (projected)")
    plt.colorbar(im, ax=ax[1])

    sel = pos[:, 2] < box / n * 4  # a thin slab of particles
    ax[2].scatter(pos[sel, 0], pos[sel, 1], s=1, alpha=0.3, color="0.3")
    if len(halos["x"]):
        ax[2].scatter(halos["x"], halos["y"], s=20, color="tab:red",
                      marker="x", label="halos")
        ax[2].legend()
    ax[2].set_xlim(0, box)
    ax[2].set_ylim(0, box)
    ax[2].set_xlabel("x [Mpc/h]")
    ax[2].set_ylabel("y [Mpc/h]")
    ax[2].set_title("Zel'dovich particles + halos")

    fig.tight_layout()
    out_png = OUT / "linear_sim_overview.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im2:
        assert im2.width >= 800 and im2.height >= 200
