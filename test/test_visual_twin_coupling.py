"""C-series capstone — Wiener-mean vs constrained-realization figure."""
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
from oneuniverse.twin.constrained import constrained_realization  # noqa: E402
from oneuniverse.twin.validation import recover_metrics  # noqa: E402
from oneuniverse.twin.wiener import wiener_reconstruct  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_wf_vs_cr_visual():
    c = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
    box, n, bias, nbar = 256.0, 64, 1.5, 1e-3
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=2)
    rng = np.random.default_rng(3)
    dg = bias * truth + rng.normal(0, 1 / np.sqrt(nbar * (box / n) ** 3),
                                   (n, n, n))
    wf = wiener_reconstruct(dg, c, box_size=box, nbar=nbar, bias=bias)
    cr = constrained_realization(dg, c, box_size=box, nbar=nbar, bias=bias,
                                 seed=7)
    mw, mc = (recover_metrics(wf, truth, box_size=box),
              recover_metrics(cr, truth, box_size=box))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mw.k, mw.power_ratio, "o-", ms=3, label="Wiener mean")
    ax.plot(mc.k, mc.power_ratio, "s-", ms=3, label="constrained real.")
    ax.axhline(1.0, color="0.6", ls="--"); ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("P/P_truth"); ax.legend()
    out = OUT / "twin_wf_vs_cr.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    assert out.exists() and out.stat().st_size > 15_000
    # the CR restores more high-k power than the suppressed Wiener mean
    band = mw.k > 0.3
    assert np.nanmedian(mc.power_ratio[band]) > np.nanmedian(mw.power_ratio[band])
