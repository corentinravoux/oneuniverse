"""S8 capstone — resimulation buffer-convergence figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.resim.coupling import (  # noqa: E402
    full_target_slice, run_coupled, run_full_reference,
)

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_resim_convergence_visual():
    c = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
    box, n, tlo, tside = 200.0, 48, 75.0, 50.0
    full = run_full_reference(c, box=box, n_grid=n, z_start=9.0, z_end=0.0,
                              seed=2, n_steps=12)
    ref = full_target_slice(full, box=box, n_grid=n, target_lo=tlo,
                            target_side=tside)
    buffers = [12.5, 37.5]
    corr = []
    for buf in buffers:
        res = run_coupled(c, box=box, n_grid=n, target_lo=tlo,
                          target_side=tside, buffer=buf, z_start=9.0,
                          z_end=0.0, seed=2, n_steps=12)
        corr.append(np.corrcoef(res["inner"].ravel(), ref.ravel())[0, 1])
    assert corr[1] > corr[0]            # convergence
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(buffers, corr, "o-")
    ax.set_xlabel("buffer [Mpc/h]"); ax.set_ylabel("inner vs full-box corr")
    ax.set_title("resimulation buffer convergence")
    out = OUT / "resim_convergence.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    assert out.exists() and out.stat().st_size > 10_000
