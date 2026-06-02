"""Phase C1 — diagnostic figure for the mock challenge."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.twin.mock_challenge import run_mock_challenge  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_mock_challenge_visual():
    cosmo = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                          sigma8=0.81, t_cmb=2.7255)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for nbar in (1e-3, 5e-3, 5e-2):
        res = run_mock_challenge(cosmo, box_size=256.0, n_grid=64,
                                 nbar=nbar, bias=1.5, seed=7)
        ax.plot(res["k"], res["r"], "o-", ms=3, label=f"n̄={nbar:.0e}")
    ax.axhline(0.5, color="0.6", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("r(k)")
    ax.set_title("mock challenge — field recovery vs survey density")
    ax.legend(); ax.grid(alpha=0.3)
    out = OUT / "mock_challenge_rk.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    assert out.exists() and out.stat().st_size > 20_000
