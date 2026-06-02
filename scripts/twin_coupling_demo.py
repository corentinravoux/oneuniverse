#!/usr/bin/env python3
"""End-to-end twin-coupling demo (C-series capstone).

truth (linear field) → mock-observe (sparse linear tracers) → Wiener mean
AND Hoffman-Ribak constrained realization → validation metrics vs truth.
Shows the headline C5 result: the constrained realization restores the
small-scale power the Wiener mean suppresses, while both lock the large
scales the data constrains. A ball survey footprint (C4) is overlaid as the
'where data exists' geometry.

Run: python3 scripts/twin_coupling_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.linear.gaussian_field import (  # noqa: E402
    generate_density_field,
)
from oneuniverse.twin.constrained import constrained_realization  # noqa: E402
from oneuniverse.twin.mock_survey import ball_mask  # noqa: E402
from oneuniverse.twin.validation import recover_metrics  # noqa: E402
from oneuniverse.twin.wiener import wiener_reconstruct  # noqa: E402

OUT = Path("/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation"
           "/twin_coupling")
COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
BOX, N, BIAS, NBAR, SEED = 400.0, 128, 1.5, 1e-3, 42


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = generate_density_field(COSMO, box_size=BOX, n_grid=N, z=0.0,
                                   seed=SEED)
    v_cell = (BOX / N) ** 3
    rng = np.random.default_rng(SEED + 1)
    dg = BIAS * truth + rng.normal(0, 1 / np.sqrt(NBAR * v_cell), (N, N, N))
    wf = wiener_reconstruct(dg, COSMO, box_size=BOX, nbar=NBAR, bias=BIAS)
    cr = constrained_realization(dg, COSMO, box_size=BOX, nbar=NBAR,
                                 bias=BIAS, seed=7)
    m_wf = recover_metrics(wf, truth, box_size=BOX)
    m_cr = recover_metrics(cr, truth, box_size=BOX)

    # --- r(k) + power ratio ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(m_wf.k, m_wf.r, "o-", ms=3, label="Wiener mean")
    ax[0].plot(m_cr.k, m_cr.r, "s-", ms=3, label="constrained realization")
    ax[0].axhline(0.5, color="0.6", ls="--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("k [h/Mpc]"); ax[0].set_ylabel("r(k) vs truth")
    ax[0].set_title("correlation: both lock the constrained large scales")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(m_wf.k, m_wf.power_ratio, "o-", ms=3, label="Wiener mean")
    ax[1].plot(m_cr.k, m_cr.power_ratio, "s-", ms=3, label="constrained real.")
    ax[1].axhline(1.0, color="0.6", ls="--", lw=1)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1.6)
    ax[1].set_xlabel("k [h/Mpc]"); ax[1].set_ylabel("P(k) / P_truth(k)")
    ax[1].set_title("power: CR restores what the Wiener mean suppresses")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "01_wf_vs_cr.png", dpi=120)
    plt.close(fig)

    # --- field slices ---
    mask = ball_mask(N, box_size=BOX, radius=0.42 * BOX)
    sl = N // 2
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, fld, ttl in ((ax[0], truth, "truth δ_m"),
                        (ax[1], wf, "Wiener mean"),
                        (ax[2], cr, "constrained realization")):
        a.imshow(fld[:, :, sl].T, origin="lower", extent=(0, BOX, 0, BOX),
                 cmap="RdBu_r", vmin=-2, vmax=2)
        a.contour(np.linspace(0, BOX, N), np.linspace(0, BOX, N),
                  mask[:, :, sl].T, levels=[0.5], colors="k", linewidths=0.8)
        a.set_title(ttl); a.set_xlabel("x [Mpc/h]"); a.set_ylabel("y [Mpc/h]")
    fig.suptitle("twin coupling: data-constrained field (survey footprint = black)")
    fig.tight_layout(); fig.savefig(OUT / "02_field_slices.png", dpi=120)
    plt.close(fig)

    summary = {"box": BOX, "n_grid": N, "bias": BIAS, "nbar": NBAR,
               "k_half_wf": m_wf.k_half, "k_half_cr": m_cr.k_half,
               "note": "WF mean power-suppressed at high k; CR restores P(k)."}
    (OUT / "RESULTS.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
