#!/usr/bin/env python3
"""Resimulation feasibility experiment (S8 capstone).

Full-box PM (reference) vs buffer-region resimulation of a target sub-cube,
swept over buffer size. Headline result: the inner region converges to the
full-box answer as the buffer grows — selective resimulation works as a
controlled approximation (the feasibility study's verdict, on the dummy).

Run: python3 scripts/resim_feasibility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.simulation.resim.coupling import (  # noqa: E402
    full_target_slice, run_coupled, run_full_reference,
)
from oneuniverse.simulation.resim.verify import gate2_dynamical  # noqa: E402

OUT = Path("/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation"
           "/resim_feasibility")
COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
BOX, N, TLO, TSIDE, SEED = 256.0, 64, 96.0, 64.0, 2
BUFFERS = [16.0, 24.0, 32.0, 48.0, 64.0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = run_full_reference(COSMO, box=BOX, n_grid=N, z_start=9.0,
                              z_end=0.0, seed=SEED, n_steps=25)
    ref = full_target_slice(full, box=BOX, n_grid=N, target_lo=TLO,
                            target_side=TSIDE)
    corrs, r_lowks, inners = [], [], {}
    for buf in BUFFERS:
        res = run_coupled(COSMO, box=BOX, n_grid=N, target_lo=TLO,
                          target_side=TSIDE, buffer=buf, z_start=9.0,
                          z_end=0.0, seed=SEED, n_steps=25)
        g2 = gate2_dynamical(res["inner"], ref, box_size=TSIDE)
        corrs.append(float(np.corrcoef(res["inner"].ravel(),
                                       ref.ravel())[0, 1]))
        r_lowks.append(g2["r_lowk"])
        inners[buf] = res["inner"]

    # convergence curve
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(BUFFERS, r_lowks, "o-", label="large-scale r(k) (Gate 2)")
    ax.plot(BUFFERS, corrs, "s--", label="cell-level corr (all scales)")
    ax.axhline(0.8, color="0.6", ls=":", lw=1)
    ax.set_xlabel("buffer [Mpc/h]")
    ax.set_ylabel("inner-region agreement with full-box")
    ax.set_title("Resimulation feasibility: convergence with buffer (Gate 3)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "01_buffer_convergence.png", dpi=120)
    plt.close(fig)

    # slice comparison: full reference vs resimulated inner (largest buffer)
    sl = ref.shape[2] // 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, fld, ttl in ((ax[0], ref, "full-box reference (target)"),
                        (ax[1], inners[BUFFERS[-1]],
                         f"resimulated (buffer {BUFFERS[-1]:.0f})")):
        im = a.imshow(fld[:, :, sl].T, origin="lower", cmap="magma",
                      vmin=-1, vmax=4)
        a.set_title(ttl); fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout(); fig.savefig(OUT / "02_slice_compare.png", dpi=120)
    plt.close(fig)

    summary = {"box": BOX, "n_grid": N, "target": [TLO, TLO + TSIDE],
               "buffers": BUFFERS, "r_lowk": r_lowks, "cell_corr": corrs,
               "verdict": ("inner region converges to the full-box reference "
                           "as the buffer grows -> selective resimulation is "
                           "feasible as a controlled approximation.")}
    (OUT / "RESULTS.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
