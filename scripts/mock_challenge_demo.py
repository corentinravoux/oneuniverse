#!/usr/bin/env python3
"""Minimal data↔sim coupling demo (the mock challenge).

truth (linear field) -> mock-observe (biased Poisson tracers) -> constrain
(Wiener filter) -> verify (cross-correlation r(k) vs truth). The dummy gives
ground truth, so recovery is measurable: r(k) -> 1 at low k, falling where
shot noise dominates. The scale where r=0.5 is the feasibility number per
survey number density.

Run: python3 scripts/mock_challenge_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from oneuniverse.simulation.cosmology import CosmologySpec  # noqa: E402
from oneuniverse.twin.mock_challenge import run_mock_challenge  # noqa: E402

OUT = Path("/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation"
           "/mock_challenge")
COSMO = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
BOX, NGRID, BIAS, SEED = 512.0, 128, 1.5, 42
NBARS = [1e-3, 5e-3, 5e-2]   # (Mpc/h)^-3


def _k_half(k, r):
    """Smallest k where r(k) drops below 0.5 (the reconstruction scale)."""
    good = np.isfinite(r)
    k, r = k[good], r[good]
    below = np.where(r < 0.5)[0]
    return float(k[below[0]]) if len(below) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {nbar: run_mock_challenge(COSMO, box_size=BOX, n_grid=NGRID,
                                     nbar=nbar, bias=BIAS, seed=SEED)
            for nbar in NBARS}

    # --- headline: r(k) per survey density ---
    fig, ax = plt.subplots(figsize=(6.5, 5))
    summary = {}
    for nbar, res in runs.items():
        ax.plot(res["k"], res["r"], "o-", ms=3, label=f"n̄={nbar:.0e}")
        summary[f"{nbar:.0e}"] = {"k_half_Mpc_h": _k_half(res["k"], res["r"])}
    ax.axhline(0.5, color="0.6", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]"); ax.set_ylabel("r(k)  (reconstruction × truth)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mock challenge: large-scale field recovery vs survey density")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "01_rk_recovery.png", dpi=120)
    plt.close(fig)

    # --- visual recovery: truth / observed / reconstruction slices ---
    res = runs[5e-2]
    sl = NGRID // 2
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, fld, ttl in (
        (ax[0], res["truth"], "truth δ_m"),
        (ax[1], res["delta_g"], "observed δ_g (tracers)"),
        (ax[2], res["rec"], "Wiener reconstruction"),
    ):
        im = a.imshow(fld[:, :, sl].T, origin="lower",
                      extent=(0, BOX, 0, BOX), cmap="RdBu_r",
                      vmin=-2, vmax=2)
        a.set_title(ttl); a.set_xlabel("x [Mpc/h]"); a.set_ylabel("y [Mpc/h]")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"n̄={5e-2:.0e} (Mpc/h)⁻³, b={BIAS}")
    fig.tight_layout(); fig.savefig(OUT / "02_field_slices.png", dpi=120)
    plt.close(fig)

    # --- scatter: large-scale recovery cell-by-cell (smoothed) ---
    from numpy.fft import rfftn, irfftn
    def smooth(f, n):
        kx = np.fft.fftfreq(n, d=BOX / n) * 2 * np.pi
        kz = np.fft.rfftfreq(n, d=BOX / n) * 2 * np.pi
        kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
        k2 = kxg**2 + kyg**2 + kzg**2
        R = 16.0
        return irfftn(rfftn(f) * np.exp(-0.5 * k2 * R**2), s=(n, n, n))
    ts, rs = smooth(res["truth"], NGRID).ravel(), smooth(res["rec"], NGRID).ravel()
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.hexbin(ts, rs, gridsize=60, cmap="viridis", bins="log")
    lim = max(abs(ts).max(), abs(rs).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlabel("truth δ_m (16 Mpc/h smoothed)")
    ax.set_ylabel("reconstruction (16 Mpc/h smoothed)")
    ax.set_title("Large-scale recovery, cell by cell")
    fig.tight_layout(); fig.savefig(OUT / "03_recovery_scatter.png", dpi=120)
    plt.close(fig)

    summary_full = {
        "box_Mpc_h": BOX, "n_grid": NGRID, "bias": BIAS, "seed": SEED,
        "nbars": NBARS, "k_half_per_nbar": summary,
        "note": ("k_half = scale where r(k)=0.5 = the reconstruction "
                 "resolution; larger k_half = finer scales recovered."),
    }
    (OUT / "RESULTS.json").write_text(json.dumps(summary_full, indent=2))
    print(json.dumps(summary_full, indent=2))


if __name__ == "__main__":
    main()
