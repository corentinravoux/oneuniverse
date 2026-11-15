"""Mock challenge: truth, lognormal tracer overdensity and Wiener reconstruction, plus r(k).

Runs ``reference.mock.run_mock_challenge`` on a small box. Always synthetic:
--data-root is accepted for CLI uniformity and ignored.

Usage: python examples/diagnostics/mock_challenge.py [--out DIR]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save, setup_style

from oneuniverse.reference.mock import run_mock_challenge
from oneuniverse.simulation import CosmologySpec

BOX, N_GRID = 256.0, 48


def main() -> None:
    args = parse_args(__doc__, data_root=False)
    setup_style()
    cosmo = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)
    res = run_mock_challenge(cosmo, box=BOX, n_grid=N_GRID, nbar=2e-3, bias=1.5, seed=3)
    truth, dg, wf = res["truth"], res["delta_g"], res["wiener"]
    mid = N_GRID // 2
    m = res["metrics"]
    k, r = m.k, m.r

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    vmax = float(np.percentile(np.abs(truth[:, :, mid]), 99))
    for a, (title, f) in zip(ax.flat[:3], (("truth", truth), ("tracers delta_g", dg),
                                           ("Wiener reconstruction", wf)), strict=True):
        im = a.imshow(f[:, :, mid].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      extent=(0, BOX, 0, BOX))
        a.set(title=title, xlabel="x [Mpc/h]", ylabel="y [Mpc/h]")
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[1, 1].plot(k, r, "o-", label="r(k) Wiener vs truth")
    ax[1, 1].plot(k, m.transfer, "s--", label="transfer T(k)")
    ax[1, 1].axhline(1.0, c="0.6", lw=0.8)
    ax[1, 1].set(xscale="log", xlabel="k [h/Mpc]", ylabel="r(k), T(k)", title="Wiener recovery of the truth")
    ax[1, 1].axvline(m.k_half, c="C3", ls=":", label=f"k_half = {m.k_half:.3g}")
    ax[1, 1].legend()
    fig.suptitle(f"Mock challenge: box {BOX:g} Mpc/h, {N_GRID}^3, bias 1.5")
    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.93)
    save(fig, args.out, "mock_challenge")


if __name__ == "__main__":
    main()
