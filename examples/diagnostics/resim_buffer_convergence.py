"""Buffered resimulation convergence: inner-cube agreement with the full box versus buffer size.

Plots ``reference.resim.buffer_convergence`` (cell_corr and r_lowk) for the periodic
``buffered`` and the ``treepm`` coupling on a small synthetic Gaussian initial field.
Always synthetic: --data-root is accepted for CLI uniformity and ignored.

Usage: python examples/diagnostics/resim_buffer_convergence.py [--out DIR]
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from _common import parse_args, save, setup_style

from oneuniverse.reference.linear import gaussian_field
from oneuniverse.reference.resim import (
    buffer_convergence,
    inner_reference,
    run_buffered,
    run_full_reference_from_ic,
)
from oneuniverse.simulation import CosmologySpec, Cube

BOX, N_GRID = 64.0, 32
BUFFERS = [2.0, 4.0, 8.0, 12.0]
RUN = {"z_start": 9.0, "z_end": 0.0, "n_steps": 6}


def main() -> None:
    args = parse_args(__doc__, data_root=False)
    setup_style()
    cosmo = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)
    ic = gaussian_field(cosmo, box=BOX, n_grid=N_GRID, seed=0)
    cube = Cube((16.0, 16.0, 16.0), (32.0, 32.0, 32.0))
    tables = {
        "buffered": buffer_convergence(cosmo, ic, box=BOX, cube=cube, buffers=BUFFERS,
                                       method="buffered", **RUN),
        "treepm": buffer_convergence(cosmo, ic, box=BOX, cube=cube, buffers=BUFFERS,
                                     method="treepm", k_split=0.6, **RUN),
    }
    # inner slice of the full-box reference and of the smallest / largest buffered runs
    full, _ = run_full_reference_from_ic(cosmo, ic, box=BOX, n_grid=N_GRID, **RUN)
    ref = inner_reference(full, box=BOX, cube=cube)
    inner = {b: run_buffered(cosmo, ic, box=BOX, cube=cube, buffer=b, **RUN)["inner"]
             for b in (BUFFERS[0], BUFFERS[-1])}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for k, (col, lab) in enumerate((("cell_corr", "cell-by-cell correlation"),
                                    ("r_lowk", "low-k cross-correlation r"))):
        for method, t in tables.items():
            ax[0, k].plot(t["buffer"].to_numpy(), t[col].to_numpy(), "o-", label=method)
        ax[0, k].set(xlabel="buffer [Mpc/h]", ylabel=col, title=lab)
        ax[0, k].legend()
        ax[0, k].grid(alpha=0.3)
    ax[0, 2].axis("off")
    ax[0, 2].text(0.0, 0.5, f"box {BOX:g} Mpc/h, {N_GRID}^3 grid\ninner cube {cube.lo} - {cube.hi}\n"
                  f"z {RUN['z_start']:g} -> {RUN['z_end']:g}, {RUN['n_steps']} steps\n"
                  "treepm: k_split = 0.6", fontsize=11, va="center")
    mid = ref.shape[2] // 2
    vmax = float(abs(ref[:, :, mid]).max())
    for a, (title, f) in zip(ax[1], (("full-box reference", ref),
                                     (f"buffered, buffer {BUFFERS[0]:g}", inner[BUFFERS[0]]),
                                     (f"buffered, buffer {BUFFERS[-1]:g}", inner[BUFFERS[-1]])),
                             strict=True):
        im = a.imshow(f[:, :, mid].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.set(title=title + ": inner-cube slice")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("Buffer convergence of the resimulation")
    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.92)
    save(fig, args.out, "resim_buffer_convergence")


if __name__ == "__main__":
    main()
