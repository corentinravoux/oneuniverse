"""Buffer-region resimulation (sCOLA-lite).

Resimulate a target sub-cube by running the PM on the particles whose
*Lagrangian* positions fall in an enlarged **buffer** cube (target padded by
``buffer`` on each side), in an isolated periodic sub-box. The buffer carries
the large-scale modes + tidal field around the target so the *inner* region
(away from the buffer edge) approaches the full-box result; the boundary
error is pushed out into the buffer. Convergence with buffer size is the
quantity Gate-3 (S8.5) measures.

Honest scope (feasibility study): this captures the modes resolved within
the buffer; super-buffer tides are dropped (the irreducible truncation).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.pm.deposit import deposit_cic
from oneuniverse.simulation.pm.run import (
    run_pm,
    zeldovich_pm_ic,
    zeldovich_pm_ic_from_field,
)


def _cubic_cells(centre_lo: float, side: float, cell: float) -> Tuple[int, int]:
    i0 = int(round(centre_lo / cell))
    i1 = int(round((centre_lo + side) / cell))
    return i0, i1


def run_full_reference(cosmo: CosmologySpec, *, box: float, n_grid: int,
                       z_start: float, z_end: float, seed: int,
                       n_steps: int = 20) -> np.ndarray:
    """Full-box PM evolved overdensity field (the reference truth)."""
    pos, p0 = zeldovich_pm_ic(cosmo, box=box, n_grid=n_grid,
                              z_start=z_start, seed=seed)
    x, _ = run_pm(pos, p0, box=box, n_grid=n_grid, cosmo=cosmo,
                  a_start=1.0 / (1.0 + z_start), a_end=1.0 / (1.0 + z_end),
                  n_steps=n_steps)
    rho = deposit_cic(x, n_grid, box)
    return rho / rho.mean() - 1.0


def run_coupled(cosmo: CosmologySpec, *, box: float, n_grid: int,
                target_lo: float, target_side: float, buffer: float,
                z_start: float, z_end: float, seed: Optional[int] = None,
                ic_field: Optional[np.ndarray] = None,
                n_steps: int = 20) -> Dict:
    """Resimulate a cubic target with a buffer; return inner-region field.

    The cubic target is ``[target_lo, target_lo+target_side]`` on each axis;
    the buffer cube pads it by ``buffer`` per side. The IC comes either from a
    ``seed`` (fresh Zel'dovich) or a provided ``ic_field`` (a z=0 density
    field, e.g. a constrained realization — the data-driven path). Returns the
    inner-target overdensity sub-grid + bookkeeping.
    """
    cell = box / n_grid
    bsize = target_side + 2.0 * buffer
    blo = target_lo - buffer
    bi0 = int(round(blo / cell))
    bi1 = int(round((blo + bsize) / cell))
    n_buf = bi1 - bi0
    origin = bi0 * cell
    box_buf = n_buf * cell

    # full IC (from a provided field, or fresh from a seed) + Lagrangian grid
    if ic_field is not None:
        pos, p0 = zeldovich_pm_ic_from_field(cosmo, ic_field, box=box,
                                             n_grid=n_grid, z_start=z_start)
    else:
        pos, p0 = zeldovich_pm_ic(cosmo, box=box, n_grid=n_grid,
                                  z_start=z_start, seed=seed if seed else 0)
    g = (np.arange(n_grid) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    # select particles whose Lagrangian position lies in the buffer cube
    lo, hi = origin, origin + box_buf
    m = np.all((q >= lo) & (q < hi), axis=1)
    sub_pos = (pos[m] - origin) % box_buf
    sub_p = p0[m]

    x, _ = run_pm(sub_pos, sub_p, box=box_buf, n_grid=n_buf, cosmo=cosmo,
                  a_start=1.0 / (1.0 + z_start), a_end=1.0 / (1.0 + z_end),
                  n_steps=n_steps)
    rho = deposit_cic(x, n_buf, box_buf)
    delta_buf = rho / rho.mean() - 1.0

    # trim the buffer -> inner target cells
    pad = int(round(buffer / cell))
    ti0 = int(round(target_side / cell))
    inner = delta_buf[pad:pad + ti0, pad:pad + ti0, pad:pad + ti0]
    return {"inner": inner, "target_cells": ti0, "n_buf": n_buf,
            "n_particles": int(m.sum())}


def full_target_slice(delta_full: np.ndarray, *, box: float, n_grid: int,
                      target_lo: float, target_side: float) -> np.ndarray:
    """The full-box reference field restricted to the target cube."""
    cell = box / n_grid
    i0 = int(round(target_lo / cell))
    i1 = i0 + int(round(target_side / cell))
    return delta_full[i0:i1, i0:i1, i0:i1]
