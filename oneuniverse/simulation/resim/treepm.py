"""TreePM-split coupled resimulation — the working sCOLA-class solution.

The mini-sim paradox: a small tile cannot feel the large-scale gravity beyond
its buffer. Fix by splitting the force **by Fourier scale** (TreePM-style):

- **long-range** force from the cheap full-box *linear* field (low-pass) —
  carries the external tide, evolves analytically as D(a);
- **short-range** force from the local tile PM (high-pass) — the non-linear
  detail.

The two ranges are complementary in k, so there is no double-counting (the
failure mode of the COLA-frame attempts). Result: the same inner-region
accuracy as the uncoupled buffered run at a **much smaller buffer** (≈4× on
the dummy). The full-box linear field is the "large grid"; the tile PM is the
mini-sim coupled to it.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.growth import growth_factor
from oneuniverse.simulation.pm.deposit import deposit_cic, interpolate_cic
from oneuniverse.simulation.pm.poisson import pm_force
from oneuniverse.simulation.pm.run import _factor, zeldovich_pm_ic_from_field


def _kmag(ng: int, box: float) -> np.ndarray:
    kx = np.fft.fftfreq(ng, d=box / ng) * 2.0 * np.pi
    kz = np.fft.rfftfreq(ng, d=box / ng) * 2.0 * np.pi
    a, b, d = np.meshgrid(kx, kx, kz, indexing="ij")
    return np.sqrt(a * a + b * b + d * d)


def _filter_force(force, ng, W):
    return [np.fft.irfftn(np.fft.rfftn(g) * W, s=(ng, ng, ng)) for g in force]


def run_coupled_treepm(cosmo: CosmologySpec, ic_field: np.ndarray, *,
                       box: float, n_grid: int, target_lo: float,
                       target_side: float, buffer: float, z_start: float,
                       z_end: float, n_steps: int = 20,
                       ksplit_factor: float = 2.0) -> Dict:
    """Resimulate a cubic target with a TreePM-split long/short-range force.

    ``ic_field`` is the full-box z=0 linear field (the "large grid"); its
    low-pass force supplies the external tide to the tile, whose own PM
    supplies the high-pass detail. ``ksplit_factor`` sets the split scale in
    units of the buffer-box fundamental (≈2 works well).
    """
    om = cosmo.omega_m
    cell = box / n_grid
    bsize = target_side + 2.0 * buffer
    blo = target_lo - buffer
    bi0 = int(round(blo / cell)); bi1 = int(round((blo + bsize) / cell))
    n_buf = bi1 - bi0; box_buf = n_buf * cell; origin = bi0 * cell
    ksplit = ksplit_factor * 2.0 * np.pi / box_buf

    # long-range: full-box linear force, low-pass (the external tide)
    g_full = pm_force(np.asarray(ic_field, float), box)
    w_low = np.exp(-0.5 * (_kmag(n_grid, box) / ksplit) ** 2)
    g_low = _filter_force(g_full, n_grid, w_low)
    # short-range: tile force, high-pass
    w_high = 1.0 - np.exp(-0.5 * (_kmag(n_buf, box_buf) / ksplit) ** 2)

    # buffer particles inherit the full-box displacement
    pos, p0 = zeldovich_pm_ic_from_field(cosmo, ic_field, box=box,
                                         n_grid=n_grid, z_start=z_start)
    g = (np.arange(n_grid) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    m = np.all((q >= origin) & (q < origin + box_buf), axis=1)
    x = (pos[m] - origin) % box_buf
    ps = p0[m].copy()

    def D(a):
        return growth_factor(1.0 / a - 1.0, cosmo)

    def acc(xs, a):
        rho = deposit_cic(xs, n_buf, box_buf)
        gfh = _filter_force(pm_force(rho / rho.mean() - 1.0, box_buf),
                            n_buf, w_high)
        fs = np.stack([interpolate_cic(gfh[i], xs, box_buf)
                       for i in range(3)], axis=1)
        fl = D(a) * np.stack([interpolate_cic(g_low[i], (xs + origin) % box, box)
                              for i in range(3)], axis=1)
        return 1.5 * om * (fs + fl)

    a_grid = np.linspace(1.0 / (1.0 + z_start), 1.0 / (1.0 + z_end), n_steps + 1)
    for i in range(n_steps):
        a0, a1 = a_grid[i], a_grid[i + 1]; ah = 0.5 * (a0 + a1)
        ps += acc(x, a0) * _factor(a0, ah, om, 2)
        x = (x + ps * _factor(a0, a1, om, 3)) % box_buf
        ps += acc(x, a1) * _factor(ah, a1, om, 2)

    rho = deposit_cic(x, n_buf, box_buf)
    delta = rho / rho.mean() - 1.0
    pad = int(round(buffer / cell)); ti = int(round(target_side / cell))
    inner = delta[pad:pad + ti, pad:pad + ti, pad:pad + ti]
    return {"inner": inner, "n_buf": n_buf}
