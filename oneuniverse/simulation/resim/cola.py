"""COLA-frame particle mesh (Tassev, Zaldarriaga & Eisenstein 2013, 1LPT).

The trajectory is split x(a) = q + D(a)·Ψ + s(a): the large-scale Zel'dovich
motion D(a)·Ψ is carried analytically (it includes the external tide when Ψ is
the full-box displacement), and the PM solves only the residual s. The kick
uses (F_full − F_LPT) so the residual feels only the *non-LPT* (small-scale)
force — no double-counting. Large scales are exact by construction, so few PM
steps suffice.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.growth import growth_factor
from oneuniverse.simulation.pm.deposit import deposit_cic, interpolate_cic
from oneuniverse.simulation.pm.poisson import pm_force
from oneuniverse.simulation.pm.run import _E, _factor, _zeldovich_displacement


def cola_run_pm(cosmo: CosmologySpec, delta_z0: np.ndarray, *, box: float,
                n_grid: int, a_start: float, a_end: float,
                n_steps: int = 10) -> np.ndarray:
    """Evolve a z=0 density field in the COLA frame; return final positions."""
    om = cosmo.omega_m
    n = n_grid
    psi = _zeldovich_displacement(np.asarray(delta_z0, float), box, n)  # (n³,3)
    g_unit = pm_force(np.asarray(delta_z0, float), box)        # 3 grids, no prefactor
    cell = box / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    def D(a):
        return growth_factor(1.0 / a - 1.0, cosmo)

    x = (q + D(a_start) * psi) % box
    ps = np.zeros_like(x)                                       # residual momentum

    def res_acc(xp, a):
        rho = deposit_cic(xp, n, box)
        gf = pm_force(rho / rho.mean() - 1.0, box)
        Da = D(a)
        acc = np.empty_like(xp)
        for i in range(3):
            acc[:, i] = interpolate_cic(gf[i] - Da * g_unit[i], xp, box)
        return 1.5 * om * acc

    a_grid = np.linspace(a_start, a_end, n_steps + 1)
    for i in range(n_steps):
        a0, a1 = a_grid[i], a_grid[i + 1]
        ah = 0.5 * (a0 + a1)
        ps += res_acc(x, a0) * _factor(a0, ah, om, 2)
        x = (x + ps * _factor(a0, a1, om, 3) + (D(a1) - D(a0)) * psi) % box
        ps += res_acc(x, a1) * _factor(ah, a1, om, 2)
    return x
