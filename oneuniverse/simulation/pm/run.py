"""KDK leapfrog particle-mesh driver.

Symplectic kick-drift-kick in scale factor a (H0 = 1 units). Force from the
mesh Poisson solve scaled by (3/2)Ω_m (comoving Poisson source). Kick/drift
factors are the standard cosmological integrals ∫ da/(a²E), ∫ da/(a³E) with
E(a) = sqrt(Ω_m a^-3 + 1 − Ω_m). Validated against linear growth +
Zel'dovich: large-scale phases track linear theory, small scales develop the
excess (non-linear) power gravity produces.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.pm.deposit import deposit_cic, interpolate_cic
from oneuniverse.simulation.pm.poisson import pm_force


def _E(a, om):
    return np.sqrt(om * a ** -3 + (1.0 - om))


def _factor(a1, a2, om, power):
    a = np.linspace(a1, a2, 64)
    return float(np.trapz(1.0 / (a ** power * _E(a, om)), a))


def _accel(pos, box, n_grid, om):
    rho = deposit_cic(pos, n_grid, box)
    delta = rho / rho.mean() - 1.0
    gx, gy, gz = pm_force(delta, box)
    fac = 1.5 * om
    ax = interpolate_cic(gx, pos, box) * fac
    ay = interpolate_cic(gy, pos, box) * fac
    az = interpolate_cic(gz, pos, box) * fac
    return np.stack([ax, ay, az], axis=1)


def run_pm(pos: np.ndarray, vel: np.ndarray, *, box: float, n_grid: int,
           cosmo: CosmologySpec, a_start: float, a_end: float,
           n_steps: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve (pos, vel) from a_start to a_end. Returns (pos, vel).

    ``vel`` is the canonical momentum p = a² dx/dt (H0=1). Positions wrap to
    [0, box).
    """
    om = cosmo.omega_m
    x = (np.asarray(pos, dtype=np.float64) % box).copy()
    p = np.asarray(vel, dtype=np.float64).copy()
    a_grid = np.linspace(a_start, a_end, n_steps + 1)

    acc = _accel(x, box, n_grid, om)
    for i in range(n_steps):
        a0, a1 = a_grid[i], a_grid[i + 1]
        ah = 0.5 * (a0 + a1)
        # half kick -> full drift -> half kick (KDK)
        p += acc * _factor(a0, ah, om, 2)
        x = (x + p * _factor(a0, a1, om, 3)) % box
        acc = _accel(x, box, n_grid, om)
        p += acc * _factor(ah, a1, om, 2)
    return x, p


def zeldovich_pm_ic(cosmo: CosmologySpec, *, box: float, n_grid: int,
                    z_start: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Build (positions, canonical momenta) for a PM run from Zel'dovich ICs.

    The growing-mode momentum p = a²·f·E·Ψ (Ψ = displacement at a_start) is
    what makes the PM reproduce linear growth; starting from rest under-grows.
    """
    from oneuniverse.simulation.linear.growth import growth_rate
    from oneuniverse.simulation.linear.zeldovich import zeldovich_particles

    a0 = 1.0 / (1.0 + z_start)
    pos, _ = zeldovich_particles(cosmo, box_size=box, n_grid=n_grid,
                                 z=z_start, seed=seed)
    g = (np.arange(n_grid) + 0.5) * box / n_grid
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    disp = (pos - q + box / 2.0) % box - box / 2.0
    p0 = a0 ** 2 * growth_rate(z_start, cosmo) * _E(a0, cosmo.omega_m) * disp
    return pos, p0
