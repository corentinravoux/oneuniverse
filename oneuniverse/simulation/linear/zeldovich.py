"""Zel'dovich (1LPT) particle realisation from the linear density field.

This is the OUF-Sim "particles" product. The Zel'dovich displacement is
psi(q) = inverse-Laplacian gradient of -delta, i.e. in Fourier space
psi_k = i k / k^2 * delta_k. Particles start on a uniform Lagrangian
grid q and move to x = q + psi (already growth-scaled via delta(z)).
Velocities (km/s) follow v = a H(a) f psi in linear theory; here we use
the simple proportionality v = (H0 f / (1+z)) * psi for a toy field.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.growth import growth_rate


def zeldovich_particles(
    cosmo: CosmologySpec,
    *,
    box_size: float,
    n_grid: int,
    z: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities), each (n_grid^3, 3).

    Positions in Mpc/h wrapped to [0, box_size); velocities in km/s
    (toy linear-theory scaling).
    """
    c = require_cosmo(cosmo)
    n = int(n_grid)
    delta = generate_density_field(
        c, box_size=box_size, n_grid=n, z=z, seed=seed,
    )
    delta_k = np.fft.rfftn(delta)

    kx = np.fft.fftfreq(n, d=box_size / n) * 2.0 * np.pi
    kz = np.fft.rfftfreq(n, d=box_size / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kz, indexing="ij")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    k2[0, 0, 0] = 1.0  # avoid division by zero; DC displacement set to 0

    # psi_k = i k / k^2 * delta_k  (per component)
    psi = []
    for kg in (kxg, kyg, kzg):
        psi_k = 1j * kg / k2 * delta_k
        comp = np.fft.irfftn(psi_k, s=(n, n, n))
        psi.append(comp)
    psi = np.stack([p.ravel() for p in psi], axis=1)  # (n^3, 3)

    # Lagrangian grid centres.
    cell = box_size / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)

    pos = (q + psi) % box_size

    # Toy velocity: v = (H0 * f / (1+z)) * psi (km/s); H0 = 100 h km/s/Mpc.
    h0 = 100.0 * c.h
    f = growth_rate(z, c)
    vel = (h0 * f / (1.0 + z)) * psi
    return pos.astype(np.float64), vel.astype(np.float64)
