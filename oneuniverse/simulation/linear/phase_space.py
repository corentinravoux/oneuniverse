"""Phase-space sheet (the Zel'dovich Lagrangian → Eulerian map).

The OUF-Sim ``phase_space`` product: each particle's Lagrangian grid
position q together with its Eulerian position x and velocity v. Natural
access is by Lagrangian region, so the store partitions by q.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles


def phase_space_sheet(cosmo: CosmologySpec, *, box_size, n_grid, z=0.0,
                      seed=0) -> Dict[str, np.ndarray]:
    pos, vel = zeldovich_particles(cosmo, box_size=box_size, n_grid=n_grid,
                                   z=z, seed=seed)
    cell = box_size / n_grid
    g = (np.arange(n_grid) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    return {
        "qx": qx.ravel(), "qy": qy.ravel(), "qz": qz.ravel(),
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
    }
