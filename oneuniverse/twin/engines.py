"""Concrete twin engines — the first plugin of each role.

- ``WienerReconstruction`` (reconstruction): the C1 Wiener filter, wrapped.
- ``LinearForwardEngine`` (forward): the dummy linear field generator.

Two engines, two roles → the generality contract is demonstrated on the
dummy. The fast-PM mini-sim (S8.1) becomes the second ForwardEngine.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.engine import (
    ForwardEngine,
    Observation,
    ProductBundle,
    ReconstructionEngine,
    register_engine,
)
from oneuniverse.twin.wiener import wiener_reconstruct


@register_engine
class WienerReconstruction(ReconstructionEngine):
    name = "wiener"

    def reconstruct(self, observation: Observation, *,
                    cosmo: CosmologySpec, box_size: float,
                    z: float = 0.0) -> np.ndarray:
        return wiener_reconstruct(
            observation.delta_g, cosmo, box_size=box_size,
            nbar=observation.nbar, bias=observation.bias, z=z,
        )


@register_engine
class LinearForwardEngine(ForwardEngine):
    name = "linear"

    def forward(self, *, cosmo: CosmologySpec, box_size: float, n_grid: int,
                z: float = 0.0, seed: int = 0,
                ic: Optional[np.ndarray] = None) -> ProductBundle:
        # if a z=0 IC field is supplied (the store-boundary path), the linear
        # forward is that field; otherwise generate one from the seed.
        delta = (np.asarray(ic, float) if ic is not None
                 else generate_density_field(cosmo, box_size=box_size,
                                             n_grid=n_grid, z=z, seed=seed))
        return ProductBundle(
            fields={"delta": delta},
            meta={"engine": self.name, "box_size": box_size,
                  "n_grid": n_grid, "z": z, "seed": seed},
        )


@register_engine
class PMForwardEngine(ForwardEngine):
    """Non-linear forward model — the fast particle-mesh mini-sim."""
    name = "fastpm"

    def forward(self, *, cosmo: CosmologySpec, box_size: float, n_grid: int,
                z: float = 0.0, seed: int = 0,
                ic: Optional[np.ndarray] = None,
                z_start: float = 9.0, n_steps: int = 20) -> ProductBundle:
        from oneuniverse.simulation.pm.deposit import deposit_cic
        from oneuniverse.simulation.pm.run import (
            run_pm, zeldovich_pm_ic, zeldovich_pm_ic_from_field,
        )

        if ic is not None:                         # store-boundary IC path
            pos, p0 = zeldovich_pm_ic_from_field(cosmo, np.asarray(ic, float),
                                                 box=box_size, n_grid=n_grid,
                                                 z_start=z_start)
        else:
            pos, p0 = zeldovich_pm_ic(cosmo, box=box_size, n_grid=n_grid,
                                      z_start=z_start, seed=seed)
        x, _ = run_pm(pos, p0, box=box_size, n_grid=n_grid, cosmo=cosmo,
                      a_start=1.0 / (1.0 + z_start), a_end=1.0 / (1.0 + z),
                      n_steps=n_steps)
        rho = deposit_cic(x, n_grid, box_size)
        delta = rho / rho.mean() - 1.0
        return ProductBundle(
            fields={"delta": delta},
            meta={"engine": self.name, "box_size": box_size, "n_grid": n_grid,
                  "z": z, "seed": seed, "z_start": z_start},
        )
