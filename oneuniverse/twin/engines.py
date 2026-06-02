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
        # ``ic`` is reserved (the PM engine will integrate it); the linear
        # engine returns the growth-scaled linear field for the seed.
        delta = generate_density_field(cosmo, box_size=box_size,
                                       n_grid=n_grid, z=z, seed=seed)
        return ProductBundle(
            fields={"delta": delta},
            meta={"engine": self.name, "box_size": box_size,
                  "n_grid": n_grid, "z": z, "seed": seed},
        )
