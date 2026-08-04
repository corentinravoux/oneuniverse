"""A 3rd engine on the plug-in contract: constrained realization as a
ReconstructionEngine (Wiener mean + statistically-correct small-scale power).
Proves register_engine flexes for a differently-shaped engine without contract
changes — the socket real BORG/SBI engines will later fill.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.twin.constrained import constrained_realization
from oneuniverse.twin.engine import Observation, ReconstructionEngine, register_engine


@register_engine
class ConstrainedRealization(ReconstructionEngine):
    name = "constrained"

    def reconstruct(self, observation: Observation, *, cosmo: CosmologySpec,
                    box_size: float, z: float = 0.0) -> np.ndarray:
        return constrained_realization(
            observation.delta_g, cosmo, box_size=box_size,
            nbar=observation.nbar, bias=observation.bias, z=z, seed=0)
