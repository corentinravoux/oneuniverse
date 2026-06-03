"""Initial-conditions (input) product — the white-noise realisation.

The seeded white-noise field is the deterministic IC a forward model
integrates from (it is what the Gaussian-field generator colours by
sqrt(P(k))). Storing it + a descriptor realises the format's input side
(``has_input=True``). No sampler is run (Rule 4) — this is the fixed
realisation the dummy already uses.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec


def white_noise_ic(cosmo: CosmologySpec, *, box_size: float, n_grid: int,
                   seed: int = 0) -> Tuple[np.ndarray, Dict]:
    """Return (white-noise IC field, descriptor)."""
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((n_grid, n_grid, n_grid))
    descriptor = {
        "seed": int(seed),
        "box_size": float(box_size),
        "n_grid": int(n_grid),
        "pk_model": "eisenstein_hu_nowiggle",
        "cosmology": cosmo.to_dict(),
    }
    return field, descriptor
