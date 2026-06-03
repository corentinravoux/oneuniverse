"""Full-volume far-field provider — the long-range force for mini-sims.

The full-sim's global potential mesh φ(x; a) (∇²φ = δ) is the long-range
force every mini-sim consumes (the COLA far-field / TreePM long-range split).
In linear theory δ ∝ D(a), so φ(a) = φ₀ · D(a)/D(z₀). Sub-region service
reuses the IC-extraction slicing.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gr_fields import potential_field
from oneuniverse.simulation.linear.growth import growth_factor
from oneuniverse.simulation.resim.ic_extract import extract_region
from oneuniverse.simulation.selectors import Cube


def far_field_potential(delta: np.ndarray, *, box_size: float,
                        cosmo: CosmologySpec, scale_factors: Sequence[float],
                        z_field: float = 0.0) -> Dict[float, np.ndarray]:
    """Return {a: φ(x; a)} growth-scaled from the field's potential."""
    phi0 = potential_field(delta, box_size=box_size)
    d_field = growth_factor(z_field, cosmo)
    out = {}
    for a in scale_factors:
        z = 1.0 / a - 1.0
        out[float(a)] = phi0 * (growth_factor(z, cosmo) / d_field)
    return out


def far_field_box(phi: np.ndarray, cube: Cube, *, box_size: float
                  ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Serve the far-field potential over a sub-region (partial access)."""
    return extract_region(phi, cube, box_size=box_size)
