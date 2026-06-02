"""Hoffman-Ribak constrained realization.

The Wiener filter gives the minimum-variance *mean* field — but it is
power-suppressed on small scales (it smooths away what the data cannot
constrain). For a *forward-modellable IC* we need a field that (a) matches
the data where it is informative and (b) has the correct P(k) everywhere.

Hoffman & Ribak (1991):

    δ_CR = δ_WF(d) + [ δ_rand − δ_WF(d_rand) ]

where δ_rand is an unconstrained random signal with the right P(k), and
d_rand is a mock observation of it (same bias + noise model). The residual
restores the unconstrained variance; the ensemble mean over realisations is
exactly δ_WF. This δ_CR is the IC the resimulation (S8) forward-evolves.
"""
from __future__ import annotations

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.wiener import wiener_reconstruct


def constrained_realization(delta_g, cosmo: CosmologySpec, *, box_size, nbar,
                            bias=1.0, z=0.0, seed=0) -> np.ndarray:
    """Return a constrained realization of the matter field given ``delta_g``."""
    d = np.asarray(delta_g, dtype=np.float64)
    n = d.shape[0]

    wf_data = wiener_reconstruct(d, cosmo, box_size=box_size, nbar=nbar,
                                 bias=bias, z=z)

    # Unconstrained random signal with the correct P(k).
    delta_rand = generate_density_field(cosmo, box_size=box_size, n_grid=n,
                                        z=z, seed=seed)
    # Noise realisation whose power equals the shot noise N = 1/nbar:
    # white field with per-cell variance 1/nbar_cell.
    v_cell = (box_size / n) ** 3
    rng = np.random.default_rng(seed + 104729)
    eps = rng.normal(0.0, 1.0 / np.sqrt(nbar * v_cell), size=(n, n, n))
    d_rand = bias * delta_rand + eps

    wf_rand = wiener_reconstruct(d_rand, cosmo, box_size=box_size, nbar=nbar,
                                 bias=bias, z=z)
    return wf_data + delta_rand - wf_rand
