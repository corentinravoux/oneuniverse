"""End-to-end mock challenge: truth → mock-observe → constrain → verify.

The minimal data↔sim coupling loop on the dummy, where the truth is known
so recovery is measurable. Returns the cross-correlation r(k) (the
feasibility number) plus the fields for plotting.
"""
from __future__ import annotations

from typing import Dict

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.metrics import cross_correlation, power_ratio
from oneuniverse.twin.wiener import wiener_reconstruct


def run_mock_challenge(cosmo: CosmologySpec, *, box_size, n_grid, nbar,
                       bias=1.5, z=0.0, seed=0) -> Dict:
    truth = generate_density_field(cosmo, box_size=box_size, n_grid=n_grid,
                                   z=z, seed=seed)
    obs = mock_tracer_field(truth, box_size=box_size, nbar=nbar, bias=bias,
                            seed=seed + 1)
    rec = wiener_reconstruct(obs["delta_g"], cosmo, box_size=box_size,
                             nbar=nbar, bias=bias, z=z)
    k, r = cross_correlation(rec, truth, box_size=box_size)
    _, ratio = power_ratio(rec, truth, box_size=box_size)
    return {"truth": truth, "delta_g": obs["delta_g"], "rec": rec,
            "k": k, "r": r, "power_ratio": ratio,
            "nbar": nbar, "bias": bias, "box_size": box_size,
            "n_grid": n_grid}
