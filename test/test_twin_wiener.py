"""Phase C1 T2 — Wiener reconstruction of the matter field."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.wiener import wiener_reconstruct


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_noise_free_recovers_truth():
    c = _cosmo()
    box, n, b = 256.0, 64, 1.5
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    # noise-free observed field: delta_g = b * truth (tests the operator)
    delta_g = b * truth
    rec = wiener_reconstruct(delta_g, c, box_size=box, nbar=1e9, bias=b)
    assert np.corrcoef(rec.ravel(), truth.ravel())[0, 1] > 0.98


def test_high_noise_suppresses_small_scales():
    c = _cosmo()
    box, n, b = 256.0, 64, 1.5
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    rng = np.random.default_rng(0)
    delta_g = b * truth + rng.standard_normal(truth.shape) * 2.0
    rec = wiener_reconstruct(delta_g, c, box_size=box, nbar=1e-3, bias=b)
    assert rec.var() < delta_g.var()
