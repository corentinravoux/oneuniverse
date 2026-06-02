"""Phase C1 T1 — mock biased Poisson tracers."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.mock_observe import mock_tracer_field


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_mean_density_and_overdensity():
    c = _cosmo()
    d = generate_density_field(c, box_size=256.0, n_grid=64, z=0.0, seed=1)
    obs = mock_tracer_field(d, box_size=256.0, nbar=5e-2, bias=1.5, seed=2)
    # mean tracer overdensity ~ 0 (Poisson around 1+bδ, δ mean 0)
    assert abs(float(obs["delta_g"].mean())) < 0.1
    # more tracers land in overdense cells: positive correlation with δ
    corr = np.corrcoef(obs["counts"].ravel(), d.ravel())[0, 1]
    assert corr > 0.2


def test_reproducible_and_nonnegative_counts():
    c = _cosmo()
    d = generate_density_field(c, box_size=256.0, n_grid=32, z=0.0, seed=1)
    a = mock_tracer_field(d, box_size=256.0, nbar=1e-2, bias=2.0, seed=7)
    b = mock_tracer_field(d, box_size=256.0, nbar=1e-2, bias=2.0, seed=7)
    np.testing.assert_array_equal(a["counts"], b["counts"])
    assert a["counts"].min() >= 0
