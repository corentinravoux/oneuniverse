"""Field-validation estimator suite."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.validation import validate_field


def _ref():
    c = CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                      sigma8=0.81, t_cmb=2.7255)
    return generate_density_field(c, box_size=200.0, n_grid=48, z=0.0, seed=2)


def test_identical_field_is_perfect():
    b = _ref()
    v = validate_field(b, b, box=200.0)
    g = np.isfinite(v.r)
    assert np.allclose(v.r[g], 1.0, atol=1e-6)
    assert np.allclose(v.transfer[g], 1.0, atol=1e-6)
    assert np.allclose(v.power_ratio[g], 1.0, atol=1e-6)
    assert np.allclose(v.stochasticity[g], 0.0, atol=1e-6)
    assert np.isnan(v.k_half)


def test_scaled_field_transfer_and_power():
    b = _ref()
    v = validate_field(0.5 * b, b, box=200.0)
    g = np.isfinite(v.transfer)
    assert np.allclose(v.transfer[g], 0.5, atol=1e-6)   # amplitude
    assert np.allclose(v.power_ratio[g], 0.25, atol=1e-6)
    assert np.allclose(v.r[g], 1.0, atol=1e-6)          # phases unchanged


def test_noise_raises_stochasticity():
    b = _ref()
    rng = np.random.default_rng(0)
    v = validate_field(b + 3.0 * rng.standard_normal(b.shape), b, box=200.0)
    band = v.k > 0.3
    assert np.nanmedian(v.stochasticity[band]) > 0.2    # noise -> stochastic
    assert np.isfinite(v.k_half)
