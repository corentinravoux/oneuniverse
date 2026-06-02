"""Phase S3 T5 — Gaussian density field (mesh / voxel product)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.power_spectrum import linear_power


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_shape_and_real():
    d = generate_density_field(_cosmo(), box_size=200.0, n_grid=32, z=0.0, seed=1)
    assert d.shape == (32, 32, 32)
    assert np.isrealobj(d)


def test_mean_near_zero():
    d = generate_density_field(_cosmo(), box_size=200.0, n_grid=32, z=0.0, seed=1)
    assert abs(float(d.mean())) < 0.05


def test_reproducible_with_seed():
    a = generate_density_field(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=7)
    b = generate_density_field(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=7)
    np.testing.assert_array_equal(a, b)


def test_variance_matches_mode_sum():
    """Real-space variance ~ (1/V) sum_k P(k_grid), within cosmic scatter."""
    c = _cosmo()
    box, n = 200.0, 32
    d = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=3)
    measured = float(d.var())
    # Predicted variance from the same grid's mode sum.
    kx = np.fft.fftfreq(n, d=box / n) * 2.0 * np.pi
    kxg, kyg, kzg = np.meshgrid(kx, kx, kx, indexing="ij")
    kmag = np.sqrt(kxg ** 2 + kyg ** 2 + kzg ** 2)
    kmag_flat = kmag.ravel()
    pk = np.zeros_like(kmag_flat)
    nz = kmag_flat > 0
    pk[nz] = linear_power(kmag_flat[nz], c, z=0.0)
    predicted = pk.sum() / box ** 3
    assert measured == pytest.approx(predicted, rel=0.35)
