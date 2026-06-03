"""Phase S8.2 — far-field potential provider."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.gr_fields import laplacian
from oneuniverse.simulation.linear.growth import growth_factor
from oneuniverse.simulation.resim.farfield import far_field_box, far_field_potential
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_potential_solves_poisson_and_scales_with_growth():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=2)
    phis = far_field_potential(d, box_size=200.0, cosmo=c,
                               scale_factors=[1.0, 0.5])
    # a=1 -> ∇²φ = δ (the z=0 field)
    np.testing.assert_allclose(laplacian(phis[1.0], box_size=200.0),
                               d - d.mean(), atol=1e-6)
    # a=0.5 (z=1) -> scaled by D(1)/D(0)
    ratio = growth_factor(1.0, c) / growth_factor(0.0, c)
    np.testing.assert_allclose(phis[0.5], phis[1.0] * ratio, rtol=1e-6)


def test_far_field_box_subregion():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=2)
    phi = far_field_potential(d, box_size=200.0, cosmo=c,
                              scale_factors=[1.0])[1.0]
    sub, origin = far_field_box(phi, Cube(0, 60, 0, 60, 0, 60), box_size=200.0)
    nx, ny, nz = sub.shape
    np.testing.assert_array_equal(
        sub, phi[origin[0]:origin[0] + nx, origin[1]:origin[1] + ny,
                 origin[2]:origin[2] + nz])
