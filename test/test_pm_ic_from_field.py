"""Phase S9 T1 — build a PM IC from a provided density field."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.pm.run import (
    zeldovich_pm_ic,
    zeldovich_pm_ic_from_field,
)


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_from_field_matches_seed_path():
    c = _cosmo()
    box, n, zs = 200.0, 32, 9.0
    # the from-field IC built from the z=0 field of a seed must match the
    # seed path (consistency: delta(z)=D(z)*delta(0), so psi scales by D)
    delta0 = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=5)
    pos_a, p_a = zeldovich_pm_ic(c, box=box, n_grid=n, z_start=zs, seed=5)
    pos_b, p_b = zeldovich_pm_ic_from_field(c, delta0, box=box, n_grid=n,
                                            z_start=zs)
    np.testing.assert_allclose(pos_a, pos_b, atol=1e-6)
    np.testing.assert_allclose(p_a, p_b, atol=1e-6)


def test_from_field_runs_for_arbitrary_field():
    c = _cosmo()
    box, n = 200.0, 32
    field = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=1)
    pos, p0 = zeldovich_pm_ic_from_field(c, field, box=box, n_grid=n,
                                         z_start=9.0)
    assert pos.shape == (n ** 3, 3)
    assert pos.min() >= 0.0 and pos.max() < box
