"""Phase S3 T6 — Zel'dovich particles (particle product)."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_particle_count_and_shapes():
    pos, vel = zeldovich_particles(
        _cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=1,
    )
    assert pos.shape == (16 ** 3, 3)
    assert vel.shape == (16 ** 3, 3)


def test_positions_wrapped_in_box():
    box = 200.0
    pos, _ = zeldovich_particles(
        _cosmo(), box_size=box, n_grid=16, z=0.0, seed=1,
    )
    assert pos.min() >= 0.0
    assert pos.max() < box


def test_reproducible():
    a_pos, a_vel = zeldovich_particles(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=4)
    b_pos, b_vel = zeldovich_particles(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=4)
    np.testing.assert_array_equal(a_pos, b_pos)
    np.testing.assert_array_equal(a_vel, b_vel)


def test_mean_displacement_small():
    box, n = 200.0, 16
    pos, _ = zeldovich_particles(_cosmo(), box_size=box, n_grid=n, z=0.0, seed=2)
    # Lagrangian grid centres.
    cell = box / n
    g = (np.arange(n) + 0.5) * cell
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    q = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    disp = pos - q
    # Periodic-wrap the displacement to [-box/2, box/2).
    disp = (disp + box / 2.0) % box - box / 2.0
    assert abs(float(disp.mean())) < 1.0
