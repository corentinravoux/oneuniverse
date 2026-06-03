"""Phase S8.1 T1 — CIC deposit + FFT Poisson force."""
import numpy as np

from oneuniverse.simulation.pm.deposit import deposit_cic, interpolate_cic
from oneuniverse.simulation.pm.poisson import pm_force


def test_cic_conserves_mass():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 100, (5000, 3))
    rho = deposit_cic(pos, 32, 100.0)
    assert abs(rho.sum() - 5000) < 1e-6        # CIC conserves total mass


def test_cic_partition_of_unity():
    # a single particle deposits total weight 1 over its 8 nodes
    rho = deposit_cic(np.array([[10.3, 20.7, 5.1]]), 32, 100.0)
    assert abs(rho.sum() - 1.0) < 1e-9
    assert (rho > 0).sum() <= 8


def test_interpolate_constant_field():
    grid = np.full((16, 16, 16), 3.0)
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 50, (100, 3))
    vals = interpolate_cic(grid, pos, 50.0)
    np.testing.assert_allclose(vals, 3.0)      # CIC of a constant = constant


def test_force_points_toward_overdensity():
    # single overdense cell -> neighbour force points toward it
    n, box = 16, 16.0
    delta = np.full((n, n, n), -1.0 / (n ** 3 - 1))
    delta[8, 8, 8] = 1.0
    delta -= delta.mean()
    gx, gy, gz = pm_force(delta, box)
    # cell at (7,8,8) should be pulled in +x toward (8,8,8)
    assert gx[7, 8, 8] > 0
    assert gx[9, 8, 8] < 0
