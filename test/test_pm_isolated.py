"""Phase S16 T1 — isolated (zero-padded) Poisson force: no periodic images."""
import numpy as np

from oneuniverse.simulation.pm.poisson import pm_force, pm_force_isolated


def _spike(n):
    d = np.zeros((n, n, n)); d[n // 2, n // 2, n // 2] = 1.0
    return d - d.mean()


def test_isolated_force_points_toward_mass():
    n = 32; gx = pm_force_isolated(_spike(n), 32.0)[0]
    c = n // 2
    assert gx[c - 1, c, c] > 0          # cell left of mass -> +x toward it
    assert gx[c + 1, c, c] < 0


def test_isolated_falls_as_inverse_square():
    n = 32; gx = pm_force_isolated(_spike(n), 32.0)[0]; c = n // 2
    g = [abs(gx[c + r, c, c]) for r in (2, 4, 8)]
    # 1/r^2: doubling r quarters the force (within discretisation)
    assert 3.0 < g[0] / g[1] < 5.0
    assert 3.0 < g[1] / g[2] < 5.0


def test_isolated_removes_periodic_images():
    n = 32; c = n // 2
    gi = pm_force_isolated(_spike(n), 32.0)[0]
    gp = pm_force(_spike(n), 32.0)[0]
    # far from the mass (r = box/4) the periodic force is image-contaminated;
    # the isolated one is much smaller (true open 1/r^2)
    assert abs(gi[c + 8, c, c]) < 0.5 * abs(gp[c + 8, c, c])
