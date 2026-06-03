"""Phase S8.1 T2/T3 — PM leapfrog + linear-growth validation."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.pm.deposit import deposit_cic
from oneuniverse.simulation.pm.run import _factor, run_pm, zeldovich_pm_ic
from oneuniverse.twin.verify import cross_correlation, power_ratio


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_drift_kick_factors_positive():
    assert _factor(0.1, 1.0, 0.31, 2) > 0
    assert _factor(0.1, 1.0, 0.31, 3) > 0


def test_no_force_no_blowup():
    # uniform particles, zero momentum -> no force -> stays put, bounded
    rng = np.random.default_rng(0)
    n = 16
    g = (np.arange(n) + 0.5) * 100.0 / n
    qx, qy, qz = np.meshgrid(g, g, g, indexing="ij")
    pos = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    x, p = run_pm(pos, np.zeros_like(pos), box=100.0, n_grid=n, cosmo=_cosmo(),
                  a_start=0.1, a_end=1.0, n_steps=5)
    assert np.isfinite(x).all() and np.abs(p).max() < 1e-6


def test_pm_reproduces_linear_growth():
    c = _cosmo()
    box, n = 200.0, 64
    pos, p0 = zeldovich_pm_ic(c, box=box, n_grid=n, z_start=9.0, seed=2)
    x, _ = run_pm(pos, p0, box=box, n_grid=n, cosmo=c, a_start=0.1,
                  a_end=1.0, n_steps=30)
    rho = deposit_cic(x, n, box)
    dpm = rho / rho.mean() - 1.0
    lin0 = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=2)
    k, r = cross_correlation(dpm, lin0, box_size=box)
    _, pr = power_ratio(dpm, lin0, box_size=box)
    # large-scale phases + amplitude track linear growth to a few %
    assert np.nanmedian(r[k < 0.05]) > 0.95
    assert 0.85 < np.nanmedian(pr[k < 0.06]) < 1.15
