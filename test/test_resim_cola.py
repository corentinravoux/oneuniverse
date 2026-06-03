"""Phase S11b T1/T2 — COLA-frame PM (few-step large-scale accuracy)."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.pm.deposit import deposit_cic
from oneuniverse.simulation.pm.run import run_pm, zeldovich_pm_ic_from_field
from oneuniverse.simulation.resim.cola import cola_run_pm
from oneuniverse.twin.verify import cross_correlation, power_ratio


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _delta(x, n):
    rho = deposit_cic(x, n, _BOX)
    return rho / rho.mean() - 1.0


_BOX = 200.0


def test_cola_few_steps_matches_full_pm_large_scales():
    c = _cosmo()
    n = 64
    d0 = generate_density_field(c, box_size=_BOX, n_grid=n, z=0.0, seed=2)
    # full PM reference, many steps
    pos, p0 = zeldovich_pm_ic_from_field(c, d0, box=_BOX, n_grid=n, z_start=9.0)
    xf, _ = run_pm(pos, p0, box=_BOX, n_grid=n, cosmo=c, a_start=0.1,
                   a_end=1.0, n_steps=25)
    df = _delta(xf, n)
    # COLA with only 5 steps
    xc = cola_run_pm(c, d0, box=_BOX, n_grid=n, a_start=0.1, a_end=1.0,
                     n_steps=5)
    dc = _delta(xc, n)
    k, r = cross_correlation(dc, df, box_size=_BOX)
    _, pr = power_ratio(dc, df, box_size=_BOX)
    low = k < 0.1
    # large scales reproduced by few-step COLA (carried by the analytic LPT)
    assert np.nanmedian(r[low]) > 0.95
    assert 0.9 < np.nanmedian(pr[low]) < 1.1
