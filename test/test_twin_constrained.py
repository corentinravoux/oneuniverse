"""Phase C5 — Hoffman-Ribak constrained realization.

Tested on a *clean linear* observation δ_g = b·δ_m + Gaussian noise (the
model the Wiener/HR machinery assumes), sparse enough (low n̄) that shot
noise dominates the small scales — so the Wiener mean is visibly power-
suppressed and the constrained realization visibly restores it. The Poisson
mock's bias non-linearity (clip at σ≳1) is a separate concern exercised by
the C1/C4 r(k) tests; here we isolate the HR algorithm.
"""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.constrained import constrained_realization
from oneuniverse.twin.verify import cross_correlation, power_ratio
from oneuniverse.twin.wiener import wiener_reconstruct


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _linear_obs(box=256.0, n=64, nbar=1e-3, bias=1.5, seed=2):
    c = _cosmo()
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=seed)
    v_cell = (box / n) ** 3
    rng = np.random.default_rng(seed + 7)
    eps = rng.normal(0.0, 1.0 / np.sqrt(nbar * v_cell), size=(n, n, n))
    dg = bias * truth + eps                       # exactly b·δ_m + noise(N)
    return c, box, nbar, bias, truth, dg


def test_cr_restores_small_scale_power():
    c, box, nbar, bias, truth, dg = _linear_obs()
    wf = wiener_reconstruct(dg, c, box_size=box, nbar=nbar, bias=bias)
    cr = constrained_realization(dg, c, box_size=box, nbar=nbar, bias=bias,
                                 seed=99)
    k, rwf = power_ratio(wf, truth, box_size=box)
    _, rcr = power_ratio(cr, truth, box_size=box)
    band = k > 0.3
    assert np.nanmedian(rwf[band]) < 0.5      # Wiener mean is suppressed
    assert 0.7 < np.nanmedian(rcr[band]) < 1.4   # CR restores P(k) ~ 1


def test_cr_preserves_large_scale_constraint():
    c, box, nbar, bias, truth, dg = _linear_obs()
    cr = constrained_realization(dg, c, box_size=box, nbar=nbar, bias=bias,
                                 seed=99)
    k, r = cross_correlation(cr, truth, box_size=box)
    assert np.nanmedian(r[k < 0.05]) > 0.7    # large scales still constrained


def test_cr_ensemble_mean_approaches_wiener():
    c, box, nbar, bias, truth, dg = _linear_obs()
    wf = wiener_reconstruct(dg, c, box_size=box, nbar=nbar, bias=bias)
    crs = [constrained_realization(dg, c, box_size=box, nbar=nbar, bias=bias,
                                   seed=s) for s in range(8)]
    mean = np.mean(crs, axis=0)
    assert np.corrcoef(mean.ravel(), wf.ravel())[0, 1] > 0.9
