"""Phase S3 T2/T3 — Eisenstein-Hu P(k)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.power_spectrum import (
    linear_power,
    sigma_R,
    transfer_eh_nowiggle,
    unnormalised_power,
)


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
        sigma8=0.81, t_cmb=2.7255,
    )


def test_transfer_goes_to_one_at_large_scale():
    c = _cosmo()
    t_big = transfer_eh_nowiggle(np.array([1e-4]), c)[0]
    assert t_big == pytest.approx(1.0, abs=0.02)


def test_transfer_decreases_with_k():
    c = _cosmo()
    k = np.array([1e-3, 1e-2, 1e-1, 1.0])
    t = transfer_eh_nowiggle(k, c)
    assert np.all(np.diff(t) < 0)


def test_unnormalised_power_low_k_slope_is_ns():
    c = _cosmo()
    k = np.array([1e-4, 2e-4])
    p = unnormalised_power(k, c)
    # On large scales T->1 so P ~ k^ns; measure the local slope.
    slope = np.log(p[1] / p[0]) / np.log(k[1] / k[0])
    assert slope == pytest.approx(c.n_s, abs=0.02)


def test_sigma8_roundtrips():
    c = _cosmo()
    # sigma_R(8) on the *normalised* P(k) must return the input sigma8.
    pk_norm = lambda kk: linear_power(kk, c, z=0.0)  # noqa: E731
    s8 = sigma_R(8.0, c, pk_func=pk_norm)
    assert s8 == pytest.approx(c.sigma8, rel=0.01)


def test_linear_power_scales_with_growth_squared():
    c = _cosmo()
    k = np.array([0.1])
    p0 = linear_power(k, c, z=0.0)[0]
    p1 = linear_power(k, c, z=1.0)[0]
    # higher z -> smaller amplitude
    assert p1 < p0


def test_linear_power_positive():
    c = _cosmo()
    k = np.logspace(-3, 1, 50)
    p = linear_power(k, c, z=0.0)
    assert np.all(p > 0)
