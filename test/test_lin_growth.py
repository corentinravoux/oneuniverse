"""Phase S3 T4 — linear growth factor D(z), rate f(z)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.growth import growth_factor, growth_rate


def _cosmo() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    )


def test_growth_normalised_to_one_at_z0():
    assert growth_factor(0.0, _cosmo()) == pytest.approx(1.0, abs=1e-9)


def test_growth_decreases_with_redshift():
    c = _cosmo()
    z = np.array([0.0, 0.5, 1.0, 2.0])
    d = np.array([growth_factor(zi, c) for zi in z])
    assert np.all(np.diff(d) < 0)


def test_growth_high_z_approaches_eds():
    # At high z (matter domination) D ~ a = 1/(1+z).
    c = _cosmo()
    z = 9.0
    d = growth_factor(z, c)
    a = 1.0 / (1.0 + z)
    # D(z)/D(0) ~ a/1 only loosely; check D(9) ~ 0.1 within 30%.
    assert d == pytest.approx(a, rel=0.3)


def test_growth_rate_is_omega_matter_power():
    c = _cosmo()
    f0 = growth_rate(0.0, c)
    # f(0) ~ Omega_m^0.55 ~ 0.31^0.55 ~ 0.52
    assert f0 == pytest.approx(0.31 ** 0.55, rel=0.05)
