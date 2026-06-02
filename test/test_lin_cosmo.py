"""Phase S3 T1 — linear-sim cosmology validator."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def _full() -> CosmologySpec:
    return CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
        sigma8=0.81, t_cmb=2.7255,
    )


def test_require_passes_for_full_cosmo():
    c = require_cosmo(_full())
    assert c.omega_m == 0.31
    # t_cmb defaulted if missing handled separately
    assert c.t_cmb == 2.7255


def test_require_defaults_tcmb():
    c = require_cosmo(CosmologySpec(
        omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
    ))
    assert c.t_cmb == 2.7255


def test_require_rejects_missing_field():
    with pytest.raises(ValueError, match="omega_m"):
        require_cosmo(CosmologySpec(
            omega_b=0.048, h=0.67, n_s=0.96, sigma8=0.81,
        ))
