import numpy as np
from oneuniverse.twin.engine import get_engine, registered_engines, Observation
from oneuniverse.simulation.cosmology import CosmologySpec
import oneuniverse.twin.engines_extra  # noqa: F401  (triggers registration)


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_constrained_engine_registered_and_runs():
    assert "constrained" in registered_engines()
    Eng = get_engine("constrained")
    eng = Eng()
    rng = np.random.default_rng(0)
    obs = Observation(delta_g=rng.normal(size=(16, 16, 16)), nbar=5e-3, bias=1.5)
    field = eng.reconstruct(obs, cosmo=_cosmo(), box_size=200.0, z=0.0)
    assert field.shape == (16, 16, 16)
    assert np.isfinite(field).all()
