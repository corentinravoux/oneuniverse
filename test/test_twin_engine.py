"""Phase C2 — engine contracts (ReconstructionEngine + ForwardEngine)."""
import numpy as np
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.twin.engine import (
    ForwardEngine,
    Observation,
    ProductBundle,
    ReconstructionEngine,
    get_engine,
    registered_engines,
)
from oneuniverse.twin.engines import LinearForwardEngine, PMForwardEngine, WienerReconstruction
from oneuniverse.twin.mock_challenge import run_mock_challenge
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.wiener import wiener_reconstruct


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_abcs_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ReconstructionEngine()
    with pytest.raises(TypeError):
        ForwardEngine()


def test_roles_and_registry():
    assert WienerReconstruction.role == "reconstruction"
    assert LinearForwardEngine.role == "forward"
    assert get_engine("wiener") is WienerReconstruction
    assert get_engine("linear") is LinearForwardEngine
    assert {"wiener", "linear"} <= set(registered_engines())


def test_wiener_engine_matches_function():
    c = _cosmo()
    box, n, b, nbar = 256.0, 64, 1.5, 5e-2
    truth = generate_density_field(c, box_size=box, n_grid=n, z=0.0, seed=4)
    obs_d = mock_tracer_field(truth, box_size=box, nbar=nbar, bias=b, seed=5)
    obs = Observation(delta_g=obs_d["delta_g"], nbar=nbar, bias=b)
    eng = WienerReconstruction()
    rec = eng.reconstruct(obs, cosmo=c, box_size=box, z=0.0)
    ref = wiener_reconstruct(obs_d["delta_g"], c, box_size=box, nbar=nbar,
                             bias=b)
    np.testing.assert_array_equal(rec, ref)


def test_linear_forward_engine_returns_bundle():
    c = _cosmo()
    eng = LinearForwardEngine()
    bundle = eng.forward(cosmo=c, box_size=200.0, n_grid=32, z=0.0, seed=1)
    assert isinstance(bundle, ProductBundle)
    ref = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=1)
    np.testing.assert_array_equal(bundle.fields["delta"], ref)


def test_c1_loop_through_contract_unchanged():
    c = _cosmo()
    box, n, b, nbar, seed = 256.0, 64, 1.5, 5e-2, 11
    truth = LinearForwardEngine().forward(cosmo=c, box_size=box, n_grid=n,
                                          z=0.0, seed=seed).fields["delta"]
    obs_d = mock_tracer_field(truth, box_size=box, nbar=nbar, bias=b,
                              seed=seed + 1)
    obs = Observation(delta_g=obs_d["delta_g"], nbar=nbar, bias=b)
    rec = WienerReconstruction().reconstruct(obs, cosmo=c, box_size=box)
    ref = run_mock_challenge(c, box_size=box, n_grid=n, nbar=nbar, bias=b,
                             seed=seed)
    np.testing.assert_array_equal(rec, ref["rec"])


def test_two_forward_engines_satisfy_contract():
    from oneuniverse.twin.engine import ProductBundle, registered_engines
    from oneuniverse.twin.engines import LinearForwardEngine, PMForwardEngine
    from oneuniverse.simulation.oufsim import write_oufsim_store  # noqa: F401
    assert {"linear", "fastpm", "wiener"} <= set(registered_engines())
    c = _cosmo()
    out = PMForwardEngine().forward(cosmo=c, box_size=200.0, n_grid=32,
                                    z=0.0, seed=2, n_steps=12)
    assert isinstance(out, ProductBundle) and out.fields["delta"].shape == (32,) * 3
