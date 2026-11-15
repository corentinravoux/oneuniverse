"""Reference engines registered on the twin contract."""
import numpy as np
import pytest
from fixtures.cosmology import COSMO

import oneuniverse.reference.engines  # noqa: F401  (registers the engines at collection time)
from oneuniverse._core.errors import ValidationError
from oneuniverse.twin.engine import get_engine, registered_engines


def test_four_engines_registered():
    assert {"linear", "fastpm", "wiener", "constrained"} <= set(registered_engines())
    b = get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=0); assert b.fields["delta"].shape == (16, 16, 16)


def test_linear_scales_ic_by_growth_and_records_provenance():
    from oneuniverse.reference.linear import growth_factor
    ic = get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=3).fields["delta"]
    b = get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=16, z=1.0, ic_field=ic)
    assert np.allclose(b.fields["delta"], ic * growth_factor(1.0, COSMO)) and b.z == 1.0
    assert b.provenance["engine"] == "linear"
    assert get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=3).provenance["seed"] == 3


def test_forward_needs_seed_or_ic_and_checks_ic_shape():
    eng = get_engine("linear")()
    with pytest.raises(ValidationError):
        eng.forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0)
    with pytest.raises(ValidationError):
        eng.forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, ic_field=np.zeros((8, 8, 8)))


def test_fastpm_runs_and_is_deterministic():
    eng = get_engine("fastpm")(n_steps=4)
    a = eng.forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=1)
    b = eng.forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=1)
    assert a.fields["delta"].shape == (16,) * 3 and np.array_equal(a.fields["delta"], b.fields["delta"])
    assert a.provenance["parameters"]["z_start"] == 9.0 and a.provenance["parameters"]["n_steps"] == 4
    assert abs(a.fields["delta"].mean()) < 1e-10


def test_constrained_requires_seed_and_reconstruction_engines_run():
    from oneuniverse.twin.engine import Observation
    with pytest.raises(TypeError):
        get_engine("constrained")()
    d = np.random.default_rng(0).standard_normal((8, 8, 8)); d -= d.mean()
    obs = Observation(d, 1e-3, 1.0, None, 64.0)
    w = get_engine("wiener")().reconstruct(obs, cosmo=COSMO)
    c1 = get_engine("constrained")(seed=5).reconstruct(obs, cosmo=COSMO)
    c2 = get_engine("constrained")(seed=5).reconstruct(obs, cosmo=COSMO)
    assert w.shape == c1.shape == (8, 8, 8) and np.array_equal(c1, c2)


def test_reconstruction_needs_box():
    from oneuniverse.twin.engine import Observation
    with pytest.raises(ValidationError):
        get_engine("wiener")().reconstruct(Observation(np.zeros((8, 8, 8)), 1e-3, 1.0), cosmo=COSMO)


@pytest.mark.parametrize("bad", [1.5, 2.0, "3", True, -1])
def test_non_integer_seed_and_n_steps_refused(bad):
    with pytest.raises(ValidationError):
        get_engine("constrained")(seed=bad)
    with pytest.raises(ValidationError):
        get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=8, z=0.0, seed=bad)
    with pytest.raises(ValidationError):
        get_engine("fastpm")(n_steps=bad)


def test_numpy_integers_accepted():
    b = get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=8, z=0.0, seed=np.int64(4))
    assert b.provenance["seed"] == 4 and get_engine("fastpm")(n_steps=np.int32(3)).n_steps == 3
