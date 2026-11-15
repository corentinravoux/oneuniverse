"""Engine contracts and registry of ``oneuniverse.twin.engine`` (no physics)."""

import dataclasses

import numpy as np
import pytest

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse.twin import engine as E
from oneuniverse.twin.engine import (
    ForwardEngine,
    Observation,
    ProductBundle,
    ReconstructionEngine,
    get_engine,
    register_engine,
    registered_engines,
)


@pytest.fixture
def clean_registry():
    """Drop any engine registered by a test, even when the test fails."""
    before = set(registered_engines())
    yield
    for name in set(registered_engines()) - before:
        E._REGISTRY.items_dict.pop(name, None)


def test_registry_and_abstracts(clean_registry):
    @register_engine
    class Dummy(ForwardEngine):
        name = "dummy"
        def forward(self, *, cosmo, box, n_grid, z, seed=None, ic_field=None): return ProductBundle({"delta": np.zeros((n_grid,) * 3)}, z, {"engine": "dummy"})
    assert get_engine("dummy") is Dummy and "dummy" in registered_engines()
    with pytest.raises(ValueError): register_engine(Dummy)
    with pytest.raises(TypeError): ForwardEngine()
    obs = Observation(np.zeros((4, 4, 4)), nbar=1e-3, bias=1.0); assert obs.mask is None


def test_reconstruction_engine_shares_registry(clean_registry):
    @register_engine
    class DummyRec(ReconstructionEngine):
        name = "dummy_rec"
        def reconstruct(self, observation, *, cosmo, z=0.0):
            return np.zeros_like(observation.delta_g)

    assert get_engine("dummy_rec") is DummyRec
    assert isinstance(registered_engines(), tuple)
    with pytest.raises(TypeError): ReconstructionEngine()


def test_unknown_name_and_missing_name(clean_registry):
    with pytest.raises(NotFoundError):
        get_engine("no_such_engine_xyz")

    class Nameless(ForwardEngine):
        def forward(self, *, cosmo, box, n_grid, z, seed=None, ic_field=None):
            raise NotImplementedError

    with pytest.raises(ValidationError):
        register_engine(Nameless)
    assert "" not in registered_engines()


def test_observation_defaults(clean_registry):
    obs = Observation(np.zeros((2, 2, 2)), nbar=1e-3, bias=1.5)
    assert obs.box == 0.0 and obs.meta == {}
    other = Observation(np.zeros((2, 2, 2)), nbar=1e-3, bias=1.5)
    assert obs.meta is not other.meta
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.box = 1.0  # frozen


def test_register_engine_rejects_non_engine_class(clean_registry):
    class NotAnEngine:
        name = "nope"
    with pytest.raises(ValidationError):
        register_engine(NotAnEngine)
    with pytest.raises(ValidationError):
        register_engine("linear")
    assert "nope" not in registered_engines()


def test_unknown_engine_hint_mentions_import():
    with pytest.raises(NotFoundError, match="imported"):
        get_engine("surely-not-registered")
