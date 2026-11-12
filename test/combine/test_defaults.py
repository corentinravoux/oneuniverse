import pyarrow as pa
import pytest

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse.combine import default_weight_for, register_default
from oneuniverse.combine import defaults as _defaults_mod
from oneuniverse.combine.defaults import defaults
from oneuniverse.combine.weights import ConstantWeight, InverseVarianceWeight
from oneuniverse.data.schema import z_types


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = dict(_defaults_mod._DEFAULTS)
    yield
    _defaults_mod._DEFAULTS.clear()
    _defaults_mod._DEFAULTS.update(saved)


def test_builtin_keys_use_registered_z_types():
    for (_, zt) in defaults():
        assert zt in z_types()
    w = default_weight_for("peculiar_velocity", "pv")
    assert isinstance(w, InverseVarianceWeight) and w.err_column == "v_pec_err"
    with pytest.raises(NotFoundError, match="peculiar_velocity"):
        default_weight_for("spectroscopic", "pv")


def test_register_default_replace_policy():
    with pytest.raises(ValueError):
        register_default("spectroscopic", "spec", lambda: ConstantWeight())
    with pytest.raises(ValidationError, match="z_type"):
        register_default("spectroscopic", "nope", lambda: ConstantWeight())
    register_default("spectroscopic", "spec", lambda: ConstantWeight(3.0), replace=True)
    try:
        assert default_weight_for("spectroscopic", "spec")(pa.table({"x": [1]}))[0] == 3.0
    finally:
        register_default("spectroscopic", "spec", lambda: InverseVarianceWeight("z_err"), replace=True)


def test_defaults_returns_copy():
    d = defaults()
    d.clear()
    assert defaults()
