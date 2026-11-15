"""Linear fixture simulation, packed layout and the two converters."""
import json

import pytest
from fixtures.native_sim import make_native_sim, write_packed_native_fixture

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.native import generate_linear_sim, write_packed_native
from oneuniverse.simulation.converter import detect_converter, get_converter
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.specs import CosmologySpec, Cube

COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)


def test_generate_and_convert_linear(tmp_path):
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=128.0, n_grid=32, redshifts=(1.0, 0.0), seed=0)
    assert detect_converter(src).code == "linear"; st = get_converter("linear")().convert(src, tmp_path / "store", plan=ExecutionPlan())
    assert set(st.products) >= {"particles", "field", "halos", "lightcone"} and st.manifest.cosmology == COSMO and st.manifest.product("lightcone").extra.get("physical") is False


def test_packed_converter_reads_identically(tmp_path):
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=128.0, n_grid=32, redshifts=(0.0,), seed=1); packed = write_packed_native(src, tmp_path / "packed", n_side=4)
    a = get_converter("linear")().convert(src, tmp_path / "a", plan=ExecutionPlan(), n_side=4); b = get_converter("packed_npy")().convert(packed, tmp_path / "b", plan=ExecutionPlan(), projection="reference", n_side=4)
    cube = Cube((10, 10, 10), (50, 50, 50)); assert a.read_box("particles", 0.0, cube).sort_by("x").equals(b.read_box("particles", 0.0, cube).sort_by("x"))


def test_detection_is_exclusive(tmp_path):
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=2)
    packed = write_packed_native(src, tmp_path / "packed", n_side=2)
    assert json.loads((packed / "header.json").read_text())["layout"] == "packed_npy"
    assert detect_converter(src).code == "linear" and detect_converter(packed).code == "packed_npy"
    fx = make_native_sim(tmp_path / "fx", n_particles=1000, n_grid=8, redshifts=(0.0,))
    fxp = write_packed_native_fixture(fx, n_side=2)
    assert detect_converter(fx).code == "fixture" and detect_converter(fxp).code == "fixture"
    assert detect_converter(tmp_path) is None


def test_arguments_validated_before_writing(tmp_path):
    with pytest.raises(ValidationError):
        generate_linear_sim(tmp_path / "bad", COSMO, box=64.0, n_grid=16, redshifts=(), seed=0)
    assert not (tmp_path / "bad").exists()
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=0)
    with pytest.raises(ValidationError):
        write_packed_native(src, tmp_path / "p", n_side=0)
    assert not (tmp_path / "p").exists()
    with pytest.raises(ValidationError):
        write_packed_native(tmp_path, tmp_path / "p", n_side=2)
