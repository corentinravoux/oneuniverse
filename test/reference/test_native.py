"""Linear fixture simulation, packed layout and the two converters."""
import json

import pytest
from fixtures.cosmology import COSMO
from fixtures.native_sim import make_native_sim, write_packed_native_fixture

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.native import generate_linear_sim, write_packed_native
from oneuniverse.simulation.converter import detect_converter, get_converter
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.specs import Cube


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


@pytest.mark.parametrize("content", [b"[1, 2]", b"\xff\xfe\x00binary", b'"text"', b"null"])
def test_detection_ignores_non_object_and_binary_config(tmp_path, content):
    d = tmp_path / "weird"; d.mkdir(); (d / "config.json").write_bytes(content)
    assert detect_converter(d) is None
    (d / "header.json").write_bytes(content)
    assert detect_converter(d) is None


def test_generate_and_pack_are_atomic_and_refuse_overwrite(tmp_path, monkeypatch):
    from oneuniverse.reference import native
    monkeypatch.setattr(native, "find_peaks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        generate_linear_sim(tmp_path / "lin", COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=0)
    assert not (tmp_path / "lin").exists() and not (tmp_path / "lin.tmp").exists()
    monkeypatch.undo()
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=0)
    with pytest.raises(ValidationError):
        generate_linear_sim(src, COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=0)
    before = sorted(p.name for p in src.iterdir())
    with pytest.raises(ValidationError):
        write_packed_native(src, src, n_side=2)
    with pytest.raises(ValidationError):
        write_packed_native(src, tmp_path / "lin" / ".." / "lin", n_side=2)
    assert sorted(p.name for p in src.iterdir()) == before and not (tmp_path / "lin.tmp").exists()
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValidationError):
        write_packed_native(src, tmp_path / "taken", n_side=2)
    monkeypatch.setattr(native, "chunk_ids", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        write_packed_native(src, tmp_path / "packed", n_side=2)
    assert not (tmp_path / "packed").exists() and not (tmp_path / "packed.tmp").exists()


@pytest.mark.parametrize("edit", [
    lambda c: c.pop("box_size"), lambda c: c.pop("cosmology"), lambda c: c.pop("seed"),
    lambda c: c.update(n_grid="16"), lambda c: c.update(n_grid=None), lambda c: c.update(redshifts=[]),
    lambda c: c.update(redshifts="0.0"), lambda c: c.update(box_size="big"), lambda c: c.update(cosmology=[1]),
])
def test_malformed_config_is_format_error(tmp_path, edit):
    from oneuniverse._core.errors import FormatError
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=64.0, n_grid=16, redshifts=(0.0,), seed=0)
    cfg = json.loads((src / "config.json").read_text()); edit(cfg); (src / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(FormatError):
        write_packed_native(src, tmp_path / "p", n_side=2)
    with pytest.raises(FormatError):
        get_converter("linear")().manifest(src)
    assert not (tmp_path / "p").exists()
