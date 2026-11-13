import json

import pytest

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.manifest import (
    OUFSIM_FORMAT_VERSION, OUFSimManifest, read_manifest, write_manifest)
from oneuniverse.simulation.specs import (
    CosmologySpec, ProductDecl, ProvenanceSpec, RegionSpec, SimulationRequest, UnitFrameSpec)


def _manifest(**kw):
    return OUFSimManifest(
        OUFSIM_FORMAT_VERSION, "sim", 100.0, CosmologySpec(0.67, 0.31, 0.049, 0.81, 0.965),
        UnitFrameSpec(), ProvenanceSpec("fixture", "1", 0, "abc"),
        (ProductDecl("particles", "particles", (1.0, 0.0), ("x", "y", "z")),),
        {"format": "npy", "root": ".", "projection": "reencode"}, "2026-11-12T00:00:00Z", **kw)


def test_manifest_round_trip_and_version_gate(tmp_path):
    m = _manifest()
    write_manifest(tmp_path / "manifest.json", m)
    assert read_manifest(tmp_path / "manifest.json") == m
    d = m.to_dict()
    d["format_version"] = "0.1.0"
    (tmp_path / "m.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="0.1.0"):
        read_manifest(tmp_path / "m.json")
    with pytest.raises(ValueError):
        ProductDecl("x", "weird", (0.0,))


def test_manifest_missing_key_and_product_lookup(tmp_path):
    m = _manifest(parent="p", lineage={"request_id": "r1",
                                       "region": RegionSpec(Cube((0, 0, 0), (1, 1, 1))).to_dict()})
    write_manifest(tmp_path / "a.json", m)
    assert read_manifest(tmp_path / "a.json") == m
    assert m.product("particles").kind == "particles"
    with pytest.raises(NotFoundError):
        m.product("nope")
    d = m.to_dict()
    del d["box_size"]
    (tmp_path / "b.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="box_size"):
        read_manifest(tmp_path / "b.json")
    d = m.to_dict()
    del d["cosmology"]["sigma8"]
    (tmp_path / "c.json").write_text(json.dumps(d))
    with pytest.raises(FormatError, match="sigma8"):
        read_manifest(tmp_path / "c.json")


def test_region_outer_and_request_validation():
    r = RegionSpec(Cube((10, 10, 10), (20, 20, 20)), buffer=5.0)
    assert r.outer == Cube((5, 5, 5), (25, 25, 25))
    with pytest.raises(ValueError):
        SimulationRequest("r1", "sim", r, physics="mhd")
    with pytest.raises(ValidationError):
        SimulationRequest("../r", "sim", r)
    with pytest.raises(ValidationError):
        RegionSpec(Cube((0, 0, 0), (1, 1, 1)), buffer=-1.0)
    req = SimulationRequest("r1", "sim", RegionSpec(Cube((0, 0, 0), (1, 1, 1)), 0.5, 16),
                            status="failed", error="boom")
    assert SimulationRequest.from_dict(json.loads(json.dumps(req.to_dict()))) == req


def test_spec_validation_and_round_trips():
    for bad in (dict(h=0), dict(omega_b=0.4), dict(sigma8=-1), dict(n_s=0)):
        kw = dict(h=0.67, omega_m=0.31, omega_b=0.049, sigma8=0.81, n_s=0.965) | bad
        with pytest.raises(ValidationError):
            CosmologySpec(**kw)
    c = CosmologySpec(0.67, 0.31, 0.049, 0.81, 0.965, w0=-0.9, wa=0.1)
    assert CosmologySpec.from_dict(c.to_dict()) == c
    for z in ((), (float("nan"),), (-1.0,)):
        with pytest.raises(ValidationError):
            ProductDecl("x", "field", z)
    with pytest.raises(ValidationError):
        ProductDecl("x", "field", (0.0,), n_grid=0)
    p = ProductDecl("f", "field", [0, 1], n_grid=8, extra={"a": 1})
    assert p.redshifts == (0.0, 1.0)
    assert ProductDecl.from_dict(p.to_dict()) == p
    u, pr = UnitFrameSpec(), ProvenanceSpec("c", "1", None, "h", {"k": [1]})
    assert UnitFrameSpec.from_dict(u.to_dict()) == u
    assert ProvenanceSpec.from_dict(pr.to_dict()) == pr


def test_execution_plan_batch_and_mpi_guard():
    assert ExecutionPlan(memory_budget_bytes=2**20).batch_for(100) == 2**20 // 100 // 2
    assert ExecutionPlan(memory_budget_bytes=10).batch_for(100) == 1
    with pytest.raises(ValueError):
        ExecutionPlan(mode=ExecutionMode.MPI)
    ExecutionPlan(mode=ExecutionMode.MPI, mpi_comm=object())
    with pytest.raises(ValidationError):
        ExecutionPlan(n_threads=0)
    with pytest.raises(ValidationError):
        ExecutionPlan(memory_budget_bytes=0)
    assert not hasattr(ExecutionMode, "GPU")
