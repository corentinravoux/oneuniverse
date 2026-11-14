import pytest
from fixtures.native_sim import make_native_sim

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.converter import (
    detect_converter,
    get_converter,
    registered_codes,
)
from oneuniverse.simulation.db import SimDatabase
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.specs import RegionSpec
from oneuniverse.simulation.store.read import SimStore


def _db_with_request(tmp_path, region=None):
    src = make_native_sim(tmp_path / "sim")
    get_converter("fixture")().convert(src, tmp_path / "db/parent", plan=ExecutionPlan())
    db = SimDatabase(tmp_path / "db").scan()
    req = db.request_region("parent", region or RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0))
    return db, req


def test_converter_registry_and_convert(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    assert detect_converter(src).code == "fixture" and "fixture" in registered_codes()
    assert detect_converter(tmp_path) is None
    st = get_converter("fixture")().convert(src, tmp_path / "store", plan=ExecutionPlan())
    assert isinstance(st, SimStore) and st.manifest.provenance.code == "fixture"
    with pytest.raises(NotFoundError):
        get_converter("gadget")


def test_simdatabase_scan_request_dispatch(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    get_converter("fixture")().convert(src, tmp_path / "db/parent", plan=ExecutionPlan())
    db = SimDatabase(tmp_path / "db").scan()
    assert list(db.stores()) == ["parent"]
    req = db.request_region("parent", RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0))
    assert req.status == "pending" and db.requests()[0].id == req.id

    def engine(request, store):  # stand-in engine: constant field over the outer region at z=0
        sub = store.read_field_box("field", 0.0, request.region.outer)
        return sub * 0 + 1.0, 0.0

    child = db.dispatch(req, engine)
    assert (child.manifest.parent == "parent" and child.manifest.lineage["request_id"] == req.id
            and "field" in child.products)
    db2 = SimDatabase(tmp_path / "db").scan()
    assert db2.requests()[0].status == "ingested" and db2.lineage().num_rows == 1
    child_name = db2.requests()[0].child
    assert list(db2.stores()) == ["parent", child_name]
    got = db2.store(child_name).read_field_box("field", 0.0, Cube((0, 0, 0), (10, 10, 10)))
    assert got.mean() == 1.0
    with pytest.raises(NotFoundError):
        db2.store("nope")


def test_dispatch_failure_recorded(tmp_path):
    db, req = _db_with_request(tmp_path)

    def bad(request, store):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        db.dispatch(req, bad)
    assert db.requests()[0].status == "failed" and "boom" in db.requests()[0].error
    assert db.lineage().num_rows == 0
    db2 = SimDatabase(tmp_path / "db").scan()
    assert db2.requests()[0].status == "failed" and list(db2.stores()) == ["parent"]


def test_dispatch_bad_shape_leaves_no_store(tmp_path):
    import numpy as np
    db, req = _db_with_request(
        tmp_path, RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0, n_grid=8))
    with pytest.raises(ValidationError):
        db.dispatch(req, lambda r, s: (np.zeros((4, 4, 4), np.float32), 0.0))
    assert db.requests()[0].status == "failed"
    assert list(SimDatabase(tmp_path / "db").scan().stores()) == ["parent"]


def test_request_region_validation(tmp_path):
    db, _ = _db_with_request(tmp_path)
    with pytest.raises(NotFoundError):
        db.request_region("ghost", RegionSpec(Cube((0, 0, 0), (1, 1, 1))))
    with pytest.raises(ValidationError):
        db.request_region("parent", RegionSpec(Cube((0, 0, 0), (1, 2, 1))))
