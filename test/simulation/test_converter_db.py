import shutil

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

_PARENT_TEMPLATE: list = []


@pytest.fixture(scope="module", autouse=True)
def _parent_store_template(tmp_path_factory):
    """Convert the fixture simulation once; each test copies the resulting store."""
    root = tmp_path_factory.mktemp("parent_template")
    src = make_native_sim(root / "sim")
    get_converter("fixture")().convert(src, root / "parent", plan=ExecutionPlan(), n_side=4)
    _PARENT_TEMPLATE.append(root / "parent")
    yield
    _PARENT_TEMPLATE.clear()


def _db_with_request(tmp_path, region=None):
    shutil.copytree(_PARENT_TEMPLATE[0], tmp_path / "db/parent")
    db = SimDatabase(tmp_path / "db").scan()
    req = db.request_region("parent", region or RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0))
    return db, req


def test_converter_registry_and_convert(tmp_path):
    src = make_native_sim(tmp_path / "sim")
    assert detect_converter(src).code == "fixture" and "fixture" in registered_codes()
    assert detect_converter(tmp_path) is None
    st = get_converter("fixture")().convert(src, tmp_path / "store", plan=ExecutionPlan(), n_side=4)
    assert isinstance(st, SimStore) and st.manifest.provenance.code == "fixture"
    with pytest.raises(NotFoundError):
        get_converter("gadget")


def test_simdatabase_scan_request_dispatch(tmp_path):
    shutil.copytree(_PARENT_TEMPLATE[0], tmp_path / "db/parent")
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


def _engine(request, store):
    sub = store.read_field_box("field", 0.0, request.region.outer)
    return sub * 0 + 1.0, 0.0


def test_crash_after_publish_is_reconciled(tmp_path, monkeypatch):
    db, req = _db_with_request(tmp_path)
    orig = SimDatabase._set

    def crashy(self, request, **changes):
        if changes.get("status") == "ingested":
            raise RuntimeError("killed")
        return orig(self, request, **changes)

    monkeypatch.setattr(SimDatabase, "_set", crashy)
    with pytest.raises(RuntimeError, match="killed"):
        db.dispatch(req, _engine)
    monkeypatch.undo()
    db2 = SimDatabase(tmp_path / "db").scan()
    r = db2.requests()[0]
    assert r.status == "ingested" and r.child == f"parent__{req.id}"
    assert db2.lineage().num_rows == 1 and db2.lineage().column("request_id")[0].as_py() == req.id
    assert SimDatabase(tmp_path / "db").scan().requests()[0].status == "ingested"


def test_running_without_child_reconciled_to_failed(tmp_path, monkeypatch):
    db, req = _db_with_request(tmp_path)
    orig = SimDatabase._set

    def engine(request, store):
        raise KeyboardInterrupt  # stands in for a hard kill: nothing else is persisted

    with pytest.raises(KeyboardInterrupt):
        monkeypatch.setattr(SimDatabase, "_set", lambda self, r, **c: orig(self, r, **c)
                            if c.get("status") == "running" else r)
        db.dispatch(req, engine)
    monkeypatch.undo()
    assert db.requests()[0].status == "running"
    db2 = SimDatabase(tmp_path / "db").scan()
    r = db2.requests()[0]
    assert r.status == "failed" and r.error == "interrupted: no child store found at scan"
    assert db2.lineage().num_rows == 0
    assert SimDatabase(tmp_path / "db").scan().requests()[0].status == "failed"


def test_dispatch_status_guard(tmp_path):
    db, req = _db_with_request(tmp_path)
    db.dispatch(req, _engine)
    with pytest.raises(ValidationError):
        db.dispatch(req, _engine, child_name="second")
    assert db.lineage().num_rows == 1 and list(db.stores()) == ["parent", f"parent__{req.id}"]


def test_failed_request_can_be_redispatched(tmp_path):
    db, req = _db_with_request(tmp_path)

    def bad(request, store):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        db.dispatch(req, bad)
    db.dispatch(req, _engine)
    r = db.requests()[0]
    assert r.status == "ingested" and r.child == f"parent__{req.id}" and db.lineage().num_rows == 1


def test_detect_converter_ambiguous_raises(tmp_path):
    from oneuniverse.simulation import converter as C

    class Twin(C.SimConverter):
        code = "twin"

        @classmethod
        def detect(cls, path):
            return (path / "config.json").is_file()

        def manifest(self, path):  # pragma: no cover
            raise NotImplementedError

        def products(self, path):  # pragma: no cover
            raise NotImplementedError

    src = make_native_sim(tmp_path / "sim", n_particles=100)
    C.register(Twin)
    try:
        with pytest.raises(ValidationError, match="fixture.*twin|twin.*fixture"):
            detect_converter(src)
    finally:
        C._REGISTRY.items_dict.pop("twin")
    assert detect_converter(src).code == "fixture"


def test_fixture_detect_needs_marker(tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other/config.json").write_text('{"box_size": 1}')
    assert detect_converter(tmp_path / "other") is None


def test_convert_forwards_build_kw(tmp_path):
    from fixtures.native_sim import write_packed_native_fixture
    src = make_native_sim(tmp_path / "sim", n_particles=500)
    packed = write_packed_native_fixture(src, n_side=4)
    st = get_converter("fixture")().convert(
        src, tmp_path / "a", plan=ExecutionPlan(), n_side=2, tile_cells=16, nside_part=2)
    assert st.layout["products"]["particles"]["0.000"]["n_side"] == 2
    assert st.layout["products"]["field"]["0.000"]["tile_cells"] == 16
    assert st.layout["products"]["lightcone"]["0.000"]["nside_part"] == 2
    ref = get_converter("fixture")().convert(packed, tmp_path / "b", plan=ExecutionPlan(),
                                             projection="reference")
    assert ref.manifest.native["projection"] == "reference"


@pytest.mark.parametrize("name", ["kid.tmp", "kid.old", "_kid", "a/b", "a\\b", "..", ".", "../x"])
def test_dispatch_rejects_bad_child_name(tmp_path, name):
    db, req = _db_with_request(tmp_path)
    with pytest.raises(ValidationError, match="child store name"):
        db.dispatch(req, _engine, child_name=name)
    assert db.requests()[0].status == "pending"
    assert SimDatabase(tmp_path / "db").scan().requests()[0].status == "pending"


def test_scan_skips_private_and_staging_dirs(tmp_path):
    import shutil
    _db_with_request(tmp_path)
    for n in ("_x", "kid.tmp", "kid.old"):
        shutil.copytree(tmp_path / "db/parent", tmp_path / "db" / n)
    assert list(SimDatabase(tmp_path / "db").scan().stores()) == ["parent"]


def test_request_region_outer_larger_than_parent(tmp_path):
    db, _ = _db_with_request(tmp_path)
    with pytest.raises(ValidationError, match="box_size"):
        db.request_region("parent", RegionSpec(Cube((0, 0, 0), (90, 90, 90)), buffer=10.0))
    db.request_region("parent", RegionSpec(Cube((0, 0, 0), (100, 100, 100))))


def test_child_lineage_records_origin(tmp_path):
    db, req = _db_with_request(tmp_path)
    child = db.dispatch(req, _engine)
    assert child.manifest.lineage["origin"] == [5.0, 5.0, 5.0]


def test_dispatch_checks_shape_without_materialising(tmp_path):
    import numpy as np

    class Lazy:
        shape = (4, 4, 4)
        ndim = 3

        def __array__(self, *a, **k):  # pragma: no cover
            raise AssertionError("materialised")

    db, req = _db_with_request(
        tmp_path, RegionSpec(Cube((10, 10, 10), (40, 40, 40)), buffer=5.0, n_grid=8))
    with pytest.raises(ValidationError):
        db.dispatch(req, lambda r, s: (Lazy(), 0.0))
    assert np  # shape check precedes any conversion
