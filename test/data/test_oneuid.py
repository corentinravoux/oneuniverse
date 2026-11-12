import time
from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.synthetic import synthetic_point_table

from oneuniverse._core.errors import FormatError
from oneuniverse._core.geometry import Cone
from oneuniverse.data.crossmatch import CrossMatchRules
from oneuniverse.data.database import Database
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.oneuid import ONEUID_SCHEMA
from oneuniverse.data.schema import Geometry, stable_object_id
from oneuniverse.data.writer import write_dataset


def _write(root, name, table, **kw):
    return write_dataset(table, root / name, geometry=Geometry.POINT,
                         survey=SurveyMeta(name, "test"), loader=LoaderInfo("test", "1", {}),
                         partition_nside=4, extra_stats_columns=["mag"], **kw)


def _pair_tables(n, n_common, seed=0):
    """a and b with n rows each; source rows < n_common are the same objects (jittered)."""
    kw = {"dec_range": (-80.0, 80.0)}
    ta = synthetic_point_table(n, seed, survey="a", **kw)
    tb = synthetic_point_table(n, seed + 1, survey="b", **kw)
    pos_a = np.argsort(ta["source_row"].to_numpy())  # position of source row k
    src_b = tb["source_row"].to_numpy()
    common = src_b < n_common
    rng = np.random.default_rng(seed + 2)
    cols = {c: tb[c].to_numpy().copy() for c in ("ra", "dec", "z")}
    ia = pos_a[src_b[common]]
    k = int(common.sum())
    cols["ra"][common] = ta["ra"].to_numpy()[ia] + rng.uniform(-0.2, 0.2, k) / 3600 / np.cos(
        np.radians(ta["dec"].to_numpy()[ia]))
    cols["dec"][common] = ta["dec"].to_numpy()[ia] + rng.uniform(-0.2, 0.2, k) / 3600
    cols["z"][common] = ta["z"].to_numpy()[ia] + rng.uniform(-1e-4, 1e-4, k)
    for c, v in cols.items():
        tb = tb.set_column(tb.schema.get_field_index(c), c, pa.array(v))
    return ta, tb


def _make_db(root, n=30_000, n_common=10_000):
    ta, tb = _pair_tables(n, n_common)
    _write(root, "a", ta)
    _write(root, "b", tb)
    return Database(root)


@pytest.fixture(scope="module")
def db2(tmp_path_factory):
    return _make_db(tmp_path_factory.mktemp("db2"))


def test_build_and_load(db2):
    idx = db2.build_oneuid(["a", "b"], rules=CrossMatchRules(sky_tol_arcsec=1.0), name="ab")
    assert idx.n_objects == 50_000 and idx.n_multi == 10_000 and idx.datasets == ["a", "b"]
    assert idx.table.schema.names == list(ONEUID_SCHEMA.names)
    back = db2.load_oneuid("ab")
    assert back.table.equals(idx.table) and back.rules == idx.rules
    assert "ab" in db2.list_oneuids()


def test_oneuid_row_identity_regression(db2):
    """Hydrated rows must be the objects the index points at (the OUF 2.x bug)."""
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    q = db2.oneuid_query("ab")
    sample = q.index.table.slice(0, 2_000)
    sample = sample.set_column(1, "dataset", sample["dataset"].cast(pa.string()))
    hyd = q.hydrate_table(sample["oneuid"].to_numpy(), columns=["object_id", "source_row", "mag"])
    for ds_name in ["a", "b"]:
        part = hyd.filter(pc.field("dataset") == ds_name)
        src = sample.filter(pc.field("dataset") == ds_name).select(
            ["oneuid", "source_row"])
        joined = part.join(src, keys=["oneuid"], right_suffix="_idx")
        assert joined.num_rows == src.num_rows
        assert np.array_equal(joined["source_row"].to_numpy(),
                              joined["source_row_idx"].to_numpy())
        assert np.array_equal(joined["object_id"].to_numpy(),
                              stable_object_id(ds_name, joined["source_row"].to_numpy()))


def test_select_by_geometry(db2):
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    q = db2.oneuid_query("ab")
    cone = Cone(185, 15, 5)
    ids = q.select(cone=cone)
    rows = q.rows(ids)
    assert len(ids) > 0
    inside = cone.contains(rows["ra"].to_numpy(), rows["dec"].to_numpy())
    # every selected object has at least one member in the cone
    assert set(np.unique(rows["oneuid"].to_numpy()[inside])) == set(ids.tolist())
    assert len(q.select(cone=cone, datasets=["a"])) <= len(ids)
    zs = q.select(z=(0.5, 0.6))
    zr = q.rows(zs)["z"].to_numpy()
    assert len(zs) > 0 and ((zr >= 0.5) & (zr <= 0.6)).any()


def test_hydrate_streams_and_merges(db2):
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    q = db2.oneuid_query("ab")
    ids = q.index.multi_only()["oneuid"].unique().to_numpy()[:5_000]
    batches = list(q.hydrate(ids, columns=["z"], batch_size=1_000))
    assert len(batches) >= 10 and all(b.num_rows <= 1_000 for b in batches)
    t = pa.concat_tables(batches)
    assert t.num_rows == 10_000 and set(t.column_names) == {"oneuid", "dataset", "z"}


def test_of_and_unknown_ids(db2):
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    q = db2.oneuid_query("ab")
    multi = int(q.index.multi_only()["oneuid"][0].as_py())
    assert q.index.of(multi).num_rows == 2
    assert q.rows([-5, 10**12]).num_rows == 0
    assert q.hydrate_table([-5], columns=["z"]).num_rows == 0


def test_restrict_to_keeps_ids(db2):
    idx = db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    r = idx.restrict_to(["a"])
    assert r.datasets == ["a"] and r.n_objects == 30_000
    assert np.isin(r.table["oneuid"].to_numpy(), idx.table["oneuid"].to_numpy()).all()


def test_rebuild_archives_and_as_of(tmp_path):
    db = _make_db(tmp_path, 2_000, 500)
    db.build_oneuid(["a"], rules=CrossMatchRules(), name="x")
    t0 = datetime.now(UTC)
    time.sleep(0.01)
    db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="x")
    assert db.load_oneuid("x").datasets == ["a", "b"]
    assert db.load_oneuid("x", as_of=t0).datasets == ["a"]


def test_hydrate_refuses_changed_dataset(tmp_path):
    db = _make_db(tmp_path, 2_000, 500)
    db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    ta, _ = _pair_tables(2_000, 500, seed=7)
    _write(tmp_path, "a", ta, overwrite=True)
    q = Database(tmp_path).oneuid_query("ab")
    ids = q.index.table["oneuid"].to_numpy()[:10]
    with pytest.raises(FormatError, match="dataset a changed since ONEUID ab was built"):
        q.hydrate_table(ids, columns=["z"])
    db = Database(tmp_path)
    db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    q = db.oneuid_query("ab")
    assert q.hydrate_table(ids, columns=["z"]).num_rows == len(
        q.rows(ids))


@pytest.mark.slow
def test_crossmatch_two_million_rows_under_a_minute(tmp_path):
    n = 1_000_000
    ta, tb = _pair_tables(n, n // 5)
    _write(tmp_path, "a", ta)
    _write(tmp_path, "b", tb)
    db = Database(tmp_path)
    t = time.perf_counter()
    idx = db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="big")
    dt = time.perf_counter() - t
    print(f"two-million build: {dt:.1f} s")
    assert idx.n_multi == n // 5
    assert dt < 60
