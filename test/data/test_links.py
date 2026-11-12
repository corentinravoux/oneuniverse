import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.synthetic import synthetic_point_table

from oneuniverse._core.errors import ValidationError
from oneuniverse.data.crossmatch import CrossMatchRules
from oneuniverse.data.database import Database
from oneuniverse.data.links import LINKS_SCHEMA, SubobjectRules, load_links
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import write_dataset

_ARC = 1.0 / 3600.0


def _write(root, name, kind, ra, dec, z, seed):
    n = len(ra)
    t = synthetic_point_table(n, seed, survey=name)
    src = t["source_row"].to_numpy()
    for c, v in (("ra", ra), ("dec", dec), ("z", z)):
        t = t.set_column(t.schema.get_field_index(c), c, pa.array(v[src]))
    write_dataset(t, root / name, geometry=Geometry.POINT, survey=SurveyMeta(name, kind),
                  loader=LoaderInfo("test", "1", {}), partition_nside=4)


def _make(root):
    """2000 hosts (1980 + 20 twins 0.6" north of hosts 0..19), 500 sne, 10 clusters."""
    rng = np.random.default_rng(5)
    n0 = 1980
    ra = rng.uniform(0, 360, n0)
    dec = np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, n0)))
    z = rng.uniform(0.01, 0.3, n0)
    hra = np.concatenate([ra, ra[:20]])
    hdec = np.concatenate([dec, dec[:20] + 0.6 * _ARC])
    hz = np.concatenate([z, z[:20]])
    # sne 0..479 sit 0.2" from hosts 20..499; sne 480..499 midway between twin hosts
    sra = np.concatenate([ra[20:500], ra[:20]])
    sdec = np.concatenate([dec[20:500] + 0.2 * _ARC, dec[:20] + 0.3 * _ARC])
    sz = np.concatenate([z[20:500], z[:20]])
    _write(root, "hosts", "galaxy", hra, hdec, hz, 1)
    _write(root, "sne", "snia", sra, sdec, sz, 2)
    _write(root, "clusters", "cluster", ra[20:30], dec[20:30] + 0.1 * _ARC, z[20:30], 3)
    return Database(root), sra, sdec


@pytest.fixture
def db_hosts(tmp_path):
    db, _, _ = _make(tmp_path)
    # identity index with a tiny tolerance: every row is its own object
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(sky_tol_arcsec=0.01, dz_tol=None))
    return db


def _sne_oneuids(db):
    t = db.load_oneuid().table
    return set(t.filter(pc.equal(t["dataset"].cast(pa.string()), "sne"))["oneuid"].to_pylist())


def test_build_links_unambiguous(db_hosts):
    links = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules(
        "galaxy", "snia", sky_tol_arcsec=1.0, dz_tol=None), name="host")
    assert links.table.num_rows == 480 and links.table.schema.names == list(LINKS_SCHEMA.names)
    assert db_hosts.list_links() == ["host"]
    child = links.table["child_oneuid"][0].as_py()
    assert isinstance(links.parent_of(child), int)
    m = links.meta
    assert m["n_links"] == 480 and m["n_ambiguous_children"] == 20
    assert m["oneuid_name"] == "default" and m["oneuid_version"] == db_hosts.load_oneuid().meta["version"]
    assert m["parents"] == ["hosts"] and m["children"] == ["sne"] and m["rules_hash"]
    assert links.rules.child_kind == "snia"
    assert load_links(db_hosts, "host").table.equals(links.table)
    t = links.table
    assert np.all(np.diff(t["child_oneuid"].to_numpy()) >= 0)
    assert np.all(t["confidence"].to_numpy() == 1.0) and np.all(t["sep_arcsec"].to_numpy() < 1.0)
    assert np.all(t["dz"].to_numpy() == 0.0)


def test_build_links_ambiguous_policy(db_hosts):
    l2 = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules(
        "galaxy", "snia", dz_tol=None, accept_ambiguous=True), name="amb")
    assert l2.table.num_rows == 520
    amb = l2.table.filter(pc.field("confidence") < 1)
    assert np.isclose(amb["confidence"].to_numpy(), 0.5).all()
    amb_child = amb["child_oneuid"][0].as_py()
    assert len(l2.parent_of(amb_child)) == 2
    assert l2.parent_of(-5) is None
    assert l2.children_of(l2.parent_of(amb_child)[0]).dtype == np.int64


def test_dz_cut(db_hosts):
    # sne share their host's z exactly: a dz cut keeps everything
    l = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules(
        "galaxy", "snia", dz_tol=1e-3), name="z")
    assert l.table.num_rows == 480


def test_kind_guard(db_hosts):
    with pytest.raises(ValidationError, match="kind"):
        db_hosts.build_links(parents=["sne"], children=["hosts"], rules=SubobjectRules("galaxy", "snia"), name="bad")


def test_dataset_not_in_index(db_hosts):
    db_hosts.build_oneuid(["hosts"], rules=CrossMatchRules(sky_tol_arcsec=0.01), name="only")
    with pytest.raises(ValidationError, match="sne"):
        db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules("galaxy", "snia"),
                             name="x", oneuid_name="only")


def test_children_of_order(db_hosts):
    l = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules(
        "galaxy", "snia", dz_tol=None, accept_ambiguous=True), name="amb")
    t = l.table
    parent = t["parent_oneuid"][0].as_py()
    ch = l.children_of(parent)
    assert len(ch) >= 1 and np.all(np.diff(ch) > 0)
    assert len(l.children_of(10**9)) == 0


def test_chain_walk(db_hosts):
    db = db_hosts
    db.build_links(parents=["clusters"], children=["hosts"], rules=SubobjectRules(
        "cluster", "galaxy", dz_tol=None), name="members")
    db.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules(
        "galaxy", "snia", dz_tol=None), name="host")
    idx = db.load_oneuid().table
    cl = idx.filter(pc.equal(idx["dataset"].cast(pa.string()), "clusters"))["oneuid"].to_numpy()
    leaves = db.chain([int(cl[0])], ["members", "host"])
    assert len(leaves) == 1 and set(leaves.tolist()) <= _sne_oneuids(db)
    all_leaves = db.chain(cl, ["members", "host"])
    assert len(all_leaves) == 10 and leaves.dtype == np.int64
    assert np.array_equal(all_leaves, np.unique(all_leaves))
    assert db.chain([3, 1], []).tolist() == [3, 1]


def test_chain_mixed_oneuid_rejected(db_hosts):
    db = db_hosts
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(sky_tol_arcsec=0.01), name="other")
    db.build_links(parents=["clusters"], children=["hosts"], rules=SubobjectRules("cluster", "galaxy"),
                   name="members")
    db.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules("galaxy", "snia"),
                   name="host2", oneuid_name="other")
    with pytest.raises(ValidationError, match="ONEUID"):
        db.chain([0], ["members", "host2"])


def test_rules_validation():
    with pytest.raises(ValidationError):
        SubobjectRules("a", "b", relation="bogus")
    assert SubobjectRules("a", "b").hash() != SubobjectRules("a", "b", accept_ambiguous=True).hash()
