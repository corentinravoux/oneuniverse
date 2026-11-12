import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.synthetic import synthetic_point_table

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse.data.crossmatch import CrossMatchRules
from oneuniverse.data.database import Database
from oneuniverse.data.links import LINKS_SCHEMA, SubobjectRules, load_links
from oneuniverse.data.manifest import LoaderInfo, SurveyMeta
from oneuniverse.data.schema import Geometry
from oneuniverse.data.writer import write_dataset

_ARC = 1.0 / 3600.0


def _write(root, name, kind, ra, dec, z, seed, **kw):
    n = len(ra)
    t = synthetic_point_table(n, seed, survey=name)
    src = t["source_row"].to_numpy()
    for c, v in (("ra", ra), ("dec", dec), ("z", z)):
        t = t.set_column(t.schema.get_field_index(c), c, pa.array(v[src]))
    write_dataset(t, root / name, geometry=Geometry.POINT, survey=SurveyMeta(name, kind),
                  loader=LoaderInfo("test", "1", {}), partition_nside=4, **kw)


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


def test_kinds_are_labels_recorded_in_meta(db_hosts):
    # kinds are descriptive: the dataset lists are authoritative, nothing is checked
    l = db_hosts.build_links(parents=["sne"], children=["hosts"], rules=SubobjectRules(
        "galaxy", "snia", dz_tol=None), name="swapped")
    assert l.meta["rules"]["parent_kind"] == "galaxy" and l.meta["rules"]["child_kind"] == "snia"
    assert l.meta["parents"] == ["sne"] and l.meta["children"] == ["hosts"]


def test_strings_rejected_as_name_lists(db_hosts):
    r = SubobjectRules("galaxy", "snia", dz_tol=None)
    with pytest.raises(ValidationError, match="list of names"):
        db_hosts.build_links(parents="hosts", children=["sne"], rules=r)
    with pytest.raises(ValidationError, match="list of names"):
        db_hosts.build_links(parents=["hosts"], children="sne", rules=r)
    with pytest.raises(ValidationError, match="list of names"):
        db_hosts.chain([0], "host")


def test_links_refuse_stale_dataset(tmp_path):
    db, _, _ = _make(tmp_path)
    db.build_oneuid(["hosts", "sne"], rules=CrossMatchRules(sky_tol_arcsec=0.01, dz_tol=None))
    rng = np.random.default_rng(1)
    n = 50
    _write(tmp_path, "sne", "snia", rng.uniform(0, 360, n), rng.uniform(-10, 10, n),
           rng.uniform(0.1, 0.2, n), 9, overwrite=True)
    with pytest.raises(FormatError, match="dataset sne changed since ONEUID default"):
        Database(tmp_path).build_links(parents=["hosts"], children=["sne"],
                                       rules=SubobjectRules("galaxy", "snia", dz_tol=None))


def test_pinned_database_is_read_only(db_hosts):
    from datetime import UTC, datetime
    past = db_hosts.as_of(datetime.now(UTC))
    with pytest.raises(ValidationError, match="read-only"):
        past.build_links(parents=["hosts"], children=["sne"],
                         rules=SubobjectRules("galaxy", "snia", dz_tol=None))
    with pytest.raises(ValidationError, match="read-only"):
        past.build_oneuid(["hosts"], rules=CrossMatchRules())


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


def test_stale_oneuid_rejected_and_as_of(db_hosts):
    from datetime import UTC, datetime
    db = db_hosts
    rules = SubobjectRules("galaxy", "snia", dz_tol=None)
    db.build_links(parents=["hosts"], children=["sne"], rules=rules, name="host")
    cl = db.build_links(parents=["clusters"], children=["hosts"], rules=SubobjectRules(
        "cluster", "galaxy", dz_tol=None), name="members")
    before = datetime.now(UTC)
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(sky_tol_arcsec=0.01, dz_tol=None))
    with pytest.raises(FormatError, match="rebuild the links"):
        db.load_links("host")
    with pytest.raises(FormatError, match="rebuild the links"):
        db.chain([0], ["members", "host"])
    assert db.load_links("host", as_of=before).table.num_rows == 480
    assert len(db.chain([0], ["members", "host"], as_of=before)) <= 1
    assert db.as_of(before).load_links("members").table.equals(cl.table)


def test_chain_version_mismatch(db_hosts):
    db = db_hosts
    db.build_links(parents=["clusters"], children=["hosts"], rules=SubobjectRules(
        "cluster", "galaxy", dz_tol=None), name="members")
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(sky_tol_arcsec=0.01, dz_tol=None),
                    name="other")
    db.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules("galaxy", "snia", dz_tol=None),
                   name="h2", oneuid_name="other")
    with pytest.raises(ValidationError, match="not comparable"):
        db.chain([0], ["members", "h2"])


def _mini(root):
    """p1/p2 and c1/c2 hold two member rows of one parent / one child object (3" apart)."""
    s = 1.0 / 3600
    def put(name, kind, ra, dec, z, seed):
        _write(root, name, kind, np.array(ra, float), np.array(dec, float), np.array(z, float), seed)
    # object P at dec 0 (p1 0, p2 0.2"), child C 3" away (c1 3.0, c2 3.2"); Q/D at ra 20
    put("p1", "galaxy", [10, 20], [0, 0], [0.1, 0.2], 1)
    put("p2", "galaxy", [10, 50], [0.2 * s, 0], [0.1, 0.3], 2)
    put("c1", "snia", [10, 20], [3.0 * s, 2.0 * s], [0.1, np.nan], 3)
    put("c2", "snia", [10, 80], [3.2 * s, 0], [0.1, 0.4], 4)
    return Database(root)


def test_multi_member_objects(tmp_path):
    db = _mini(tmp_path)
    ds = ["p1", "p2", "c1", "c2"]
    db.build_oneuid(ds, rules=CrossMatchRules(sky_tol_arcsec=0.5, dz_tol=None))
    l = db.build_links(parents=["p1", "p2"], children=["c1", "c2"], rules=SubobjectRules(
        "galaxy", "snia", sky_tol_arcsec=5.0, dz_tol=None), name="m")
    t = l.table
    assert t.num_rows == 2 and l.meta["n_self_merged"] == 0 and l.meta["n_ambiguous_children"] == 0
    seps = np.sort(t["sep_arcsec"].to_numpy())
    assert np.isclose(seps[0], 2.0, atol=0.01) and np.isclose(seps[1], 2.8, atol=0.01)
    dz = t["dz"].to_numpy()
    assert np.isnan(dz).sum() == 1 and np.isclose(np.nanmax(dz), 0.0)
    assert (t["confidence"].to_numpy() == 1.0).all()


def test_self_merged_raises_unless_allowed(db_hosts):
    db = db_hosts
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(sky_tol_arcsec=1.0, dz_tol=None))
    rules = SubobjectRules("galaxy", "snia", dz_tol=None)
    with pytest.raises(ValidationError, match="separate=") as ei:
        db.build_links(parents=["hosts"], children=["sne"], rules=rules, name="h")
    assert "allow_self_merged" in str(ei.value)
    l = db.build_links(parents=["hosts"], children=["sne"], rules=rules, name="h",
                       allow_self_merged=True)
    # 480 sne merged with their host, 20 with one twin: 500 self objects; the other twin stays a parent
    assert l.meta["n_self_merged"] == 500 and l.meta["n_links"] == 20
    assert l.meta["allow_self_merged"] is True
    with pytest.raises(ValidationError, match="merge parents with children"):
        db.build_links(parents=["clusters"], children=["hosts"], rules=SubobjectRules(
            "cluster", "galaxy", dz_tol=None), name="c", allow_self_merged=True)


def test_separate_gives_full_links(db_hosts):
    db = db_hosts
    db.build_oneuid(["hosts", "sne", "clusters"], rules=CrossMatchRules(
        sky_tol_arcsec=1.0, dz_tol=None, separate=(("hosts", "sne"),)))
    idx = db.load_oneuid().table
    ds = idx["dataset"].cast(pa.string()).to_numpy(zero_copy_only=False)
    oid = idx["oneuid"].to_numpy()
    assert not set(oid[ds == "hosts"]) & set(oid[ds == "sne"])
    rules = SubobjectRules("galaxy", "snia", dz_tol=None)
    l = db.build_links(parents=["hosts"], children=["sne"], rules=rules, name="h")
    assert l.meta["n_self_merged"] == 0 and l.table.num_rows == 480
    l2 = db.build_links(parents=["hosts"], children=["sne"], name="h2", rules=SubobjectRules(
        "galaxy", "snia", dz_tol=None, accept_ambiguous=True))
    assert l2.table.num_rows == 520


def test_separate_blocks_transitive_merge(tmp_path):
    s = 1.0 / 3600
    for k, (nm, dec) in enumerate((("x", 0.0), ("y", 0.5 * s), ("z", 1.0 * s))):
        _write(tmp_path, nm, "galaxy", np.array([10.0]), np.array([dec]), np.array([0.1]), k + 1)
    db = Database(tmp_path)
    r = CrossMatchRules(sky_tol_arcsec=0.6, dz_tol=None)
    def ids(rules):
        t = db.build_oneuid(["x", "y", "z"], rules=rules).table
        ds = t["dataset"].cast(pa.string()).to_pylist()
        return dict(zip(ds, t["oneuid"].to_pylist()))
    m = ids(r)
    assert m["x"] == m["y"] == m["z"]
    m = ids(CrossMatchRules(sky_tol_arcsec=0.6, dz_tol=None, separate=(("z", "x"),)))
    assert m["x"] != m["z"] and len({m["x"], m["y"], m["z"]}) == 2
    with pytest.raises(ValidationError, match="separate"):
        db.build_oneuid(["x", "y"], rules=CrossMatchRules(separate=(("x", "q"),)))
