# DuckDB 2.0 relations have no fetchone: tests use con.execute(q).fetchone()/fetchall().
import importlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("duckdb")

from fixtures.synthetic import write_synthetic_dataset
from fixtures.two_surveys import pair_tables, write_survey

import oneuniverse.data.sql
from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cone
from oneuniverse.data.database import Database
from oneuniverse.data.links import SubobjectRules, load_links
from oneuniverse.data.oneuid import CrossMatchRules
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.sql import cone_where, connect


@pytest.fixture(scope="module")
def db(two_survey_db_large):
    db = two_survey_db_large; db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); return db


def test_catalog_tables(db):
    con = connect(db.root); assert con.execute("SELECT name, n_rows FROM datasets ORDER BY name").fetchall() == [("a", 12_000), ("b", 12_000)]
    assert con.execute("SELECT count(*) FROM partitions WHERE dataset='a'").fetchone()[0] == db["a"].n_partitions


CONES = [Cone(185.0, 15.0, 5.0), Cone(0.5, 15.0, 5.0), Cone(359.5, 15.0, 5.0),
         Cone(0.0, 89.5, 3.0), Cone(0.0, -89.5, 3.0), Cone(185.0, 15.0, 40.0),
         Cone(185.0, 15.0, 0.01)]


@pytest.mark.parametrize("cone", CONES, ids=lambda c: f"{c.ra}_{c.dec}_{c.radius}")
def test_dataset_view_and_cone_matches_python(db, cone):
    con = connect(db.root); v = db["a"]
    n_sql = con.execute(f"SELECT count(*) FROM a WHERE {cone_where(cone, v.nside)}").fetchone()[0]
    assert n_sql == v.table(cone=cone).num_rows


def test_nz_groupby(db):
    con = connect(db.root)
    rows = con.execute("SELECT round(floor(z*10)/10, 1) AS zb, count(*) FROM a GROUP BY zb ORDER BY zb").fetchall()
    z = db["a"].table(columns=["z"])["z"].to_numpy()
    bins, counts = np.unique(np.round(np.floor(z * 10) / 10, 1), return_counts=True)
    assert [r[0] for r in rows] == bins.tolist() and [r[1] for r in rows] == counts.tolist()
    assert sum(r[1] for r in rows) == 12_000


def test_oneuid_join_counts_multi(db):
    con = connect(db.root); n_multi = con.execute("SELECT count(*) FROM (SELECT oneuid FROM oneuid_ab GROUP BY oneuid HAVING count(*) > 1)").fetchone()[0]
    assert n_multi == db.load_oneuid("ab").n_multi
    joined = con.execute("SELECT count(*) FROM oneuid_ab o JOIN a ON o.dataset='a' AND o.source_row=a.source_row").fetchone()[0]; assert joined == 12_000


def test_oneuid_columns_and_versions(db):
    con = connect(db.root)
    cols = [r[0] for r in con.execute("DESCRIBE oneuid_ab").fetchall()]
    assert cols == ["oneuid", "dataset", "source_row", "partition", "offset", "ra", "dec", "z", "z_type"]
    assert [r[0] for r in con.execute("DESCRIBE datasets").fetchall()] == [
        "name", "geometry", "kind", "survey", "n_rows", "nside", "format_version",
        "created_utc", "version", "valid_from", "valid_to", "path"]
    assert con.execute("SELECT count(*) FROM versions WHERE dataset='a'").fetchone()[0] == 1


def test_cone_where_empty_and_names():
    assert cone_where(Cone(10.0, 10.0, 1.0), 4).startswith('"cell" IN (')
    assert cone_where(Cone(10.0, 10.0, 1.0), 4, cell_column="c", ra_column="r", dec_column="d").count('"c"') == 1


def test_naive_as_of_rejected(db):
    with pytest.raises(ValidationError):
        connect(db.root, as_of=datetime.now())


def test_staging_and_underscore_dirs_hidden(tmp_path):
    write_synthetic_dataset(tmp_path / "p", 300, 0)
    write_synthetic_dataset(tmp_path / "q.tmp", 300, 1)
    write_synthetic_dataset(tmp_path / "_hidden", 300, 2)
    con = connect(tmp_path)
    assert con.execute("SELECT name FROM datasets").fetchall() == [("p",)]


def test_view_name_mapping_and_collision(tmp_path):
    write_synthetic_dataset(tmp_path / "x" / "y-z", 300, 0)
    con = connect(tmp_path)
    assert con.execute("SELECT name FROM datasets").fetchall() == [("x__y-z",)]
    assert con.execute('SELECT count(*) FROM "x_y_z"').fetchone()[0] == 300
    write_synthetic_dataset(tmp_path / "x_y_z", 300, 1)
    with pytest.raises(ValidationError, match="collides"):
        connect(tmp_path)


def test_invalid_and_reserved_view_names(tmp_path):
    write_synthetic_dataset(tmp_path / "datasets", 300, 0)
    with pytest.raises(ValidationError):
        connect(tmp_path)
    write_synthetic_dataset(tmp_path / "1bad", 300, 0, overwrite=False)
    (tmp_path / "datasets").rename(tmp_path / "ok")
    with pytest.raises(ValidationError, match="identifier"):
        connect(tmp_path)


def test_as_of_dataset_rewritten_mirrors_python(tmp_path):
    ta, tb = pair_tables(2000, 500)
    write_survey(tmp_path, "a", ta); write_survey(tmp_path, "b", tb)
    Database(tmp_path).build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    time.sleep(0.01); t1 = datetime.now(UTC); time.sleep(0.01)
    write_survey(tmp_path, "b", tb.slice(0, 1500), overwrite=True)
    con = connect(tmp_path, as_of=t1)
    names = [r[0] for r in con.execute("SELECT name FROM datasets ORDER BY name").fetchall()]
    assert names == Database(tmp_path).as_of(t1).names() == ["a"]
    assert con.execute("SELECT count(*) FROM versions WHERE dataset='b'").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM oneuid_ab").fetchone()[0] == \
        Database(tmp_path).load_oneuid("ab", as_of=t1).table.num_rows
    live = connect(tmp_path)
    assert live.execute("SELECT count(*) FROM b").fetchone()[0] == 1500
    assert live.execute("SELECT count(*) FROM versions WHERE dataset='b'").fetchone()[0] == 2


def test_as_of_inside_live_interval_keeps_dataset(tmp_path):
    t0 = datetime.now(UTC); time.sleep(0.01)
    write_synthetic_dataset(tmp_path / "p", 300, 0)
    assert connect(tmp_path, as_of=t0).execute("SELECT count(*) FROM datasets").fetchone()[0] == 0
    time.sleep(0.01); t1 = datetime.now(UTC)
    con = connect(tmp_path, as_of=t1)
    assert con.execute("SELECT name FROM datasets").fetchall() == [("p",)]
    assert con.execute("SELECT count(*) FROM p").fetchone()[0] == 300


def test_partition_ordinal_joins_partitions_and_offset(db):
    con = connect(db.root); v = db["a"]
    ords = [r[0] for r in con.execute("SELECT partition FROM partitions WHERE dataset='a' ORDER BY partition").fetchall()]
    assert ords == list(range(v.n_partitions))
    paths = [r[0] for r in con.execute("SELECT path FROM partitions WHERE dataset='a' ORDER BY partition").fetchall()]
    assert paths == [str(v.path / p) for p in v.index["path"].to_pylist()]
    assert con.execute('SELECT count(*) FROM oneuid_ab o JOIN partitions p ON p.dataset=o.dataset '
                       'AND p."partition"=o."partition" WHERE o.dataset=\'a\'').fetchone()[0] == 12_000
    rows = con.execute('SELECT "partition", "offset", source_row FROM oneuid_ab WHERE dataset=\'a\' '
                       'ORDER BY source_row LIMIT 500').fetchall()
    got = v.take([r[0] for r in rows], [r[1] for r in rows], columns=["source_row"])["source_row"].to_pylist()
    assert got == [r[2] for r in rows]


def test_links_views_match_python(two_survey_db):
    db = two_survey_db; tmp_path = db.root; db.build_oneuid(["a", "b"], rules=CrossMatchRules(sky_tol_arcsec=0.01, dz_tol=None, separate=(("a", "b"),)), name="ab")
    links = db.build_links(parents=["a"], children=["b"], rules=SubobjectRules(
        "galaxy", "galaxy", sky_tol_arcsec=1.0, dz_tol=None), name="ab", oneuid_name="ab")
    con = connect(tmp_path)
    assert [r[0] for r in con.execute("DESCRIBE links_ab").fetchall()] == [
        "parent_oneuid", "child_oneuid", "sep_arcsec", "dz", "confidence"]
    n = con.execute("SELECT count(*) FROM links_ab").fetchone()[0]
    assert n == links.table.num_rows == load_links(db, "ab").table.num_rows > 0
    assert con.execute("SELECT sum(parent_oneuid), sum(child_oneuid) FROM links_ab").fetchone() == (
        int(links.table["parent_oneuid"].to_numpy().sum()), int(links.table["child_oneuid"].to_numpy().sum()))


def test_persistent_database_drops_stale_objects(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    write_synthetic_dataset(root / "p", 300, 0); write_synthetic_dataset(root / "q", 300, 1)
    Database(root).build_oneuid(["p", "q"], rules=CrossMatchRules(), name="pq")
    f = str(tmp_path / "x.duckdb")
    con = connect(root, database=f)
    assert con.execute("SELECT count(*) FROM oneuid_pq").fetchone()[0] > 0
    con.close()
    import shutil
    shutil.rmtree(root / "q"); shutil.rmtree(root / "_oneuid")
    con = connect(root, database=f)
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert names == {"datasets", "partitions", "versions", "p"}


def test_list_columns_are_native(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 11))); write_synthetic_dataset(tmp_path / "p", 500, 0, with_pdf=spec)
    con = connect(tmp_path); assert con.execute("SELECT len(z_pdf) FROM p LIMIT 1").fetchone()[0] == 11


def test_missing_duckdb_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"oneuniverse\[sql\]"): importlib.reload(oneuniverse.data.sql).connect(Path("."))


def test_as_of_selects_old_sidecar(db):
    t0 = datetime.now(UTC); time.sleep(0.01); db.build_oneuid(["a"], rules=CrossMatchRules(), name="ab")
    assert connect(db.root, as_of=t0).execute("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 2 and connect(db.root).execute("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 1
