# DuckDB 2.0 relations have no fetchone: tests use con.execute(q).fetchone()/fetchall().
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fixtures.native_sim import make_native_sim, native_manifest, native_products

pytest.importorskip("duckdb")

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse._core.geometry import Cube
from oneuniverse.simulation.execution import ExecutionPlan
from oneuniverse.simulation.sql import box_where, connect
from oneuniverse.simulation.store.build import build_store
from oneuniverse.simulation.store.index import overlapping_chunks


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("simsql")
    src = make_native_sim(tmp / "sim", n_particles=50_000)
    st = build_store(tmp / "store", manifest=native_manifest(src),
                     products=native_products(src), plan=ExecutionPlan(), n_side=4)
    return tmp, src, st


def test_chunk_query_matches_store(built):
    tmp, src, st = built
    con = connect(tmp / "store"); cube = Cube((10, 10, 10), (30, 30, 30))
    ids = sorted(r[0] for r in con.execute(f"SELECT chunk_id FROM chunks WHERE product='particles' AND z=0.0 AND {box_where(cube)}").fetchall())
    assert ids == sorted(overlapping_chunks(st.index("particles", 0.0), cube, 100.0).tolist())
    assert con.execute("SELECT count(*) FROM halos_z0_000").fetchone()[0] == pq.read_table(src / "halos_z0.000.parquet").num_rows
    assert con.execute("SELECT box_size, code FROM sims").fetchone() == (100.0, "fixture")


def test_periodic_box_where_matches_store(built):
    tmp, _, st = built
    con = connect(tmp / "store"); cube = Cube((90, 90, 5), (110, 105, 20))
    ids = sorted(r[0] for r in con.execute(
        f"SELECT chunk_id FROM chunks WHERE product='particles' AND z=0.0 AND {box_where(cube, 100.0)}").fetchall())
    assert ids == sorted(overlapping_chunks(st.index("particles", 0.0), cube, 100.0).tolist())


def test_catalog_tables(built):
    tmp, _, _ = built
    con = connect(tmp / "store")
    rows = {(r[0], r[1]): r[2:] for r in con.execute(
        "SELECT name, z, kind, layout, n_chunks FROM products").fetchall()}
    assert rows[("particles", 0.0)] == ("particles", "chunks", 64)
    assert rows[("field", 0.0)][1] == "tiles" and rows[("field", 0.0)][2] >= 1
    assert rows[("lightcone", 0.0)][1] == "sky"
    n_tiles = con.execute("SELECT count(*) FROM tiles WHERE product='field' AND z=0.0").fetchone()[0]
    assert n_tiles == rows[("field", 0.0)][2]
    p = con.execute("SELECT path FROM chunks WHERE path <> '' LIMIT 1").fetchone()[0]
    assert Path(p).is_absolute() and Path(p).exists()
    # no view for field / lightcone products
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"halos_z0_000", "particles_z1_000"} <= names
    assert not any(n.startswith(("field", "lightcone")) for n in names)


def test_database_root_and_errors(built, tmp_path):
    tmp, _, _ = built
    db = tmp_path / "db"
    db.mkdir()
    (db / "a").symlink_to(tmp / "store")
    (db / "_simdb").mkdir()
    (db / "b.tmp").mkdir()
    (db / "b.tmp" / "manifest.json").write_text("{}")
    con = connect(db)
    assert con.execute("SELECT name FROM sims").fetchall() == [("a",)]
    assert con.execute("SELECT count(*) FROM a__halos_z0_000").fetchone()[0] > 0
    with pytest.raises(NotFoundError):
        connect(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(NotFoundError):
        connect(tmp_path / "empty")
    (db / "a-b").symlink_to(tmp / "store")
    (db / "a_b").symlink_to(tmp / "store")
    with pytest.raises(ValidationError, match="collides"):
        connect(db)


def test_uncommitted_product_hidden(built, tmp_path):
    tmp, _, _ = built
    import shutil
    dst = tmp_path / "s"
    shutil.copytree(tmp / "store", dst)
    lay = json.loads((dst / "layout.json").read_text())
    del lay["products"]["halos"]
    (dst / "layout.json").write_text(json.dumps(lay))
    con = connect(dst)
    assert con.execute("SELECT count(*) FROM products WHERE name='halos'").fetchone()[0] == 0
