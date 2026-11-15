import importlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("duckdb")

from fixtures.synthetic import write_synthetic_dataset
from fixtures.two_surveys import make_two_survey_db

import oneuniverse.data.sql
from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.geometry import Cone
from oneuniverse.data.oneuid import CrossMatchRules
from oneuniverse.data.pdf import PdfSpec
from oneuniverse.data.sql import cone_where, connect


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    root = tmp_path_factory.mktemp("sql"); db = make_two_survey_db(root, n_each=20_000, n_common=5_000); db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); return db


def test_catalog_tables(db):
    con = connect(db.root); assert con.sql("SELECT name, n_rows FROM datasets ORDER BY name").fetchall() == [("a", 20_000), ("b", 20_000)]
    assert con.sql("SELECT count(*) FROM partitions WHERE dataset='a'").fetchone()[0] == db["a"].n_partitions


def test_dataset_view_and_cone_matches_python(db):
    con = connect(db.root); cone = Cone(185.0, 15.0, 5.0); v = db["a"]
    n_sql = con.sql(f"SELECT count(*) FROM a WHERE {cone_where(cone, v.nside)}").fetchone()[0]
    assert n_sql == v.table(cone=cone).num_rows


def test_nz_groupby(db):
    con = connect(db.root); rows = con.sql("SELECT floor(z*10)/10 AS zb, count(*) FROM a GROUP BY zb ORDER BY zb").fetchall(); assert sum(r[1] for r in rows) == 20_000


def test_oneuid_join_counts_multi(db):
    con = connect(db.root); n_multi = con.sql("SELECT count(*) FROM (SELECT oneuid FROM oneuid_ab GROUP BY oneuid HAVING count(*) > 1)").fetchone()[0]
    assert n_multi == db.load_oneuid("ab").n_multi
    joined = con.sql("SELECT count(*) FROM oneuid_ab o JOIN a ON o.dataset='a' AND o.source_row=a.source_row").fetchone()[0]; assert joined == 20_000


def test_oneuid_columns_and_versions(db):
    con = connect(db.root)
    cols = [r[0] for r in con.sql("DESCRIBE oneuid_ab").fetchall()]
    assert cols == ["oneuid", "dataset", "source_row", "partition", "offset", "ra", "dec", "z", "z_type"]
    assert [r[0] for r in con.sql("DESCRIBE datasets").fetchall()] == [
        "name", "geometry", "kind", "survey", "n_rows", "nside", "format_version",
        "created_utc", "version", "valid_from", "valid_to", "path"]
    assert con.sql("SELECT count(*) FROM versions WHERE dataset='a'").fetchone()[0] == 1


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
    assert con.sql("SELECT name FROM datasets").fetchall() == [("p",)]


def test_view_name_mapping_and_collision(tmp_path):
    write_synthetic_dataset(tmp_path / "x" / "y-z", 300, 0)
    con = connect(tmp_path)
    assert con.sql("SELECT name FROM datasets").fetchall() == [("x__y-z",)]
    assert con.sql('SELECT count(*) FROM "x_y_z"').fetchone()[0] == 300
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


def test_as_of_missing_files_raises_and_early_excludes(tmp_path):
    t0 = datetime.now(UTC); time.sleep(0.01)
    write_synthetic_dataset(tmp_path / "p", 300, 0)
    assert connect(tmp_path, as_of=t0).sql("SELECT count(*) FROM datasets").fetchone()[0] == 0
    time.sleep(0.01); t1 = datetime.now(UTC); time.sleep(0.01)
    write_synthetic_dataset(tmp_path / "p", 400, 1, overwrite=True)
    assert connect(tmp_path).sql("SELECT count(*) FROM p").fetchone()[0] == 400
    assert connect(tmp_path).sql("SELECT count(*) FROM versions").fetchone()[0] == 2
    with pytest.raises(FormatError, match="no longer exist"):
        connect(tmp_path, as_of=t1)


def test_list_columns_are_native(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 11))); write_synthetic_dataset(tmp_path / "p", 500, 0, with_pdf=spec)
    con = connect(tmp_path); assert con.sql("SELECT len(z_pdf) FROM p LIMIT 1").fetchone()[0] == 11


def test_missing_duckdb_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"oneuniverse\[sql\]"): importlib.reload(oneuniverse.data.sql).connect(Path("."))


def test_as_of_selects_old_sidecar(db):
    t0 = datetime.now(UTC); time.sleep(0.01); db.build_oneuid(["a"], rules=CrossMatchRules(), name="ab")
    assert connect(db.root, as_of=t0).sql("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 2 and connect(db.root).sql("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 1
