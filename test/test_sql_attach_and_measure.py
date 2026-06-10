"""SQL export P5 (DuckDB attach DDL) + P6 (MeasurementSet.to_sql)."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.data.sql import attach_sql_ddl
from oneuniverse.measure import build_galaxy_clustering

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view  # noqa: E402


def test_attach_ddl_generates_views(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1, name="synth")
    ddl = attach_sql_ddl([view])
    assert "CREATE OR REPLACE VIEW synth" in ddl
    assert "read_parquet" in ddl and "hive_partitioning=1" in ddl


def test_attach_ddl_executes_in_duckdb_if_available(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    view = synthetic_point_view(tmp_path, n=800, seed=2, name="synth")
    con = duckdb.connect()
    con.execute(attach_sql_ddl([view]))
    n, = con.execute("SELECT COUNT(*) FROM synth").fetchone()
    assert n == 800                                   # zero-copy view parity
    zlo, zhi = 0.4, 0.6
    nsql, = con.execute(
        "SELECT COUNT(*) FROM synth WHERE z BETWEEN ? AND ?",
        [zlo, zhi]).fetchone()
    assert nsql == len(view.read(z_range=(zlo, zhi)))
    con.close()


def test_measurement_set_to_sql(tmp_path):
    view = synthetic_point_view(tmp_path, n=1500, seed=3)
    ms = build_galaxy_clustering(view, tracer="gal", z_range=(0.2, 0.9),
                                 weights=[ColumnWeight("weight_comp")],
                                 nz_edges=np.linspace(0, 1.2, 13),
                                 randoms="generate", n_randoms=3000, seed=1)
    db = ms.to_sql(tmp_path / "ms.sqlite")
    con = sqlite3.connect(db)
    fam, stat = con.execute(
        "SELECT estimator_family, statistic FROM measurement_sets").fetchone()
    assert (fam, stat) == ("clustering", "pk_multipole")
    n_cat, = con.execute("SELECT COUNT(*) FROM catalog_gal").fetchone()
    assert n_cat == len(ms.products["gal"].catalog)
    n_rnd, = con.execute("SELECT COUNT(*) FROM randoms_gal").fetchone()
    assert n_rnd == 3000
    # weights came along
    w, = con.execute("SELECT weight FROM catalog_gal LIMIT 1").fetchone()
    assert w > 0
    con.close()
