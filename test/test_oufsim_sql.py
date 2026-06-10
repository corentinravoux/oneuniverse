"""SQL export P4 — OUF-Sim -> SQLite, box-query parity with SimStore."""
import sqlite3
from pathlib import Path

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.sql import export_sim_sql
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_export_sim_sql_chunks_and_box_parity(tmp_path):
    lin = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0, 0.5), seed=2)
    store = write_oufsim_store(lin, tmp_path / "s", sim_name="d",
                               particle_chunk_nside=4)
    db = export_sim_sql(store, tmp_path / "sim.sqlite")
    con = sqlite3.connect(db)

    # registry
    name, box, kind = con.execute(
        "SELECT sim_name, box_size, sim_kind FROM sims").fetchone()
    assert name == "d" and box == 200.0
    prods = {r[0] for r in con.execute(
        "SELECT DISTINCT product FROM sim_products")}
    assert {"snapshots", "fields", "halos"} <= prods

    # chunk index parity: total particle rows per z == store n_rows
    tot = con.execute(
        "SELECT SUM(n_rows) FROM sim_chunks "
        "WHERE product='snapshots' AND z=0.0").fetchone()[0]
    assert tot == 32 ** 3

    # THE parity test: the SQL bbox-overlap query returns exactly the chunks
    # SimStore.read_box touches for the same Cube.
    cube = Cube(0, 50, 0, 50, 0, 50)
    s = SimStore(store)
    s.read_box("snapshots", 0.0, cube)
    sql_chunks = con.execute(
        "SELECT COUNT(*) FROM sim_chunks WHERE product='snapshots' AND z=0.0"
        " AND NOT (xhi < ? OR xlo > ? OR yhi < ? OR ylo > ?"
        "          OR zhi < ? OR zlo > ?)",
        (cube.xlo, cube.xhi, cube.ylo, cube.yhi, cube.zlo, cube.zhi)
    ).fetchone()[0]
    assert sql_chunks == s.last_read_stats["chunks_read"]

    # halos materialised with positions
    n_h = con.execute("SELECT COUNT(*) FROM halos").fetchone()[0]
    assert n_h > 0
    x, = con.execute("SELECT MAX(x) FROM halos").fetchone()
    assert 0 <= x <= 200.0
    con.close()


def test_export_sim_sql_lineage_optional(tmp_path):
    """Stores without lineage/lightcone still export cleanly."""
    lin = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=150.0,
                              n_grid=16, redshifts=(0.0,), seed=1,
                              with_lightcone=False)
    store = write_oufsim_store(lin, tmp_path / "s", sim_name="e")
    db = export_sim_sql(store, tmp_path / "sim.sqlite")
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM sims").fetchone()[0] == 1
    con.close()
