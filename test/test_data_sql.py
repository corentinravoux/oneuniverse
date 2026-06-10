"""SQL export P1-P3 — OUF -> SQLite, parity with DatasetView.

The exported database must answer the same questions as the parquet store:
row counts, z-range selections, HEALPix-pruned cone queries, and exact PDF
payload round-trips. Cosmology-free by construction (it copies OUF verbatim).
"""
import json
import sqlite3
import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.sql import (export_oneuid, export_sql,
                                  export_subobject_links)

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import (synthetic_point_view,  # noqa: E402
                                  synthetic_shear_view)


def test_export_sql_objects_and_partitions_parity(tmp_path):
    view = synthetic_point_view(tmp_path, n=4000, seed=1, name="synth")
    db = export_sql([view], tmp_path / "cat.sqlite")
    con = sqlite3.connect(db)
    # registry row
    name, geom, fmt, nrows = con.execute(
        "SELECT survey_name, geometry, format_version, n_rows "
        "FROM datasets").fetchone()
    assert name == "synth" and geom == "point" and fmt == "2.5.0"
    assert nrows == view.n_rows
    # objects parity: total count + z-range count match the parquet reader
    assert con.execute("SELECT COUNT(*) FROM objects").fetchone()[0] == 4000
    zlo, zhi = 0.4, 0.6
    sql_n = con.execute(
        "SELECT COUNT(*) FROM objects WHERE z BETWEEN ? AND ?",
        (zlo, zhi)).fetchone()[0]
    assert sql_n == len(view.read(z_range=(zlo, zhi)))
    # partitions table mirrors the manifest partition index
    n_part = con.execute("SELECT COUNT(*) FROM partitions").fetchone()[0]
    assert n_part == view.manifest.n_partitions
    assert con.execute(
        "SELECT SUM(n_rows) FROM partitions").fetchone()[0] == 4000
    # spot row equality
    gid, ra, dec = con.execute(
        "SELECT galaxy_id, ra, dec FROM objects ORDER BY galaxy_id LIMIT 1"
    ).fetchone()
    row = view.read().sort_values("galaxy_id").iloc[0]
    assert gid == row["galaxy_id"]
    assert ra == pytest.approx(row["ra"]) and dec == pytest.approx(row["dec"])
    con.close()


def test_export_sql_healpix_cone_query(tmp_path):
    """The OUF pruning idiom expressed in SQL: cone -> pixel list -> WHERE IN."""
    view = synthetic_point_view(tmp_path, n=4000, seed=2, name="synth")
    db = export_sql([view], tmp_path / "cat.sqlite")
    con = sqlite3.connect(db)
    df = view.read()
    cra, cdec = float(df["ra"].median()), float(df["dec"].median())
    vec = hp.ang2vec(np.radians(90 - cdec), np.radians(cra))
    pix = hp.query_disc(32, vec, np.radians(5.0), nest=True, inclusive=True)
    qmarks = ",".join("?" * len(pix))
    got = pd.read_sql_query(
        f"SELECT ra, dec FROM objects WHERE healpix32 IN ({qmarks})",
        con, params=[int(p) for p in pix])
    # superset of the exact 5-degree cone, subset of the full catalog
    cosang = (np.sin(np.radians(cdec)) * np.sin(np.radians(got["dec"]))
              + np.cos(np.radians(cdec)) * np.cos(np.radians(got["dec"]))
              * np.cos(np.radians(got["ra"] - cra)))
    n_exact = int((np.degrees(np.arccos(np.clip(cosang, -1, 1))) <= 5.0).sum())
    assert 0 < n_exact <= len(got) < len(df)
    con.close()


def test_export_sql_pdf_blobs_round_trip(tmp_path):
    view = synthetic_shear_view(tmp_path, n=300, seed=3, with_pdf=True,
                                name="pdfsrc")
    db = export_sql([view], tmp_path / "cat.sqlite", include_pdfs=True)
    con = sqlite3.connect(db)
    spec = json.loads(con.execute(
        "SELECT spec_json FROM pdf_specs").fetchone()[0])
    assert spec["parameterisation"] == "interp"
    gid, blob = con.execute(
        "SELECT galaxy_id, pdf_values FROM pdf_payloads "
        "ORDER BY galaxy_id LIMIT 1").fetchone()
    back = np.frombuffer(blob, dtype=np.float32)
    orig = np.asarray(view.read().sort_values("galaxy_id")
                      .iloc[0]["z_pdf_values"], dtype=np.float32)
    np.testing.assert_allclose(back, orig)
    assert len(back) == spec["n_components"]
    con.close()


def test_export_oneuid_and_subobject_tables(tmp_path):
    view = synthetic_point_view(tmp_path, n=100, seed=4, name="synth")
    db = export_sql([view], tmp_path / "cat.sqlite")
    con = sqlite3.connect(db)
    # ONEUID sidecar schema: one row per (dataset, row_index)
    idx = pd.DataFrame({"oneuid": [0, 0, 1], "dataset": ["synth"] * 3,
                        "row_index": [3, 7, 11]})
    export_oneuid(con, idx, name="run1", rules_json='{"sky_tol": 1.0}')
    links = pd.DataFrame({"parent_oneuid": [0], "child_oneuid": [1],
                          "confidence": [0.9]})
    export_subobject_links(con, links, name="clu", relation_type="containment")
    # the cross-survey join the schema exists for:
    n = con.execute(
        "SELECT COUNT(*) FROM oneuid_members m "
        "JOIN oneuid_runs r ON r.run_id = m.run_id "
        "WHERE r.name='run1' AND m.oneuid=0").fetchone()[0]
    assert n == 2
    rel, = con.execute(
        "SELECT relation_type FROM subobject_links LIMIT 1").fetchone()
    assert rel == "containment"
    con.close()
