"""OUF-Sim → SQL export (SQLite materialise mode).

Schema per ``research/2026-06-10-structural-review-and-sql-design.md`` §5.3.
The sidecar ``_index.parquet`` files are already tables, so the mapping is
direct:

    sims          one row per store (manifest verbatim)
    sim_products  product × z × partition scheme × projection
    sim_chunks    every index row (bbox / pixel / native row-range) — the
                  queryable map "which file holds box X", in pure SQL
    halos / lightcone / tree   materialised (catalog-sized)
    sim_lineage   parent→child resimulation provenance

Materialisation policy (honest at scale): bulk products (particles, field
tiles) are **index-only** — SQL answers *where* the bytes are; the bytes stay
in parquet/npy. Small products materialise fully. Rule-1 clean: imports
nothing from ``oneuniverse.data``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, Union

import pyarrow.parquet as pq

_DDL = """
CREATE TABLE IF NOT EXISTS sims (
  sim_id INTEGER PRIMARY KEY,
  sim_name TEXT UNIQUE NOT NULL, sim_kind TEXT, code TEXT,
  box_size REAL, n_grid INTEGER, n_particles INTEGER,
  manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_products (
  sim_id INTEGER REFERENCES sims(sim_id),
  product TEXT NOT NULL, z REAL,
  partition_scheme TEXT, projection TEXT,
  PRIMARY KEY (sim_id, product, z)
);
CREATE TABLE IF NOT EXISTS sim_chunks (
  sim_id INTEGER, product TEXT, z REAL, chunk_id INTEGER,
  xlo REAL, xhi REAL, ylo REAL, yhi REAL, zlo REAL, zhi REAL,
  pixel INTEGER, n_rows INTEGER,
  file TEXT, native_file TEXT, row_start INTEGER, row_stop INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks
  ON sim_chunks(sim_id, product, z, xlo, xhi);
CREATE TABLE IF NOT EXISTS sim_lineage (
  parent TEXT, child TEXT, region TEXT, ic_source TEXT, valid_time TEXT
);
"""

#: per-z products materialised fully into their own SQL table
_MATERIALISE = ("halos",)
#: store-level parquet products materialised fully
_MATERIALISE_FLAT = ("lightcone", "tree")


def _insert_df_table(con: sqlite3.Connection, table, name: str,
                     extra_cols: dict) -> None:
    df = table.to_pandas()
    for k, v in extra_cols.items():
        df.insert(0, k, v)
    # 'z' as a position column collides with the redshift tag column name;
    # the linear halos use x/y/z positions -> rename position z to zpos.
    if "z" in df.columns and "z_tag" in extra_cols:
        df = df.rename(columns={"z": "zpos"})
        df = df.rename(columns={"z_tag": "z"})
    df.to_sql(name, con, if_exists="append", index=False)


def export_sim_sql(store: Union[str, Path], out: Union[str, Path],
                   *, lineage: Optional[list] = None) -> Path:
    """Materialise one OUF-Sim store into a SQLite file. Returns its path."""
    store = Path(store)
    out = Path(out)
    payload = json.loads((store / "manifest.json").read_text())
    layout = payload.get("store_layout", {})

    con = sqlite3.connect(out)
    try:
        con.executescript(_DDL)
        cur = con.execute(
            "INSERT INTO sims (sim_name, sim_kind, code, box_size, n_grid,"
            " n_particles, manifest_json) VALUES (?,?,?,?,?,?,?)",
            (payload["sim_name"], payload.get("sim_kind"),
             payload.get("code"), payload.get("box_size"),
             payload.get("n_grid"), payload.get("n_particles"),
             json.dumps(payload)))
        sim_id = cur.lastrowid

        for product, entry in layout.items():
            # entry is either {ztag: info} or a flat info dict (lightcone/tree)
            z_infos = (entry.items()
                       if isinstance(entry, dict) and any(
                           k.startswith("z") and isinstance(v, dict)
                           and "dir" in v for k, v in entry.items())
                       else [(None, entry)])
            for ztag, info in z_infos:
                if not isinstance(info, dict) or "dir" not in info:
                    continue
                z = float(ztag[1:]) if ztag else None
                con.execute(
                    "INSERT OR IGNORE INTO sim_products VALUES (?,?,?,?,?)",
                    (sim_id, product, z, info.get("partition"),
                     info.get("projection", "reencode")))
                idx_rel = info.get("index")
                if idx_rel and (store / idx_rel).exists():
                    rows = pq.read_table(store / idx_rel).to_pylist()
                    con.executemany(
                        "INSERT INTO sim_chunks VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [(sim_id, product, z,
                          r.get("chunk_id", r.get("tile_id")),
                          r.get("xlo"), r.get("xhi"), r.get("ylo"),
                          r.get("yhi"), r.get("zlo"), r.get("zhi"),
                          r.get("pixel"), r.get("n_rows"),
                          r.get("file"), r.get("native_file"),
                          r.get("row_start"), r.get("row_stop"))
                         for r in rows])
                # materialise small per-z catalog products
                if product in _MATERIALISE and idx_rel:
                    pdir = store / info["dir"]
                    for r in pq.read_table(store / idx_rel).to_pylist():
                        f = r.get("file")
                        if f:
                            _insert_df_table(
                                con, pq.read_table(pdir / f), product,
                                {"z_tag": z, "sim_id": sim_id})
            if product in _MATERIALISE_FLAT and isinstance(entry, dict) \
                    and entry.get("dir"):
                pdir = store / entry["dir"]
                for f in sorted(Path(pdir).glob("part_*.parquet")):
                    _insert_df_table(con, pq.read_table(f), product,
                                     {"sim_id": sim_id})

        for e in (lineage or []):
            con.execute("INSERT INTO sim_lineage VALUES (?,?,?,?,?)",
                        (e.get("parent"), e.get("child"), e.get("region"),
                         e.get("ic_source"), e.get("valid_time")))
        con.commit()
    finally:
        con.close()
    return out
