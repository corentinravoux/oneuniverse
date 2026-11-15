# SQL face

Both on-disk formats are readable as SQL through DuckDB. The connection holds
views over the existing parquet files; nothing is copied, and the Python API
and SQL read the same files. Array columns (photo-z PDFs, delta values) are
native DuckDB lists. DuckDB is the optional extra `[sql]`
(`pip install "oneuniverse[sql]"`); it is imported only by the two `connect`
functions.

## Survey databases: `oneuniverse.data.sql`

```python
from oneuniverse.data import sql, Cone

con = sql.connect(db_root)
con.execute("SELECT name, n_rows, survey FROM datasets").fetchall()

where = sql.cone_where(Cone(185, 15, 10), nside=4)   # nside of the dataset
con.execute(f"SELECT count(*) FROM eboss_qso WHERE {where}").fetchone()
```

`connect(root)` returns a `duckdb.DuckDBPyConnection` with:

- tables `datasets`, `partitions` and `versions`, built from the manifests, the
  partition indexes and the manifest history;
- one view per dataset over its hive-partitioned parquet (`cell` is a column);
- views `oneuid_<name>` and `links_<name>` over the versioned sidecars.

`cone_where(cone, nside)` returns a predicate on a dataset view: a prefilter on
the `cell` column plus the exact angular cut. Use the dataset's partition nside
(`DatasetView.nside`).

### `as_of`

`connect(root, as_of=<timezone-aware datetime>)` mirrors `Database.as_of`: a
dataset is kept when its live manifest was created at or before that time, and
the sidecar versions live at that time are selected. Timestamps in the catalog
tables are ISO-8601 UTC strings, exactly as stored.

### Joins

`oneuid_<name>.partition` is the partition ordinal (the row number in the
dataset's `_index.parquet`), not the HEALPix cell. Join it to
`partitions.partition` together with `dataset`. The recommended join key to a
dataset view is `source_row`.

## Simulation stores: `oneuniverse.simulation.sql`

```python
from oneuniverse.simulation import sql, Cube

con = sql.connect(store_root)     # one store, or a SimDatabase root
con.execute("SELECT sim, name, kind, z FROM products").fetchall()

where = sql.box_where(Cube((0, 0, 0), (50, 50, 50)), box_size=500.0)
con.execute(f"SELECT path FROM chunks WHERE {where}").fetchall()
```

The connection holds catalog tables `sims`, `products`, `chunks` and `tiles`,
and one view per catalog product in the `chunks` layout and per redshift, named
`<product>_z<tag>` (for example `halos_z0_000`) for a single store and
`<sim>__<product>_z<tag>` for a `SimDatabase`. Only committed products are
exposed. `box_where(cube, box_size)` selects the chunks whose bounding box
intersects the cube; with `box_size` it splits a cube across the periodic
boundary exactly as `SimStore` does.

## DuckDB 2.0 notes

- A relation has no `fetchone`: write `con.execute(query).fetchone()`.
- `partition` and `offset` are SQL keywords: quote them, `"partition"` and
  `"offset"`.
- Glob characters (`[`, `*`, `?`) in the database or store path are interpreted
  by DuckDB in the view file lists. Avoid them.
