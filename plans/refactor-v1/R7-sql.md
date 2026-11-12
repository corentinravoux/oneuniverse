# R7 — SQL face: DuckDB views over OUF and OUF-Sim

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give both on-disk formats a first-class SQL face without copying data: `oneuniverse.data.sql.connect(root)` returns an in-process DuckDB connection whose views are built on attach from the manifests, partition indexes and sidecars; `oneuniverse.simulation.sql.connect(store_root)` does the same over the chunk index and catalog-sized products.

**Architecture:** Views only (`CREATE VIEW ... AS SELECT ... FROM read_parquet(...)`), one view per dataset over its hive-partitioned parquet, catalog tables (`datasets`, `partitions`, `versions`) built as DuckDB tables from the JSON/parquet metadata at attach time (small). `duckdb` is imported inside the two modules only; the extra is `[sql]`.

**Tech Stack:** duckdb ≥ 1.0 (`read_parquet(..., hive_partitioning=true)`, list types), pyarrow.

**Spec:** design §6.11, §8 item 5. Master plan Global Constraints (no core module imports duckdb; `simulation.sql` imports nothing from `data`).

## Interfaces consumed (R1, R2, R5)

`Database`, `DatasetView`, `VersionedSidecar`, `Manifest`, `read_partition_index`, `SimStore`, `_core.healpix.cone_pixels`, `_core.geometry.Cone`.

## Interfaces produced

```
oneuniverse.data.sql
  connect(root: Path, *, as_of: datetime | None = None, database: str = ":memory:") -> duckdb.DuckDBPyConnection
    views/tables created:
      datasets(name, geometry, kind, survey, n_rows, nside, format_version, created_utc, version, valid_from, valid_to, path)
      partitions(dataset, cell, path, n_rows, ra_min, ra_max, dec_min, dec_max, z_min, z_max)
      versions(dataset, version, created_utc, valid_from, valid_to)           # live + _history manifests
      <dataset view name> = dataset name with "__" and "-" -> "_"; one view per dataset: SELECT * FROM read_parquet('<root>/<ds>/data/*/*.parquet', hive_partitioning=true)  (columns + cell)
      oneuid_<name>(oneuid, dataset, source_row, partition, offset, ra, dec, z, z_type)   for every ONEUID sidecar (current version, or the version live at as_of)
      links_<name>(parent_oneuid, child_oneuid, sep_arcsec, dz, confidence)
  cone_where(cone: Cone, nside: int, *, cell_column="cell", ra_column="ra", dec_column="dec") -> str   # "cell IN (...) AND <haversine(ra,dec) <= radius>" using the dataset's partition nside
  ImportError with a clear message ("pip install oneuniverse[sql]") when duckdb is missing
oneuniverse.simulation.sql
  connect(store_root: Path, *, database=":memory:") -> duckdb connection
    sims(name, box_size, h, omega_m, sigma8, code, created_utc, parent)        # one row (or many when store_root is a SimDatabase root)
    products(sim, name, kind, z, layout, n_chunks)
    chunks(sim, product, z, chunk_id, path, row_start, row_stop, x_min, x_max, y_min, y_max, z_min, z_max)
    tiles(sim, product, z, tile_id, path, i0, i1, j0, j1, k0, k1)
    <product>_z<tag> views for catalog-layout products (halos, particles in reencode projection): SELECT * FROM read_parquet(chunk files)
  box_where(cube: Cube) -> str   # "x_min < hi AND x_max > lo AND ..." for the chunks table
```

---

### Task 1: `data.sql`

**Files:**
- Create: `oneuniverse/data/sql.py`
- Test: `test/data/test_sql.py` (module-level `pytest.importorskip("duckdb")`)

- [ ] **Step 1: Write the failing tests**

```python
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

def test_as_of_selects_old_sidecar(db):
    t0 = datetime.now(timezone.utc); time.sleep(0.01); db.build_oneuid(["a"], rules=CrossMatchRules(), name="ab")
    assert connect(db.root, as_of=t0).sql("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 2 and connect(db.root).sql("SELECT count(DISTINCT dataset) FROM oneuid_ab").fetchone()[0] == 1

def test_list_columns_are_native(tmp_path):
    spec = PdfSpec("interp", grid=tuple(np.linspace(0, 3, 11))); write_synthetic_dataset(tmp_path / "p", 500, 0, with_pdf=spec)
    con = connect(tmp_path); assert con.sql("SELECT len(z_pdf) FROM p LIMIT 1").fetchone()[0] == 11

def test_missing_duckdb_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"oneuniverse\[sql\]"): importlib.reload(oneuniverse.data.sql).connect(Path("."))
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(data): DuckDB SQL face over OUF`.

---

### Task 2: `simulation.sql`

**Files:**
- Create: `oneuniverse/simulation/sql.py`
- Test: `test/simulation/test_sql.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_chunk_query_matches_store(tmp_path):
    src = make_native_sim(tmp_path / "sim", n_particles=50_000); st = build_store(tmp_path / "store", manifest=native_manifest(src), products=native_products(src), plan=ExecutionPlan(), n_side=4)
    con = connect(tmp_path / "store"); cube = Cube((10, 10, 10), (30, 30, 30))
    ids = sorted(r[0] for r in con.sql(f"SELECT chunk_id FROM chunks WHERE product='particles' AND z=0.0 AND {box_where(cube)}").fetchall())
    assert ids == sorted(overlapping_chunks(st.index("particles", 0.0), cube, 100.0).tolist())
    assert con.sql("SELECT count(*) FROM halos_z0_000").fetchone()[0] == pq.read_table(src / "halos_z0.000.parquet").num_rows
    assert con.sql("SELECT box_size, code FROM sims").fetchone() == (100.0, "fixture")
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(simulation): DuckDB SQL face over OUF-Sim`.

---

### Task 3: Close-out

- [ ] **Step 1:** `test/test_guards.py`: add the rule "no module outside `data/sql.py` and `simulation/sql.py` imports `duckdb`".
- [ ] **Step 2:** `pytest -n 6`; single-process < 60 s.
- [ ] **Step 3:** STATUS.md + memory; commit + push claude_config. Commit — `docs: R7 close-out`.

**Exit criterion:** cone count, n(z) group-by and ONEUID join in SQL equal the Python API results on the fixtures; `as_of` resolves sidecar versions.
