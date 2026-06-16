# oneuniverse — Complete Structural Review, Bug Register & SQL Database Design

**Date:** 2026-06-10 · **Scope:** the whole package (16,364 LOC: `data` 8,461,
`simulation` 4,570, `measure` 1,436, `combine` 1,253, `twin` 613) ·
**Method:** external-reviewer pass — code read of every load-bearing module,
pattern sweeps (exception handling, resource handling, float comparisons,
mutable defaults), and **empirical bug confirmation** (each confirmed bug was
reproduced with a script before being listed). Complements the 2026-06-05
[`REVIEW.md`](../REVIEW.md) (whose H1/H2/M1/M2/L1/L2 items are fixed); this
review goes deeper on *structure* and adds the **storage redesign + SQL export
design** requested by the owner.

---

## 1 · The package's goals, restated (the yardstick)

1. **G1 — Universal ingestion.** Any cosmology survey (catalog, sightline,
   map, cube, light curve) → one standard, partial-access on-disk format
   (OUF), *cosmology-free*.
2. **G2 — One identity.** The same physical object cross-identified across
   surveys (ONEUID, bitemporal) with sub-object hierarchies.
3. **G3 — Analysis-ready output.** A general, cosmology-free `MeasurementSet`
   per probe (the P1→P2 handoff) that external estimators consume.
4. **G4 — Simulation substrate.** Store/query simulations at scale
   (partial access, wrap-in-place), run/resimulate (PM, TreePM), couple to
   data (twin).
5. **G5 — Honesty & discipline.** No cosmology in data; import boundaries
   enforced; toy components labelled.

### Scorecard

| Goal | Status | Evidence | Main structural gap |
|---|---|---|---|
| G1 | **strong** (2 real + 7 scaffold loaders) | OUF 2.5, 6 geometries, real eBOSS/DESI | converter monolith; loader scaffolds; no SQL face (→ §6) |
| G2 | **strong, under-integrated** | oneuid.py + subobject.py + bitemporal archive | ONEUID sidecars are parquet-silos; no relational query surface (→ §6) |
| G3 | **strong** (6 builders, 3 subtypes, save/load) | measure/, 12-class generality test | pipeline copy-paste across builders; weak typing of atom slots |
| G4 | **machinery strong, physics toy (by design)** | OUF-Sim multi-backend, TreePM, Wiener | 4 duplicate validation modules; twin module sprawl |
| G5 | **strong** | guards + cosmology-column scan | `measure` lacks its own import guard |

---

## 2 · Structural review by subpackage

### 2.1 `oneuniverse.data` (8.5 kLOC — half the package)

**Strengths.** The OUF design is right: typed `Manifest` + HEALPix-partitioned
parquet + per-partition stats gives real partial access (measured 64× pruning);
geometry polymorphism (POINT/SIGHTLINE/HEALPIX/GW_SKYMAP/CUBE/LIGHTCURVE) is
clean; atomic writes; the bitemporal ONEUID archive (`__{ISO}Z` versions +
`as_of`) is genuinely good engineering.

**S1 — `converter.py` is a 913-line monolith.** It contains the column
validator, the POINT writer, the SIGHTLINE writer (`convert_sightlines`), CSV/
FITS linkback, row counting, and orchestration. LIGHTCURVE already lives in its
own `_converter_lightcurve.py` — the precedent exists. *Refactor:* split into
`_converter_point.py`, `_converter_sightline.py`, `_linkback.py`, keeping
`converter.py` as the dispatch façade. Mechanical, low-risk, testable by the
existing suite.

**S2 — Three registries, three idioms.** `data/_registry.py` (loaders,
instance-on-get), `simulation/converter.py::_REGISTRY` (classes),
`twin/engine.py::register_engine` (engines), plus `oufsim/native.py::ADAPTERS`
(adapter instances). Four hand-rolled registries with different duplicate-
handling and lookup semantics. *Refactor:* one `oneuniverse._registry.Registry`
utility (register/get/list/status), parameterised per use. Also enables a
single plugin entry-point mechanism (`importlib.metadata.entry_points`) so
**community survey loaders can be installed as separate packages** — the right
long-term answer to the 7 scaffold loaders.

**S3 — Linkback only covers FITS/CSV** (`converter.py:546` raises for
parquet/HDF5 originals). DESI VACs increasingly ship parquet. *Fix:* a
parquet branch is ~10 lines (`pq.read_table(path, columns=..).take(rows)`).

**S4 — `database.py` scan swallow-and-warn.** The directory scan catches
`Exception` per-manifest and warns (acceptable — a corrupt dataset should not
brick the database), but there is no `strict=True` escape for CI use.

### 2.2 `oneuniverse.combine` (1.3 kLOC)

Healthy. Weight ABC + `__mul__` composition is clean; the
`(survey_type, sub_kind, z_type)` default registry is the right shape. One
inconsistency: `FKPWeight` takes a **callable** `nbar(z)` while every other
weight takes a **column name** — this cost a real debugging round during the
measure build. *Refactor:* accept `Union[str, Callable]` (column name → lookup)
for uniformity.

### 2.3 `oneuniverse.measure` (1.4 kLOC — the newest layer)

**Strengths.** The Universal DataProduct (PointSet/Sightline/FieldMap +
optional atom slots) demonstrably covers the probe space (12-class coverage
test); builders are composable; `to_dir/from_dir` completes the handoff; the
cosmology-free invariant is enforced on *contents*.

**S5 — Pipeline copy-paste.** `clustering.py`, `lensing.py`, `pvsn.py` each
re-implement select→weight→window→region→metadata assembly (pvsn has a private
`_base()`; the others inline it). *Refactor:* one
`measure/_pipeline.py::prepare_pointset(view, *, z_range, weights, nside_*) →
(cat, window, region, meta)` consumed by every PointSet builder. Removes ~80
duplicated lines and makes new probes a ~30-line builder.

**S6 — Weak typing of atom slots.** `nz: object`, `window: object`,
`photoz: object`, `covariance: object` — the container's generality was bought
with `object`. *Refactor:* `Union`/`Protocol` types (`NzLike`, `WindowLike`)
— documentation + IDE support without restricting duck-typing.

**S7 — No import guard.** `simulation` is guarded against `data`; `measure`
has no guard at all. It must never import `simulation` (it is the P1→P2 layer).
*Fix:* a `test_measure_import_boundary.py` mirroring the Rule-1 guard.

**S8 — `MeasurementSpec.covariance: str | CovariancePlan`** — a stringly/
typed union with no validation on the string branch. Normalise to
`CovariancePlan` always (`"jackknife"` → `CovariancePlan(kind="jackknife")`).

### 2.4 `oneuniverse.simulation` (4.6 kLOC)

**Strengths.** OUF-Sim mirrors OUF correctly (manifest + sidecar
`_index.parquet` per product); the adapter registry + `build_store` +
`NativeProduct` generality proof (packed_npy second backend) is solid; the
wrap-in-place projection is the standout design (index-only ≈14%);
`ExecutionPlan`/`BackendCapabilities` enforcement is honest (refuse, don't
degrade).

**S9 — FOUR overlapping field-validation modules** (empirically confirmed):
`twin/validation.py` (`RecoveryMetrics`), `twin/verify.py`
(`cross_correlation`, `power_ratio`), `simulation/validation.py`
(`FieldValidation` — the most complete), `simulation/resim/verify.py`
(`_cross_r`, gates). Four k-binning conventions (linear-from-min, kf-spaced,
fixed-12-bin…) for the same estimator. A reviewer cannot tell which r(k) a
result used. *Refactor:* `simulation/validation.py` is canonical; the other
three become thin wrappers (or deprecated aliases) over it. This was already
flagged informally during the session; it is the top intra-package science-code
debt.

**S10 — `twin` module sprawl.** 10 modules for 613 LOC: `engine.py` (ABCs +
registry) vs `engines.py` (implementations) differ by one letter;
`mock_observe.py` vs `mock_survey.py` overlap; `validation.py` + `verify.py`
both exist (see S9). *Refactor:* fold into 4 modules: `engine.py` (ABCs +
registry + implementations), `mock.py`, `reconstruct.py` (wiener +
constrained), and re-export validation from `simulation.validation`.

**S11 — `store_layout` lives inside `manifest.json`.** Measured: 6 KB for a
toy (10 products × 3 z). It stays KB-scale because chunk detail is correctly
in the parquet sidecars — but the manifest now mixes *identity* (cosmology,
provenance, format version) with *layout* (paths, partition counts), so every
product addition rewrites the identity file. *Refactor:* split
`store_layout` into `layout.json` beside the manifest; manifest carries only a
pointer + hash. Cheap, and it makes the manifest stable/diffable.

### 2.5 Cross-cutting

- **Resource handling:** 10 `json.dump/load(open(...))` without context
  managers, all in `measure/io.py` (B5 below).
- **No mutable-default-arg bugs, no float-equality bugs** found in the sweeps.
- **Naming debt:** `_base_loader.py` is private-named but exports the public
  `SurveyConfig`/`BaseSurveyLoader`; promote to `base_loader.py` with a shim.

---

## 3 · Bug register

Severity: **C**ritical (wrong results / crash on valid input) · **M**edium ·
**L**ow. "Confirmed" = reproduced by script during this review.

### Confirmed

- **B1 (C) — Silent data loss: `FieldMap.axes/beam/interloper` dropped on
  save.** `measure/io.py::_save_product` writes these only `if _jsonable(x)`;
  a numpy frequency axis (the *normal* case for a LIM cube) silently
  serialises as `None` and the round-trip loses the spectral axis. **Repro:**
  FieldMap with `axes={"freq_ghz": np.linspace(...)}` → `to_dir` → `from_dir`
  → `axes is None`. *Fix:* recursively convert ndarray→list before the JSON
  check; **raise** (never silently drop) if still unserialisable.

- **B2 (C) — Crash: weight component named `"total"`.**
  `np.savez(pdir/"weights.npz", total=..., **components)` — a component named
  `total` (a perfectly natural name) raises
  `TypeError: got multiple values for keyword argument 'total'`. *Fix:*
  namespace component keys (`comp_<name>`) in the npz.

- **B3 (C) — Silent garbage: randoms from an empty n(z).**
  `generate_randoms` computes `cdf = cumsum(counts)/cdf[-1]`; for all-zero
  counts this is 0/0 → NaN cdf → `searchsorted` quietly returns bin 0 and the
  function returns *plausible-looking but meaningless* redshifts (confirmed:
  z∈[0.02, 0.22] from a zero histogram). Worse than a crash. *Fix:* raise
  `ValueError("n(z) has zero total weight")`.

- **B4 (M) — Ingested randoms are not filtered or weighted.**
  `build_galaxy_clustering(randoms=<DatasetView>)` ingests `view.read()`
  verbatim: no `z_range` cut (data is cut, randoms are not — biases the radial
  window), and no guaranteed `weight` column (generated randoms have
  `weight=1`; ingested may have none — downstream estimators will KeyError or
  silently mis-weight). *Fix:* apply the same `z_range` to ingested randoms
  and add `weight=1.0` when absent.

- **B5 (L) — 10 unclosed file handles** in `measure/io.py`
  (`json.dump(..., open(...))`). Safe on CPython by refcounting; a real leak
  on PyPy and a Windows file-lock hazard. *Fix:* `Path.write_text/read_text`.

- **B11 (C) — `NamedWeights` silently dropped on load** *(found while fixing
  B2)*: `_load_product` built the `weights` object from `weights.npz` but
  **never passed it to the `PointSet` constructor** — every loaded
  MeasurementSet came back with `weights=None`, untested because the H2
  round-trip tests never covered named weights. A textbook case for why every
  optional atom slot needs round-trip coverage. *Fix:* pass `weights=weights`;
  the B2 regression test now covers the full named-weights round-trip.

### Code-confirmed (read, not executed)

- **B6 (M) — Linkback unsupported for parquet/HDF5 originals**
  (`converter.py:546`) — see S3.
- **B7 (L) — Random jitter is not area-uniform.**
  `randoms._uniform_in_pixels` jitters θ,φ by a constant half-resolution; near
  the poles φ-jitter compresses (area element sinθ dθ dφ), slightly
  over-densifying polar pixels. Irrelevant for mid-latitude footprints
  (eBOSS/DESI), wrong for all-sky randoms. *Fix:* jitter in
  (cosθ, φ·sinθ) or sample sub-pixels at higher NSIDE.
- **B8 (L) — `CovarianceHandle.path` stores an absolute path** — a saved
  MeasurementSet referencing an external covariance breaks when the directory
  moves. *Fix:* store relative-to-set paths with an absolute fallback.
- **B9 (L) — `read_box(device="gpu")` is silently ignored on the
  wrap-in-place branch** (the reference branch takes precedence before the
  cuDF path). Honest behaviour would record `device="cpu(reference)"` in
  `last_read_stats`.
- **B10 (cosmetic) — dead guard in `wiener_reconstruct`**: `denom = b²P + 1/n̄`
  is strictly positive, the `good = denom > 0` mask is unreachable.

### Sweep results (clean)

No mutable default arguments; no float `==` comparisons on physics values; the
broad `except Exception` sites (database scan, best-effort row count, MOC
import) are all defensive-with-logging and acceptable; `np.minimum.at`
bbox accumulation in the oufsim writer is correct; HEALPix `lonlat=True` vs
θ/φ conversions are consistent everywhere they were checked.

---

## 4 · Storage-structure redesign (the two databases)

### 4.1 What exists

**OUF (data):**
```
{survey}/oneuniverse/manifest.json        identity + schema + PartitionStats
                     data/healpix32=N/part_*.parquet
                     objects.parquet      (SIGHTLINE/LIGHTCURVE)
{root}/_oneuid/<name>.parquet + .manifest.json   (+ bitemporal archives)
{root}/_subobject/<name>.parquet + .manifest.json
```

**OUF-Sim (simulation):**
```
{sim}/oufsim/manifest.json                identity + cosmology + store_layout
             {product}/z*/part_*.parquet|tile_*.npy + _index.parquet
```

### 4.2 Critique

1. **Two index dialects for one concept.** OUF keeps partition stats *inside*
   `manifest.json` (`PartitionStats` per HEALPix cell); OUF-Sim keeps them in
   sidecar `_index.parquet` files. The OUF-Sim way is better (the index scales
   independently of the identity file, is mmap/queryable, and is exactly what
   enabled wrap-in-place). OUF's manifest-embedded stats will not scale to
   Rubin-sized partition counts and bloat every manifest read.
2. **Identity and layout are mixed** in both manifests (S11).
3. **The registry is a directory walk.** `OneuniverseDatabase.scan()` globs
   for manifests; there is no persistent catalog of datasets, ONEUID indices,
   and their validity intervals — each session re-derives it.
4. **Cross-cutting artefacts are parquet silos.** ONEUID and sub-object links
   are *relational by nature* (joins!) but live as loose parquet sidecars with
   hand-rolled lookup code (`OneuidQuery`).

### 4.3 Proposed structure (both databases, one contract)

```
{root}/
  catalog.sqlite                ← NEW: the registry DB (§6) — datasets, sims,
                                   oneuid runs, validity, provenance
  {survey}/oneuniverse/
      manifest.json             identity ONLY (schema, frame, version, hash refs)
      layout.json               partitioning declaration (NSIDE, scheme)
      _index.parquet            ← NEW: per-partition stats (cell, n_rows,
                                   z_min/max, extra ranges) — moved OUT of manifest
      data/healpix32=N/part_*.parquet
  {sim}/oufsim/
      manifest.json             identity ONLY
      layout.json               store_layout (moved out)
      {product}/.../_index.parquet      (unchanged — already right)
```

**The unifying contract:** *every* partitioned store = `manifest.json`
(identity) + `layout.json` (how it is partitioned) + `_index.parquet`
(per-partition stats) + payload files. One reader-side pruning code path
(`cube/cone/z_range → overlapping partitions`) shared by `DatasetView` and
`SimStore` — today they are parallel implementations. Migration is
non-breaking: readers accept both manifest versions (2.5 embedded-stats and
2.6 sidecar-index) for one minor cycle.

---

## 5 · SQL export — design (the headline request)

**Requirement.** SQL is the lingua franca; the package must be able to create
a SQL database from both the OUF and OUF-Sim formats.

### 5.1 Engine and philosophy: two modes, one schema

| Mode | Engine | What it does | When |
|---|---|---|---|
| **`materialize`** | **SQLite** (stdlib, zero deps, single portable file) | copies rows into SQL tables | share/archive/medium data (≤ ~10⁸ rows) |
| **`attach`** | **DuckDB** (optional extra) | creates SQL **views over the existing parquet partitions** — zero copy | huge catalogs; the SQL face of *wrap-in-place* |

This mirrors the package's own `reencode` vs `reference` storage projections:
`materialize` ≡ re-encode, `attach` ≡ wrap-in-place. DuckDB reads
hive-partitioned parquet natively
(`CREATE VIEW objects AS SELECT * FROM read_parquet('data/healpix32=*/part_*.parquet', hive_partitioning=1)`),
so the attach mode is a thin DDL generator — no data movement, and the SQL
layer inherits OUF's partition pruning through DuckDB's filter pushdown.

### 5.2 OUF → SQL schema (DDL sketch)

```sql
-- registry ---------------------------------------------------------------
CREATE TABLE datasets (
  dataset_id      INTEGER PRIMARY KEY,
  survey_name     TEXT UNIQUE NOT NULL,
  survey_type     TEXT NOT NULL,            -- spectroscopic | photometric | ...
  geometry        TEXT NOT NULL,            -- point | sightline | ...
  format_version  TEXT NOT NULL,            -- "2.5.0"
  n_rows          INTEGER,
  frame           TEXT, epoch REAL,         -- observational metadata ONLY
  valid_from      TEXT, valid_to TEXT,      -- bitemporal validity
  manifest_json   TEXT NOT NULL             -- full manifest, verbatim (audit)
);
CREATE TABLE partitions (                   -- ≡ the sidecar _index.parquet
  dataset_id INTEGER REFERENCES datasets,
  healpix32  INTEGER NOT NULL,
  n_rows     INTEGER, z_min REAL, z_max REAL,
  file       TEXT NOT NULL,                 -- relative parquet path (attach mode)
  PRIMARY KEY (dataset_id, healpix32)
);
-- objects ----------------------------------------------------------------
CREATE TABLE objects (                      -- one row per object, all surveys
  dataset_id INTEGER REFERENCES datasets,
  galaxy_id  INTEGER NOT NULL,
  ra REAL NOT NULL, dec REAL NOT NULL,
  z REAL, z_type TEXT, z_err REAL,
  healpix32  INTEGER NOT NULL,
  extra      BLOB,                          -- survey-specific cols (msgpack/JSON)
  PRIMARY KEY (dataset_id, galaxy_id)
);
CREATE INDEX idx_objects_hpx ON objects(healpix32);
CREATE INDEX idx_objects_z   ON objects(dataset_id, z);
-- variable-length payloads (photo-z PDFs, spectra): BLOB, not normal form --
CREATE TABLE pdf_payloads (
  dataset_id INTEGER, galaxy_id INTEGER,
  pdf_kind   TEXT,                          -- interp | quant | mixmod | ...
  values     BLOB NOT NULL,                 -- raw float32 array bytes
  PRIMARY KEY (dataset_id, galaxy_id),
  FOREIGN KEY (dataset_id, galaxy_id) REFERENCES objects
);
CREATE TABLE pdf_specs (dataset_id INTEGER PRIMARY KEY, spec_json TEXT);
-- identity (G2 — finally relational) --------------------------------------
CREATE TABLE oneuid_runs (
  run_id INTEGER PRIMARY KEY, name TEXT, rules_json TEXT,
  built_utc TEXT, archived_utc TEXT          -- bitemporality as rows
);
CREATE TABLE oneuid_members (
  run_id INTEGER REFERENCES oneuid_runs,
  oneuid INTEGER NOT NULL,
  dataset_id INTEGER, galaxy_id INTEGER,
  PRIMARY KEY (run_id, dataset_id, galaxy_id)
);
CREATE INDEX idx_oneuid ON oneuid_members(run_id, oneuid);
CREATE TABLE subobject_links (
  run_id INTEGER, relation_type TEXT,        -- containment|causality|association
  parent_oneuid INTEGER, child_oneuid INTEGER, confidence REAL,
  payload_json TEXT                          -- Δt, N_HI, κ_ext, ...
);
```

Design decisions, with reasons:
- **One `objects` table + `dataset_id`**, not per-survey tables: cross-survey
  joins (the whole point of ONEUID) become one-line SQL; canonical columns are
  typed, survey-specific extras go to a `extra` BLOB (normalising 56-band
  J-PAS photometry into columns would explode the schema).
- **Var-length payloads as BLOBs + a spec table**, not child rows: a 1000-bin
  p(z) × 10⁸ objects in normal form is 10¹¹ rows — unusable. BLob + `pdf_specs`
  preserves exact reconstruction (`np.frombuffer`).
- **ONEUID becomes two tables** (`runs`, `members`): the bitemporal archive
  maps to `archived_utc IS NULL` for the live run — `as_of` queries become
  `WHERE built_utc <= ? AND (archived_utc IS NULL OR archived_utc > ?)`.
- **The manifest rides along verbatim** (`manifest_json`) so the SQL file is
  self-describing and auditable.
- **Spatial queries:** `healpix32` BTree covers the common case (cone →
  `query_disc` pixel list → `WHERE healpix32 IN (...)` — exactly the OUF
  pruning, expressed in SQL). An optional SQLite R*Tree on (ra,dec) is a
  follow-up, not core.

### 5.3 OUF-Sim → SQL schema

The sidecar `_index.parquet` files are *already* tables — the mapping is
direct:

```sql
CREATE TABLE sims (
  sim_id INTEGER PRIMARY KEY, sim_name TEXT UNIQUE, sim_kind TEXT,
  code TEXT, box_size REAL, n_particles INTEGER,
  cosmology_json TEXT, unit_frame_json TEXT, provenance_json TEXT,
  manifest_json TEXT NOT NULL
);
CREATE TABLE sim_products (
  sim_id INTEGER REFERENCES sims, product TEXT, z REAL,
  partition_scheme TEXT,                     -- cartesian_chunk | grid_tile | healpix_nest
  projection TEXT,                           -- reencode | reference
  PRIMARY KEY (sim_id, product, z)
);
CREATE TABLE sim_chunks (                    -- ≡ every _index.parquet row
  sim_id INTEGER, product TEXT, z REAL, chunk_id INTEGER,
  xlo REAL, xhi REAL, ylo REAL, yhi REAL, zlo REAL, zhi REAL,
  n_rows INTEGER, file TEXT, native_file TEXT,
  row_start INTEGER, row_stop INTEGER,       -- wrap-in-place range
  PRIMARY KEY (sim_id, product, z, chunk_id)
);
CREATE INDEX idx_chunks_bbox ON sim_chunks(sim_id, product, z, xlo, xhi);
-- small products materialize fully; bulk products index-only ---------------
CREATE TABLE halos     (sim_id INTEGER, z REAL, halo_id INTEGER, x REAL, y REAL,
                        zpos REAL, mass REAL, PRIMARY KEY (sim_id, z, halo_id));
CREATE TABLE lightcone (sim_id INTEGER, lon REAL, lat REAL, redshift REAL,
                        mass REAL, healpix32 INTEGER);
CREATE TABLE sim_lineage (parent TEXT, child TEXT, region TEXT,
                          ic_source TEXT, valid_time TEXT);
```

**Materialisation policy (the honest scale answer):** `halos`, `lightcone`,
`tree`, lineage → materialise (they are catalog-sized). `snapshots`
(particles) and `fields` → **index-only in SQLite** (the `sim_chunks` table is
the queryable map: *which file/byte-range holds box X*), or **full SQL views
in DuckDB attach mode** (DuckDB reads the parquet chunks; `.npy` tiles stay
index-only). A box query in SQL:
```sql
SELECT file, row_start, row_stop FROM sim_chunks
WHERE sim_id=? AND product='snapshots' AND z=0.0
  AND NOT (xhi < :xlo OR xlo > :xhi OR yhi < :ylo OR ylo > :yhi
           OR zhi < :zlo OR zlo > :zhi);
```
— the same pruning `SimStore.read_box` does, now available to any SQL client.

### 5.4 API + implementation plan

```python
# oneuniverse/data/sql.py
export_sql(root_or_view, out="catalog.sqlite", *, mode="materialize"|"attach",
           datasets=None, include_pdfs=True) -> Path
# oneuniverse/simulation/oufsim/sql.py
export_sim_sql(store_or_root, out, *, mode=..., materialize=("halos","lightcone","tree"))
# bonus, trivially on top of measure/io.py:
MeasurementSet.to_sql(path)        # catalog+randoms as tables, spec/meta as JSON rows
```

Phased TDD plan (each phase independently shippable):
1. **P1 `data/sql.py` materialize**: datasets+partitions+objects round-trip
   test (counts, spot rows, healpix cone query parity with `DatasetView`).
2. **P2 ONEUID/subobject tables** + an `as_of` SQL parity test vs
   `database.as_of`.
3. **P3 PDF BLOBs** with `np.frombuffer` round-trip.
4. **P4 `oufsim/sql.py`**: sims/products/chunks (+halos/lightcone); box-query
   parity test vs `SimStore.read_box` chunk list.
5. **P5 DuckDB attach mode** (optional-dep `oneuniverse[sql]`): generated
   `CREATE VIEW` DDL; parity test gated on duckdb import.
6. **P6 `MeasurementSet.to_sql`** + docs/notebook cell.

Estimated ~600 LOC + tests, no new hard dependencies (SQLite is stdlib;
DuckDB optional).

---

## 6 · Prioritised roadmap

| # | Item | Type | Effort |
|---|---|---|---|
| 1 | Fix confirmed bugs B1–B5 | bug | S — **✅ done (incl. B11)** |
| 2 | SQL export P1–P4 (SQLite materialize, both formats) | feature (owner ask) | M — **✅ done (P1–P6: `data/sql.py`, `oufsim/sql.py`, `ms.to_sql`, attach DDL)** |
| 3 | S9 consolidate 4 validation modules → `simulation.validation` | debt | S — **✅ done** (`binned_mode_powers` canonical core; 3 thin wrappers, numerics identical) |
| 4 | B4 ingest-randoms filter + weight | bug | S — **✅ done** (first fix wave) |
| 5 | §4.3 move partition stats out of OUF manifest (format 2.6) + unify index contract | structure | M — **✅ done** (OUF 2.6: identity-only manifest + `_index.parquet` sidecar; pre-2.6 back-compat; version constants single-sourced) |
| 6 | S5 measure `_pipeline.py` + S7 import guard | debt | S — **✅ done** (`prepare_pointset` backs all PointSet builders; boundary guard test) |
| 7 | S1 converter split + S2 registry unification + entry-point loaders | structure | M — **✅ done 2026-06-16** (shared `oneuniverse._registry.Registry` backs all 4 registries; `converter.py` split into `_converter_core/_point/_sightline/_linkback` behind a re-export façade; entry-point group `oneuniverse.survey_loaders`. Plan: `research/2026-06-15-refactor-and-sql-surfacing-plan.md`) |
| 8 | SQL P5–P6 (DuckDB attach, MeasurementSet.to_sql) | feature | S — **✅ done** |
| 9 | S10 twin consolidation; B6–B9 small fixes | debt | S — **B6–B10 ✅ done** (parquet linkback; area-uniform NEST sub-pixel randoms; relative covariance paths; honest `cpu(reference)` device stat; wiener dead guard). **S10 ✅ done 2026-06-16** (verify+validation merged into `twin.metrics`; old modules are silent compat re-exports). Plus SQL surfaced via README + top-level `export_sql` + `scripts/export_to_sql.py`. |

## 7 · Verdict

The architecture is **sound and matches its goals**: the three-layer
data→measure→simulation split is real (enforced, not aspirational), the
partial-access + wrap-in-place storage story is the package's strongest idea
and is implemented twice consistently, and the cosmology-free discipline is
mechanically checked. The debt is *concentrated and tractable*: one monolith
(converter), one duplication cluster (validation estimators ×4), one
copy-paste family (measure builders), and a young `measure/io.py` that this
review caught with three real bugs before any user did. The single biggest
*capability* gap against the owner's goals was the absence of a SQL face for
the two databases — §5 specifies it in full, and its design falls naturally
out of the package's own materialize-vs-wrap philosophy.

*Post-review note: B1, B2, B3, B4, B5 and B11 were fixed (with regression
tests, `test/test_measure_review_bugs.py`) in the commits following this
review; fixing B2 exposed B11. B6–B10 and all structural items (S1–S11) plus
the SQL design (§5) remain open, prioritised in §6.*
