# oneuniverse 1.0 — refactor design

*2026-11-12. Status: approved by the owner on 2026-11-12 (decisions in §8). Supersedes the phase-by-phase
roadmaps in this directory as the description of what the package is.*

Audit behind this design:
`~/Software/claude_config/reports/code-analysis/2026-11-12_oneuniverse-refactor-audit/`
(two full code readings + four hand-verified bugs).

---

## 1. Intent

oneuniverse is the **data and orchestration substrate** for the owner's
three-pillar cosmology stack. Nothing about that intent changes:

- **Pillar 1 — data.** Every survey catalog into one on-disk format (OUF),
  queried lazily and partially; one identity across surveys (ONEUID);
  sub-object links; weights. **No cosmology.**
- **Pillar 1→2 — measure.** A cosmology-free, analysis-ready handoff
  (`MeasurementSet`) that external estimators (flip, pycorr, picca, …) consume.
- **Pillar 3 — simulation.** A partial-access store for simulation products and
  the orchestration (catalog, lineage, requests) of resimulation. The physics
  is pluggable; the package ships only a reference engine.
- **twin.** The contracts that couple data and simulation.

What changes is the implementation. The current code (17.5 kLOC, 187 modules)
was grown in ~30 increments by a weaker model. The architecture is right; the
code is accreted, carries real bugs in its core feature (the ONEUID index
returns the wrong objects), duplicates itself, and exposes an API that is
neither minimal nor learnable. The next phase of the project is real data in
notebooks, which needs a package one can trust and read.

**This is a rewrite under the existing name and concepts, not a patch.** Every
module is re-authored; the good designs are carried over deliberately; the
accretion is not.

## 2. Goals

1. A user reads one page and can: load a survey, convert it, query a cone,
   cross-match two surveys, weight, and build a `MeasurementSet`.
2. Correct by construction: one meaning per identifier, atomic on-disk
   publishes, no silent fallbacks, no swallowed exceptions.
3. Bounded memory everywhere: streaming conversion, vectorised pruning,
   pair-wise blocked cross-match, hydration by `take`.
4. Half the code, same real capability. Target ≤ 9 kLOC source, ≤ 80 modules,
   test suite < 60 s single-process (slow physics tests marked and opt-in).
5. The hard rules enforced by CI, not by docstrings.

## 3. Non-goals and anti-goals

- Not adding survey loaders beyond eBOSS DR16Q, DESI DR1 QSO and the test
  fixture. New loaders come with real data, after this refactor.
- Not implementing estimators, forward models, samplers (unchanged scope rule).
- Not keeping on-disk compatibility with OUF 2.x or OUF-Sim 0.1. The two real
  datasets and the demo store are rebuilt once.
- Not preserving the current import paths. 1.0 is a breaking release; the old
  API has no external consumers (only this repo's scripts and notebooks).
- Anti-goal: "generality" with no consumer. Every optional slot, spec,
  geometry or mode in 1.0 has at least one reader in the package or a real
  dataset behind it.

## 4. Hard rules (unchanged, now enforced)

| Rule | Enforcement |
|---|---|
| No cosmology in `data`, `combine`, `measure` | AST + grep guard test: no `H0`, `Omega`, `h=0.6…`, `comoving`, `astropy.cosmology` in those trees; `MeasurementSet` validation rejects cosmology-derived column names. |
| `simulation` never imports `data`, `combine`, `measure` | AST guard (kept). |
| `measure` never imports `simulation` | AST guard (kept). |
| Substrate never imports the reference engine | New AST guard: `simulation/store`, `simulation/db`, `twin/engine` import nothing from `reference/`. |
| Partial access is the only read path for simulation products | No public reader without a selector; whole-box reads need an explicit `scan_full()`. |

## 5. Package layout

```
oneuniverse/
├── __init__.py              version, top-level re-exports (≈ 15 names)
├── _core/                   shared internals, no science
│   ├── errors.py            OneuniverseError, FormatError, SchemaError, NotFoundError
│   ├── healpix.py           radec_to_pix, coarsen, cone_pixels, patch_pixels (single copy)
│   ├── geometry.py          Cone, SkyPatch (RA-wrap aware), ZRange, Cube (shared by all layers)
│   ├── fs.py                atomic_write_json/bytes, atomic_dir() context manager, hash_file
│   └── registry.py          Registry (kept from 2026-06-16)
├── data/                    Pillar 1
│   ├── schema.py            geometry column contracts + z_type registry (one source of truth)
│   ├── manifest.py          OUF 3.0 Manifest (small core + extensions) + partition index I/O
│   ├── writer.py            write_dataset(batches) — one writer, geometry strategies
│   ├── view.py              DatasetView — pyarrow dataset, vectorised partition pruning
│   ├── database.py          Database — discovery, views, sidecar access, as_of
│   ├── sidecar.py           VersionedSidecar — one crash-safe versioned store for oneuid + links
│   ├── crossmatch.py        CrossMatchRules + pair-wise blocked KD-tree matcher
│   ├── oneuid.py            OneuidIndex (build/load) + OneuidQuery (select, hydrate by take)
│   ├── links.py             SubobjectRules, SubobjectLinks, build/load, chain walk
│   ├── pdf.py               PdfSpec (interp | sample | quant) + ProbabilisticRedshift
│   ├── sql.py               connect(db_root) → DuckDB connection with views over the OUF parquet (optional extra)
│   └── loaders/
│       ├── base.py          SurveyLoader ABC + FitsCatalogLoader (declarative, batch-streaming)
│       ├── eboss_qso.py     real
│       ├── desi_qso.py      real
│       └── registry.py      register / get_loader / list_surveys (explicit list, entry points)
├── combine/
│   ├── weights.py           Weight ABC, Column, InverseVariance, FKP, HealpixMap, Shear, QualityMask, Product
│   ├── defaults.py          default_weight_for(survey_kind, z_type)
│   └── catalog.py           WeightedCatalog (reads through DatasetView, vectorised combine)
├── measure/
│   ├── product.py           Product (catalog/randoms/window/nz/extras) + FieldMap + Sightline
│   ├── measurement_set.py   MeasurementSet (validation in __post_init__), MeasurementSpec
│   ├── pipeline.py          build_pointset(view, ...) — the one spine: select, weight, window, regions, nz, randoms
│   ├── presets.py           galaxy_clustering, cosmic_shear, three_by_two — thin presets over build_pointset
│   ├── window.py / nz.py / randoms.py / regions.py / photoz.py
│   └── io.py                save / load: generic by type, versioned, atomic
├── simulation/              Pillar 3 substrate only
│   ├── manifest.py          OUFSimManifest (JSON, typed, version-checked on read)
│   ├── specs.py             CosmologySpec, UnitFrameSpec, ProvenanceSpec, ProductDecl, RegionSpec, SimulationRequest
│   ├── execution.py         ExecutionPlan (streamed | mpi); no GPU stub
│   ├── store/               index.py, write.py (streaming per product), read.py (SimStore, periodic-aware), native.py (adapters), build.py (build_store, NativeProduct)
│   ├── converter.py         SimConverter ABC + registry
│   ├── sql.py               connect(store_root) → DuckDB views over the chunk index + catalog products
│   └── db.py                SimDatabase — catalog + lineage + requests; dispatch(callable) writes a child store
├── reference/               the reference engine (one plugin, per the 2026-06-02 ADR)
│   ├── linear.py            EH98 P(k), growth, Gaussian field, Zel'dovich (fixed units), toy halos
│   ├── pm.py                CIC + FFT Poisson + KDK leapfrog (bincount deposit)
│   ├── resim.py             buffered sub-box resimulation + TreePM scale split (periodic-wrap fixed)
│   ├── reconstruct.py       Wiener filter (mask-aware mean), Hoffman–Ribak CR (independent seed)
│   ├── fieldstats.py        one k-grid, one binning, P(k), r(k), transfer
│   ├── mock.py              tracer mock (lognormal), masks, mock challenge
│   ├── native.py            native layout writer + packed layout writer + the two SimConverters
│   └── engines.py           LinearForward, PMForward, Wiener, Constrained on the twin contract
└── twin/
    ├── engine.py            Observation, ProductBundle, ForwardEngine, ReconstructionEngine, registry
    ├── observe.py           observe_from_view (DatasetView or Product → Observation)
    └── store_boundary.py    run_engine_to_store
```

Deleted outright (with their tests): the 7 loader stubs; `config_loader`,
`_dataset_entry`, `chain`, `moc`, `subobject_map`, `_linkback`;
`cube_spec`, `gwskymap_spec`, `classification_pdf`, `tomographic_nz`;
`DataGeometry.CUBE/GW_SKYMAP/HEALPIX`; the SQLite export in `data/sql.py`,
`oufsim/sql.py` and `MeasurementSet.to_sql` (replaced by §6.11); `combine/strategies` hyperparameter/unit_mean,
`PdfMeanRedshiftWeight`, `PipBitweightWeight`, BOSS named wrappers;
`measure/{lightcurve,links,lya,mapcross,pvsn,distances,covariance,shapes,tomography}`;
`simulation/linear/{amr,gr_fields,ic,phase_space,tree}`,
`simulation/resim/{cola,farfield,ic_extract,merge,zoom}`,
`simulation/oufsim/{bench,scale_bench,view.py}`, the YAML manifest path;
`twin/{engines_extra,mock_survey,verify,validation,mock_observe,mock_challenge}`
(folded into `reference/`); `notebooks/` (all six + builder); `REVIEW.md`;
`scripts/` except one `convert_surveys.py` and one `build_reference_store.py`.

Kept as design (re-authored): OUF layout (manifest + `_index.parquet` +
HEALPix-partitioned parquet), `DatasetView` pushdown, adaptive partition
NSIDE, bitemporal `as_of` on sidecars, Weight ABC with `*` composition,
OUF-Sim chunk index + wrap-in-place projection + `ExecutionPlan`, the
`SimConverter`/`NativeReaderAdapter` seams, the engine contracts, the TreePM
split result.

## 6. Decisions

### 6.1 Python and dependencies (my call)
- `requires-python >= 3.11`; CI matrix 3.11–3.14. Modern typing (`X | None`,
  `list[str]`), `dataclass(slots=True, frozen=True)`, `match` where it reads.
- Hard deps: numpy, pyarrow, pandas, healpy, scipy, astropy (FITS only), pyyaml
  dropped (JSON everywhere). Optional: `fitsio` (faster FITS), `mpi4py`.
- pandas is the user-facing frame (`view.read()` returns a DataFrame).
  Everything internal (writer, pruning, cross-match, hydration, weights) runs
  on pyarrow tables and numpy. Loaders yield `pa.RecordBatch`.

### 6.2 OUF 3.0 (my call, within the owner's "free hand on format")
Layout per dataset:
```
<dataset>/
├── manifest.json        identity only; stable across rewrites of partitions
├── _index.parquet       one row per partition: path, n_rows, cell, ra/dec/z/t min-max, extra ranges
└── data/cell=00042/part.parquet   (cell numbering at the NSIDE recorded in the manifest)
```
Manifest core: `format_version`, `geometry` (point | sightline | lightcurve),
`survey` {name, kind, description, reference}, `loader` {name, version,
kwargs}, `created_utc`, `schema` (serialised Arrow schema, authoritative),
`partitioning` {nside, nest}, `original_files` [{path, sha256, n_rows}],
`n_rows`, `validity` {valid_from, valid_to, version}, `extensions`: a dict of
named, versioned blocks owned by the module that reads them (`coordinate`,
`spectrum`, `temporal`, `pdf`). An extension nobody reads does not exist.

Core columns (POINT): `ra, dec, z, z_type, z_err, object_id, survey,
source_row, hpix32`. `object_id` is stable (`xxhash64(survey, source_row)`),
`source_row` is the true row in the original file (loaders must preserve it
through their cuts), `survey` and `z_type` are dictionary-encoded.

No reader for 2.x. `convert_surveys.py` rebuilds the two real datasets.

### 6.3 Identity (fixes the verified bug)
`source_row` is the single row identity. ONEUID and link tables carry
`(dataset, source_row)` plus `(partition_id, offset)` computed at build time,
so hydration is a `take` per partition, never a scan. `WeightedCatalog`,
`OneuidQuery`, `measure` all go through the same accessor.

### 6.4 Cross-match (my call)
Pair-wise between surveys, never a self-join: per survey pair, KD-tree on unit
vectors, blocked by coarse HEALPix cell with a neighbour ring, so memory is
bounded by the densest cell pair. Rules as integer-coded `z_type` lookup
matrices (tolerance and accept). Transitive closure across pairs, with the
same-survey merge documented and testable (an option `allow_same_survey_merge`,
default False, splits components that would merge two objects of one survey).
`attribute_filters` dropped (no consumer); a column-predicate hook can come
back with a consumer.

### 6.5 Loaders (my call)
Declarative. A real loader is ≈ 60 lines: metadata, one native→standard
column dict, a row filter, a `z_type`, observational metadata, an optional
`derive(table)` hook. The base class streams the FITS in batches (astropy or
fitsio), applies the filter, records `source_row`, stamps the core columns,
computes `hpix32`, yields batches. The dummy survey becomes
`test/fixtures/synthetic_catalog.py` (no H0). Planned surveys are a table in
the docs, not registered classes.

### 6.6 Sidecars and atomicity (my call)
One `VersionedSidecar(root, kind, name)` for ONEUID and links: write the new
version to `<name>.tmp/`, archive the live one as `<name>@<ISO>Z`, rename.
Readers resolve `as_of`. `write_dataset` writes into `<dataset>.tmp/` and
renames on success; the previous dataset is never touched before the new one
is complete.

### 6.7 `MeasurementSet` (my call)
One `Product` base (`catalog` with fixed core columns `ra, dec, z, weight,
region_id`; optional `randoms`, `window`, `nz`; `extras: dict[str,
ndarray | DataFrame]`; `provenance: dict`), two subclasses `FieldMap`
(values, mask, nside) and `Sightline` (ids, ra, dec, wavelength, delta,
weight). `MeasurementSet(products, spec, regions)` with the region scheme
carried once. Validation in `__post_init__` (lengths, shared NSIDE, no
cosmology columns) so it cannot be skipped. `build_pointset` is the one
spine; probe names are presets. A survey mask is accepted and used by default
when given; the data-inferred footprint is an explicit fallback that is
recorded in provenance. Save/load is generic by type, versioned, atomic,
drops nothing (an unknown type refuses to save).

### 6.8 Simulation substrate (my call)
`SimStore` reads only through selectors; a cube straddling the periodic
boundary is split internally. Writes stream per product under the
`ExecutionPlan` budget (`np.load(mmap_mode="r")` + slices). One typed JSON
manifest, validated on read, carrying what the store writes. `SimDatabase`
holds catalog + lineage + requests; `dispatch(request, engine)` takes a
callable and writes the child through `ingest_field`, so the lineage it
records exists on disk. MPI stays (`mpi4py` optional, exercised at size 1 in
CI); the GPU mode and `cudf` branch are removed until there is a GPU path.

### 6.9 Reference engine (my call, owner may veto — see §8)
Everything that computes physics moves to `oneuniverse.reference`, with the
four verified defects fixed and tests that use an ensemble of seeds against
stated tolerances rather than one-seed thresholds. Dead ends (COLA attempts,
AMR, gr_fields, phase space, merger tree, zoom, merge, far-field) are deleted;
their history stays in git and in `research/`.

### 6.10 Tests, docs, repo (my call)
- `test/` mirrors the package (`test/data/test_writer.py`, …). No phase names.
  Diagnostic figures move to `examples/diagnostics/*.py` writing to an
  ignored directory; nothing binary is committed.
- `plans/` and `REVIEW.md` move to `docs/history/` (kept, not maintained).
  `research/` stays. `CLAUDE.md` is rewritten as a 100-line guide with no
  changelog. README and the MkDocs site are rewritten against the new API.
- Version `1.0.0`. Branch `refactor-v1`, merged to `main` at the end.
- Commit convention: conventional commits, no phase prefixes.

### 6.11 SQL face (owner decision, 2026-11-12)
Parquet is the storage; DuckDB is the query engine. `data.sql.connect(root)`
returns an in-process DuckDB connection whose views are built on attach from
the manifests and `_index.parquet` files: `datasets`, `partitions`, `oneuid`,
`links`, `versions`, and one view per dataset over its parquet partitions
(hive partition column `cell`). Nothing is copied; the Python API and the SQL
face read the same files. Array columns (PDFs, Lyα δ) are native DuckDB lists.
`simulation.sql.connect(root)` does the same over the chunk index and the
catalog-sized products. `duckdb` is an optional extra `[sql]`; no core module
imports it.

## 7. Top-level API (what the one page shows)

```python
from oneuniverse import data, combine, measure

# Pillar 1
data.list_surveys()                       # {"eboss_qso": SurveyMeta, "desi_qso": SurveyMeta}
data.convert("eboss_qso", raw=..., out=db_root / "eboss_qso")
db   = data.Database(db_root)
view = db["eboss_qso"]                    # DatasetView
df   = view.read(cone=data.Cone(185, 15, 10), z=(0.8, 2.2), columns=["ra", "dec", "z"])
idx  = db.build_oneuid(["eboss_qso", "desi_qso"], rules=data.CrossMatchRules(sky_tol_arcsec=1), name="qso")
q    = db.oneuid_query("qso")
both = q.hydrate(q.select(cone=...), columns=[...])

# combine
wc = combine.WeightedCatalog.from_oneuid(idx, db).fill_defaults()
wc.add_weight("desi_qso", combine.FKPWeight(nbar=..., P0=3e4))

# measure
ms = measure.galaxy_clustering(view, z=(0.8, 2.2), weights=..., nz_edges=..., mask=..., randoms="generate")
measure.save(ms, path);  ms = measure.load(path)
```

## 8. Owner decisions (2026-11-12)

1. **Reference engine** lives inside the package as `oneuniverse.reference`.
2. **Dead-end physics** (COLA attempts, AMR, gr_fields, phase space, merger
   tree, zoom, merge, far-field) is deleted; history stays in git and `research/`.
3. **Bitemporal `as_of`** on datasets and sidecars is kept.
4. **SIGHTLINE** geometry is kept in OUF 3.0, with a wavelength grid.
5. **SQL face**: DuckDB views over the OUF parquet (and over the simulation
   chunk index), optional extra `[sql]`; the SQLite copy export is deleted.
   See §6.11.

## 9. Left to the builder

Internal module boundaries inside each subpackage; exact dataclass field names
beyond the core columns above; the batch size heuristics; which FITS reader
backs `FitsCatalogLoader`; test fixture shapes; MkDocs page structure;
`z_type` integer coding; `_index.parquet` extra-range columns; the
cosmology-guard allow-list (the guard scans identifiers and string literals
in code via AST, not docstrings or comments, so `Omega_b` as a survey column
name is allow-listed explicitly, not by weakening the pattern).

## 9b. Clarifications from the builder check

- **`measure` is in scope and lives inside oneuniverse.** Owner decision of
  2026-06-05 (reversing the 2026-05-29 `onemeasure` plan). The old `CLAUDE.md`
  still carries both statements; the rewrite removes the stale one.
- **The verified defects to fix in R6**, with their acceptance tests:
  1. Zel'dovich velocity: `v = a·100·E(a)·f·ψ` km/s for ψ in Mpc/h. Test:
     velocity divergence vs `−a H f δ` on a linear field, 5 seeds, ratio
     within 2 % at k < 0.1 h/Mpc.
  2. Resimulation buffer selection without periodic wrap. Test: a target with
     `target_lo < buffer` yields exactly `n_buf³` particles and the same inner
     correlation as a target in the box centre (5 seeds, |Δr| < 0.02).
  3. Constrained-realisation seed leak. Test: with the truth at seed s, the CR
     at any seed s' ≠ s differs from the truth at small scales
     (r(k > k_cut) < 0.5) while the WF mean is unchanged; CR at the truth seed
     is rejected by the API (seed must be given, no default).
  4. Wiener ignores the mask. Test: masked mock with 50 % footprint recovers
     r(k) inside the footprint within 0.05 of the unmasked run (5 seeds).
  Plus, in R2: ONEUID row identity regression (shuffled multi-cell synthetic
  catalog → build → hydrate → `object_id` equality 100 %).
  Ensemble tests use fixed seed lists and compare the ensemble mean against
  the stated tolerance; they are marked `slow` and run in CI nightly, not on
  every push.
- **Real data is not on the development machine.** R1/R2 validate loaders
  against synthetic FITS fixtures that reproduce the documented column
  layouts (`test/fixtures/desi_dr1_like.py` already exists; an eBOSS
  equivalent is written from `research/eboss_dr16q_research.md`). Real-data
  tests stay env-gated (`ONEUNIVERSE_DATA_ROOT`) and run on the laptop. The
  "two 1 M fixtures" of R2 are synthetic.
- **No parity suite against the old package.** The new tests are the
  acceptance. The old tree stays on `main` until `refactor-v1` merges, so an
  ad-hoc comparison on the synthetic fixtures is always possible.
- **Tree handling.** In place on `refactor-v1`. Each step deletes the old
  subpackage (and its tests) at the start and builds the new one; the other
  old subpackages stay importable until their own step. R0 adds `_core` and
  the guards without deleting anything.
- **Targets (9 kLOC, 80 modules) are guides.** The default `pytest` run
  under 60 s is a gate.
- **CI is GitHub Actions** (existing `tests.yml` / `docs.yml`, matrix updated).

## 10. Execution outline (detailed plan follows approval)

| Step | Scope | Exit criterion |
|---|---|---|
| R0 | branch, `_core`, guards, pyproject, CI skeleton | guards green on empty package |
| R1 | `data`: schema, manifest, writer, view, loaders | eBOSS + DESI fixtures convert and query; streaming conversion bounded |
| R2 | `data`: sidecar, crossmatch, oneuid, links, database | row-identity regression test green; cross-match of two 1 M fixtures < 1 min |
| R3 | `combine` | weights + WeightedCatalog on R2 output |
| R4 | `measure` | clustering + shear presets; save/load round-trip drops nothing |
| R5 | `simulation` substrate | store ⊥ reference guard; periodic read; manifest round-trip |
| R6 | `reference` + `twin` | four physics fixes with ensemble tests; resim loop end-to-end |
| R7 | SQL face (data + simulation) | DuckDB views answer the cone / join / n(z) queries of the old notebook 02 on the fixtures |
| R8 | tests layout, docs, CLAUDE.md, README, examples, 1.0.0 | suite < 60 s; mkdocs strict build |

Then: real data on disk, the two datasets rebuilt, notebooks written against
1.0 (separate plan).
