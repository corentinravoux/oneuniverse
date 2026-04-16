# oneuniverse Pillar 1 Stabilisation — Master Roadmap

**Date:** 2026-04-15
**Scope:** Pillar 1 (data + orchestration layer). No Pillar 2/3 interfaces
in this effort.
**Non-negotiable decisions (confirmed by user):**
- No OUF v1 backward compatibility. Bump format version, no migration
  code. Existing databases are rebuilt once against v2.
- Subset-aware dataset loading (`datasets=[...]` arg everywhere).
- Cross-match must be optional per pair; z-type-aware by default.
- ONEUID must have one unambiguous definition; no risk of two ID spaces
  for the same physical objects.
- Performance is vital (real databases are ≥10⁶ rows per dataset).
- Claude has free hand on OUF format design.

## Motivation

Architectural audit (see conversation 2026-04-15) found eight S1/S2
issues:

1. Two parallel cross-match paths produce separate universal-ID spaces
   (`oneuid` vs `WeightedCatalog.universal_id`).
2. Two divergent hydration paths (`load_universal` and `partial_for`).
3. Cross-match ignores `z_type` → catastrophic false-match risk for
   spec × phot pairs.
4. Dynamic class generation via `type()` gives no abstraction value.
5. Manifest is an unschema'd dict with scattered `.get(default)` usage.
6. Three near-duplicate converter functions (POINT/SIGHTLINE/HEALPIX).
7. Non-atomic writes leave half-valid trees on interrupt.
8. Converter never enforces schema at write time.

Plus performance gaps: full-read `pq.read_table` everywhere, no predicate
pushdown, no spatial partitioning, string-dtyped dataset column, no
streaming.

## Design decisions

### OUF 2.0 format (free-hand, approved)

- **Typed manifest** via frozen dataclass, validated on read.
- **Required CORE columns** tightened: `ra, dec, z, z_type, z_err,
  galaxy_id, _original_row_index, _healpix32`.
- **Content hashing** (sha256, truncated) on original files and
  partitions.
- **Atomic writes** — `.tmp → os.replace` for manifest, partitions, and
  the ONEUID index.
- **Schema-enforced conversion** — loader output validated before write.
  Opt-out only in `dev_mode=True`.
- **Per-partition statistics** in manifest (ra/dec/z min/max, n_rows,
  hash) for partition pruning.
- **HEALPix spatial partitioning** (NSIDE=32, 12 288 cells of ~1.8°)
  stored as a hive-partitioned subdirectory layout so
  `pyarrow.dataset` can prune spatially.

### ONEUID (single source of truth)

- Named, versioned indices stored at `{root}/_oneuid/<name>.parquet`
  with a `.manifest.json` recording datasets + their content hashes +
  rules used to build.
- Deterministic assignment: stack datasets in sorted name order,
  connected-component labels remapped in first-occurrence order. Same
  inputs → same IDs bit-identical.
- Subset builds: `db.build_oneuid(datasets=[...], rules=..., name=...)`.
- Audit columns: `sep_arcsec, dz, z_type_a, z_type_b, rule_used` stored
  per pair so users can `df[df.sep_arcsec > x]` post-hoc.
- `WeightedCatalog` no longer runs its own cross-match; it *consumes*
  an `OneuidIndex`. One universal-ID concept, everywhere.

### Cross-match rules (data, not code)

```python
CrossMatchRules(
    default_sky_tol_arcsec=1.0,
    default_dz_tol=5e-3,
    per_ztype_dz_tol={
        ("spec", "spec"): 5e-3,
        ("spec", "phot"): 0.05,
        ("phot", "phot"): 0.10,
        ("spec", "pv"):   5e-3,
    },
    per_pair_overrides={
        ("eboss_qso", "desi_qso"): {"sky_tol_arcsec": 2.0, "dz_tol": 5e-3},
    },
    scale_dz_with_opz=True,
    reject_ztype=[("phot", "phot")],  # opt-out
)
```

### Performance architecture

1. Replace `read_oneuniverse_parquet` with a `pyarrow.dataset`-based
   scanner that accepts a `pyarrow.compute` filter expression.
2. HEALPix partition pruning for cone / skypatch queries.
3. Partition statistics in manifest so the scanner skips files before
   opening them.
4. Streaming hydration (`iter_partial(..., batch_size)`) so peak memory
   is bounded regardless of query size.
5. ONEUID index: categorical `dataset`, int32 `row_index` where safe,
   row-sorted by `(oneuid, dataset)` → contiguous slicing.
6. Zero-copy pyarrow → numpy where dtype permits.

### API shape (target)

```python
# Dataset access
view = db["desi_qso"]                           # DatasetView (lazy)
tbl  = view.scan(columns=[...], filter=...)     # pyarrow.Table
df   = view.read(columns=[...], filter=...)     # pandas

# Subset databases (cheap view)
sub  = db.subset(["desi_qso", "desi_bgs"])

# ONEUID
idx = db.build_oneuid(datasets=[...], rules=rules, name="desi")
idx = db.load_oneuid("desi")
db.list_oneuids()

# Query
q = idx.query
uids  = q.from_cone(ra, dec, r)
df    = q.partial_for(uids, columns=[...])
for batch in q.iter_partial(uids, columns=[...], batch_size=100_000):
    ...

# Weights
wc = WeightedCatalog(oneuid_index=idx, weights={...})
combined = wc.combine(value_col="z", variance_col="z_err_sq",
                     strategy="ivar_average")
```

## Phase breakdown

Each phase is shippable on its own with passing tests.

### Phase 1 — OUF 2.0 manifest + atomic writes + content hashing

**Status:** in progress.
**Detailed plan:** [2026-04-15-phase1-ouf-v2.md](2026-04-15-phase1-ouf-v2.md).

Delivers: typed `Manifest` dataclass; unified `write_ouf_dataset()`
(replacing the three convert functions); atomic writes; sha256 content
hashing on original files and partitions; schema-enforced conversion
(required CORE columns include `z_type`, `z_err`, `_healpix32`);
`read_manifest()` strict validator. Old v1 validation helpers deleted.

### Phase 2 — DatasetView + pyarrow.dataset backend

Delivers: `DatasetView` class replacing `_make_loader_class`;
`scan_ouf()` / `read_ouf()` using `pyarrow.dataset` with predicate
pushdown; `db[name]` returns `DatasetView`; legacy `get_loader` retained
as shim returning a `DatasetView`. Partition statistics used for
pruning.

### Phase 3 — HEALPix spatial partitioning

Delivers: converter writes POINT datasets as
`{ouf}/data/healpix32={cell}/part_*.parquet`; cone/skypatch filters
translate to HEALPix cell lists and push through `pyarrow.dataset`.
Expected 10–100× speedup on spatial queries.

### Phase 4 — Unified ONEUID (z-type rules, subsets, named indices)

Delivers: `CrossMatchRules`, `OneuidIndex` v2 with audit columns and
manifest; `build_oneuid(datasets=, rules=, name=)`; `load_oneuid(name)`;
`list_oneuids()`; `OneuidIndex.restrict_to(datasets)` in-memory subset;
`WeightedCatalog` consumes an index. Deletes `WeightedCatalog.crossmatch()`
self-builder and the `load_universal`/`partial_for` duplication.

### Phase 5 — Streaming hydration

Delivers: `OneuidQuery.iter_partial(uids, columns, batch_size)`
generator. `partial_for` becomes a list-collecting wrapper. Keeps peak
memory bounded for huge queries.

### Phase 6 — Housekeeping + `weight/` → `combine/` redesign

Runs **after** the `data/` sub-package (Phases 1–5) is finished and
stable. Two concerns: leftover data-layer cleanups, and a focused
redesign of the `weight/` sub-package.

**Data-layer cleanups**

- `DatasetEntry` dataclass consolidating the three parallel dicts in
  `OneuniverseDatabase` (`_loaders`, `_manifests`, `_paths`).
- Categorical dtype on ONEUID `dataset` column; `int32 row_index`
  where safe.
- Per-database `data_root` (remove module-level mutable state from
  `_config.py`).
- Registry frozen after import.

**`weight/` → `combine/` redesign**

`weight/` is renamed and restructured so that a *principled weighting
scheme is the foundation of multi-survey combination* — every user
gets a sensible default without hand-picking weights per survey.

Rationale:

1. After Phase 4, `weight/crossmatch.py` is dead code (ONEUID owns
   cross-matching). What remains is weights + combination strategies.
2. "Weight" as a package name undersells the job: the package is
   about combining same-object measurements from multiple surveys
   *through a principled weighting scheme*. The new name `combine`
   names the action; the weighting scheme stays front-and-centre
   inside.

Target layout:

```
oneuniverse/combine/
├── __init__.py
├── weights/                  # weighting schemes — the backbone
│   ├── __init__.py
│   ├── base.py               # Weight, ProductWeight
│   ├── ivar.py               # InverseVarianceWeight (canonical default)
│   ├── fkp.py                # FKPWeight
│   ├── quality.py            # QualityMaskWeight, ColumnWeight,
│   │                         # ConstantWeight
│   └── registry.py           # default_weight_for(survey, ztype)
├── strategies.py             # ivar_average, hyperparameter, …
├── catalog.py                # WeightedCatalog (name kept)
└── measurements.py           # CombinedMeasurements result container
```

Deliverables:

- Rename package `oneuniverse.weight` → `oneuniverse.combine`;
  delete `crossmatch.py` (superseded by Phase 4 ONEUID).
- Split `base.py` by weight family (one file per concept); keep
  `WeightedCatalog` class name — weighting remains the user-facing
  contract.
- New `weights/registry.py` exposing `default_weight_for(survey,
  ztype)` returning the canonical baseline weight per
  `(survey_type, ztype)` pair. So a new user writes:

  ```python
  from oneuniverse.combine import default_weight_for, WeightedCatalog
  wc = WeightedCatalog(oneuid_index=idx)
  for name in db:
      wc.add_weight(name, default_weight_for(db.get_config(name).survey_type,
                                             ztype="spec"))
  combined = wc.combine(value_col="z", variance_col="z_err",
                        strategy="ivar_average")
  ```

  …instead of hand-picking `InverseVarianceWeight("z_err")` per
  survey. Overrides remain trivial.
- `WeightedCatalog` constructor accepts an `OneuidIndex` (no self-
  built crossmatch; enforced after Phase 4).
- Rename `CombinedMeasurements` → keep as-is (already well-named).
- Update `oneuniverse/__init__.py` exports; remove stale
  `oneuniverse.weight` top-level re-exports.

Open design questions to resolve at plan-writing time:

- Should `default_weight_for` dispatch on `survey_type` alone, on
  `(survey_type, ztype)`, or be data-driven (read from a small
  registry table)? Likely the middle option — same survey with
  `ztype="spec"` vs `"phot"` usually wants different weights.
- Does `WeightedCatalog` own a `default_scheme` so
  `wc.fill_defaults(db)` auto-populates weights for every dataset it
  will consume? (Probably yes — ergonomic win.)
- Which of the existing `crossmatch.py` pieces, if any, need to
  survive as utilities? (Expected: none — ONEUID is the sole path.)

Tests: full parity with existing `test_weight.py` (28 cases) plus
new tests for `default_weight_for` and for `WeightedCatalog` failing
fast when given raw DataFrames instead of an `OneuidIndex`.

## Out of scope

- Pillar 2 cross-correlation interfaces (not implementing estimators).
- Pillar 3 simulation interfaces (not implementing forward models).
- New survey loaders (orthogonal work; existing loaders must keep working).

## Ground rule for implementation

Each phase:
1. Writes a detailed plan using `superpowers:writing-plans`.
2. Executes test-first via `superpowers:executing-plans` or
   `superpowers:subagent-driven-development`.
3. Commits incrementally; tests green at every commit.
4. Does not begin until the prior phase is merged.
