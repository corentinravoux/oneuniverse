# CLAUDE.md — oneuniverse (Pillar 1)

Package-scoped guide. The repo-level `Python/CLAUDE.md` covers other
packages (mainly `flip`).

## Mission

oneuniverse is **all of Pillar 1**: the data + orchestration layer of
the three-pillar cosmology stack. Pillar 1 ends when a
`MeasurementSet` is handed to an external estimator (Pillar 2) or to
`oneuniverse.simulation` (Pillar 3).

- **Pillar 1 (here):** `oneuniverse.data` (ingest, schema, manifest,
  ONEUID, sub-object, DatasetView) → `oneuniverse.combine` (weights,
  WeightedCatalog, combination) → `oneuniverse.measure` (MeasurementSet,
  random catalogs, windows, jackknife regions, n(z)). **No cosmology,
  no estimators, no forward models.**
- **Pillar 2:** estimators + likelihoods consumed by external tools
  (`flip`, `pycorr`, `picca`, future `onecorr`). Cosmology enters
  here.
- **Pillar 3:** `oneuniverse.simulation` (planned) — constrained
  forward modelling, digital twin, mini-sim zoom-ins.

See [`plans/2026-05-28-pillar1-data-combine-measure.md`](plans/2026-05-28-pillar1-data-combine-measure.md),
[`plans/2026-05-28-pillar2-external-interfaces.md`](plans/2026-05-28-pillar2-external-interfaces.md),
[`plans/2026-05-28-pillar3-simulation-digital-twin.md`](plans/2026-05-28-pillar3-simulation-digital-twin.md)
for the large-scope roadmaps.

**Cosmology rule.** No H₀ / Ωₘ / distance models in Pillar 1.
Heliocentric / CMB / vacuum-air etc. are observational metadata
(per-column `frame` / `wavelength_convention`), not cosmological
choices. Conversion to comoving distance happens in Pillar 2/3 at
call site.

## Package layout

- `oneuniverse/data/` — schema, manifest, converter, DatasetView,
  ONEUID, sub-object links, temporal validity.
- `oneuniverse/combine/` — `WeightedCatalog` + Weight ABC +
  primitives (FKP, IVar, HealpixMap, PDF, BOSS combiner). The
  deprecated `oneuniverse.weight` shim was deleted 2026-05-28.
- `oneuniverse/measure/` — **planned (Phase 21)**: MeasurementSet,
  random catalogs, window functions, HEALPix jackknife regions,
  n(z) builders, multi-tracer bundling, downstream adapters.
- `oneuniverse/data/surveys/` — registered loaders. Add new ones with
  `@register class FooLoader(BaseSurveyLoader)`.
- `plans/` — phase-by-phase + pillar-level roadmaps. Stabilisation
  (Phases 1–15) done; generalisation (Phases 16–23) planned.
- `docs/` — Sphinx scaffold (`make html` from `docs/`).
- `research/` — topical references + design analyses. Two new docs
  from 2026-05-28: [`research/survey_landscape_review.md`](research/survey_landscape_review.md)
  (full cosmology survey landscape) and
  [`research/schema_generalisation_audit.md`](research/schema_generalisation_audit.md)
  (gap analysis + Phase 16–21 roadmap). Consult before scoping any
  new loader or schema change.
- `test/` — pytest suite (~3:30 wall-clock, 364+ tests).

## OUF 2.2 (format on disk)

Each converted dataset lives at:

    {survey_path}/oneuniverse/
    ├── manifest.json
    ├── data/healpix32=00042/part_0000.parquet
    ├── data/healpix32=00043/part_0000.parquet
    └── ...

`manifest.json` is the typed `Manifest` dataclass — see
[`oneuniverse/data/manifest.py`](oneuniverse/data/manifest.py).
Sub-specs: `PartitioningSpec` (NSIDE may be coarsened by Phase 12 F3 —
read it from the manifest, never hardcode), optional `TemporalSpec`,
`DatasetValidity`, `PdfSpec`, `CoordinateSpec` (Phase 16 — frame /
epoch / PM-parallax availability), `SpectrumSpec` (Phase 16 — vacuum
vs air, log-binning, rest-frame state, λ-unit; SIGHTLINE only).

CORE columns (every POINT dataset): `ra, dec, z, z_type, z_err,
galaxy_id, survey_id, _original_row_index, _healpix32`.

`Z_TYPE_REGISTRY = {"spec", "phot", "phot_pdf", "pv", "none", …}` —
extensible at runtime via
`oneuniverse.data.ztypes.register_z_type(name)`. The converter
validates every chunk's `z_type` against the registry and stamps
`Manifest.observed_z_types` automatically (Phase 16).

## Bitemporal ONEUID / sub-object

- `database.build_oneuid(datasets, rules, name)` writes
  `{root}/_oneuid/<name>.parquet` + `<name>.manifest.json`.
- `database.build_subobject_links(rules, parent_datasets, child_datasets, name)`
  writes `{root}/_subobject/<name>.parquet` + `<name>.manifest.json`.
- Both auto-archive prior versions on rebuild as `<name>__{ISO8601Z}`.
- `database.as_of(when)` returns a filtered clone;
  `load_oneuid(name, as_of=...)` resolves the right archived version.

## Weights

`WeightedCatalog.from_oneuid(index, database).fill_defaults(db,
z_type="spec")` is the canonical entry point. Per-survey custom weights
via `wc.add_weight(survey, Weight(...))`. Compose with `*`
(`ProductWeight`). Public registration:
`oneuniverse.combine.weights.register_default(survey_type, z_type, factory)`.

Available primitives:
- `InverseVarianceWeight(column)` — 1/σ².
- `FKPWeight(nbar, P0, z_column)` — 1/(1 + n̄P₀).
- `ColumnWeight(column)` — pass-through.
- `ConstantWeight(value)` — uniform.
- `QualityMaskWeight(column, op, threshold)` — binary mask.
- `HealpixMapWeight(nside, map_array, nest, …)` — angular map lookup.
- `PdfWidthIVarWeight(z_pdf_std)`, `PdfMeanRedshiftWeight(z_pdf_mean)`.
- `FiberCollisionWeight/ZFailureWeight/CompletenessWeight` — BOSS-style
  named wrappers around `ColumnWeight`.
- `boss_total_weight(w_sys, w_cp, w_noz, w_fkp)` — Reid 2016 combiner.

## Photo-z PDFs

A `Manifest.pdf_spec: Optional[PdfSpec]` declares the on-disk PDF
parameterisation. Three modes: `interp` (p(z) on a grid), `quant` (z(q)
at common quantile levels), `mixmod` (Gaussian-mixture components).
Read with `DatasetView.load_pdf() -> ProbabilisticRedshift`. PDF
columns stored as `pa.FixedSizeList[float32, n_components]` via the
internal `_chunk_to_table(chunk, pdf_spec)` helper.

## Phase status

See [`plans/README.md`](plans/README.md).
- Phases 1–15 complete by 2026-05-22 — stabilisation done.
- 2026-05-28: deprecated `oneuniverse.weight` shim deleted; 365/365
  green. Three-pillar structure formalised; Pillar 1 generalisation
  Phases 16–23 planned (driven by the survey-landscape +
  schema-audit research docs).
- **Phase 16+** = observational metadata expansion + variable-length
  columns + PDF polymorphism + shear + map-based ONEUID +
  `oneuniverse.measure` (MeasurementSet contract). Real-survey
  loader writes (rolled-up Phase 23) land after the schema phases.

## Test conventions

- `test/fixtures/` holds factory functions for synthetic DR1 QSO,
  photo-z PDFs, HEALPix maps. Use them in new tests — do **not**
  ship binary test fixtures.
- `test/test_output/*.png` are diagnostic figures, committed for
  inspection. Phase 15 added size + dimension regression checks.
- `eboss_default_df` session fixture shares one ~31s eBOSS load
  across tests on machines with the DR16Q data (Phase 14 T2).

## Things that bite

- `convert_survey` resolves `data_root` from kwarg → env → None.
  **No module-level state** (Phase 12 D1). `set_data_root` /
  `get_data_root` were removed.
- `DatasetView._resolve_cells` reads partition NSIDE from
  `manifest.partitioning.extra["nside"]` (Phase 12 D5). Do not
  hardcode `HEALPIX_PARTITION_NSIDE` in new code.
- `_chunk_to_table(chunk, pdf_spec)` is the single path for
  DataFrame → pa.Table in the converter; route any new
  list-column work through it.
- `convert_survey(loader=<instance>, output_dir=...)` accepts an
  inline loader without `@register` (Phase 12 D3). Useful for
  synthetic tests; no need to fall back to `write_ouf_dataset`.
- POINT writer auto-coarsens partition NSIDE for small catalogs
  (Phase 12 F3). Pin with `partition_nside=32` if a test asserts
  the canonical Phase-3 layout.

## Commit conventions

- `phase{N}/{TaskID}: ...` for in-phase work (e.g. `phase15/T3:`).
- `phase{N}: close-out — ...` for plan-README + memory updates.
- `docs(plans): ...` for plan-file authoring.
- `fix(phase{N}/F{n}): ...` for fragility fixes (Phase 9 idiom).
