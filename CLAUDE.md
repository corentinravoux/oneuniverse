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
  WeightedCatalog, combination). **No cosmology, no estimators, no
  forward models, no `MeasurementSet` (that's Pillar 2).**
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
- `oneuniverse/measure/` — **moved to Pillar 2 (2026-05-29).** A
  separate `onemeasure` package will own `MeasurementSet` + builders
  + adapters. `oneuniverse` produces typed parquet artefacts on
  disk; `onemeasure` reads them.
- `oneuniverse/simulation/` — **Pillar 3 (OUF-Sim)**, standalone.
  Types for the simulation storage + orchestration substrate:
  `OUFSimManifest`, `ExecutionPlan`/`BackendCapabilities` (optimisation
  substrate), `SimConverter` ABC + registry, `RegionSpec`,
  `SimulationRequest`, `Cube`/`Cone`/`SkyPatch` selectors,
  `CosmologySpec`/`UnitFrameSpec`/`ProvenanceSpec`. **Zero imports**
  from `oneuniverse.data` / `combine` (guarded by
  `test_sim_no_pillar1_imports.py`). Real-format backends land in
  `future`. See [[pillar3-partial-access-and-minimal-deps]].
- `oneuniverse/simulation/linear/` — pure-numpy **dummy linear
  simulation** (Eisenstein–Hu P(k), growth D(z), Gaussian field /
  voxel, Zel'dovich particles, toy halos, toy HEALPix lightcone).
  The synthetic source used to build + test the OUF-Sim machinery
  before any real backend. `generate_linear_sim(out, cosmo,
  box_size=, n_grid=, redshifts=, seed=)` writes a native layout
  (`config.yaml` + per-z `field.npy` / `particles.npy` /
  `halos.parquet` + `lightcone.parquet`). `LinearSimConverter` is the
  reference `SimConverter`. Deps: numpy + pyarrow + pyyaml + healpy.
- `oneuniverse/simulation/oufsim/` — the **OUF-Sim on-disk store**,
  mirroring OUF's stack: `manifest.json` (atomic JSON) + pyarrow
  parquet partitions + HEALPix-NEST sky partitions + memmap `.npy`
  field tiles, each product carrying a sidecar `_index.parquet`
  (per-chunk bbox / super-pixel) for **partial-access** reads.
  `write_oufsim_store(native, out_root, sim_name=)` builds it;
  `SimStore(root).read_box / read_field_box / read_cone` reads only
  the overlapping partitions (`.last_read_stats`). Layer-1 index
  toolkit in `oufsim/index.py`. Demo store + profiling + plots at
  `/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation/linsim_demo`
  (`scripts/build_demo_oufsim.py`). Optimisation hotspots +
  next-phase plan: `research/2026-06-02-oufsim-optimization-findings.md`,
  `plans/2026-06-02-phaseS5-oufsim-optimisation-and-coverage.md`.
- `oneuniverse/twin/` — the **data↔simulation coupling layer** (the
  third layer per the substrate ADR; may import BOTH `simulation` and
  `data`, which neither pillar may host — `simulation/` stays Rule-1
  clean, the guard scans `simulation/` only). MVP = the **mock
  challenge**: truth field → `mock_tracer_field` (biased Poisson
  tracers) → `wiener_reconstruct` (constrain) → `cross_correlation`
  r(k) vs truth (verify). `run_mock_challenge(...)` returns r(k), the
  feasibility number (scale where r=0.5 per survey n̄). Demo + plots at
  `…/oneuniverse_simulation/mock_challenge` (`scripts/mock_challenge_demo.py`).
  ADR + plan: `research/2026-06-02-adr-oneuniverse-as-general-substrate.md`,
  `plans/2026-06-02-phaseC1-minimal-coupling-mock-challenge.md`.
- `oneuniverse/data/surveys/` — registered loaders. Add new ones with
  `@register class FooLoader(BaseSurveyLoader)`.
- `plans/` — phase-by-phase + pillar-level roadmaps. Stabilisation
  (Phases 1–15) done; generalisation (Phases 16–23) planned.
- `docs/` — Sphinx scaffold (`make html` from `docs/`).
- `research/` — topical references + design analyses.
  - [`research/survey_landscape_review.md`](research/survey_landscape_review.md)
    — full cosmology survey landscape (Pillar 1 driver).
  - [`research/schema_generalisation_audit.md`](research/schema_generalisation_audit.md)
    — Pillar 1 schema gap analysis + Phase 16–22 roadmap.
  - [`research/cosmological_simulation_landscape.md`](research/cosmological_simulation_landscape.md)
    — Pillar 3 (digital twin) reference: codes, public suites, on-
    disk representations, proposed OUF-Sim manifest-of-manifests
    format (2026-06-01).
  Consult before scoping any new loader, schema change, or Pillar-3
  ingestor.
- `test/` — pytest suite (~3:30 wall-clock, 364+ tests).

## OUF 2.5 (format on disk)

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

Phase 17 adds:
- `PartitionStats.extra_ranges: Dict[str, (lo, hi)]` populated when
  `write_ouf_dataset(extra_stats_columns=[...])` is set; `DatasetView`
  prunes + pushes down via `extra_filters={col: (lo, hi)}`.
- Variable-length / fixed-size payloads (Lyα δ, lightcurves, GAIA XP,
  DESI BITWEIGHTS) route through
  `write_ouf_dataset(column_dtypes={"col": "list<f4>" | "f4[N]" | "i8[N]" | "large_list<f4>"})`.
  Mini-language lives in `oneuniverse.data.dtype_lang`.

Phase 18 adds:
- `PdfSpec` covers `interp / quant / mixmod / sample / hist` and
  carries configurable column aliases (`value_column`,
  `sigma_column`, `weights_column`) so RAIL / qp catalogs round-trip
  without renaming. `hist` stores per-row bin heights as `f4[N]`;
  `sample` stores per-row z-draws as `list<f4>`.
- `TomographicNzSpec` (per-bin n(z) on a shared z grid +
  `bin_assignment_column` int row column) and
  `ClassificationPdfSpec` (ordered class tuple + per-row
  `f4[n_classes]`) are dataset-level Manifest sub-specs.

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

Phase 20:
- `SubobjectRules` carries `relation_type` ∈
  ``{containment, causality, association}`` and an optional
  `next_level` pointing at the next link sidecar in a chain.
- `Database.chain_subobjects(starts=[…], relations=[name1, name2, …])`
  walks a sequence of link sidecars and returns the union of leaf
  oneuids (cluster → galaxy → spectrum, deblender hierarchies, …).
- `oneuniverse.data.subobject_map.build_subobject_links_to_map`
  matches a point catalog of parents against per-row HEALPix
  probability maps (fixed-NSIDE) and emits the canonical
  `SubobjectLinks` sidecar with ``confidence = pixel value``. Used
  for GW host association.

Phase 21:
- `CrossMatchRules.attribute_filters: Tuple[Callable, ...]` —
  pluggable predicates evaluated on candidate (left, right)
  DataFrames; return a bool mask. The matcher applies them after
  the dz cut. Filters are hashed by qualname so two semantically
  equal rule objects hash identically. The matcher's row stack
  preserves any non-canonical catalog columns (e.g. magnitudes) so
  filters can index them.
- CORE `composite_id: U64` (optional) — preserves the
  survey-published composite ID alongside the canonical `int64`
  `galaxy_id` (PLATE-MJD-FIBERID, KIDS_TILE+SeqNr, GAIA source_id, …).
- `oneuniverse.data.moc.rasterise_moc_to_healpix(moc_path, *, nside,
  nest=True)` bridges GW LIGO/Virgo multi-order MOC FITS to the
  fixed-NSIDE numpy arrays consumed by
  `build_subobject_links_to_map`. `mocpy` is an optional dev extra
  (`pip install .[dev]`).

Phase 22 (data-driven geometries; no mocks):
- `DataGeometry.CUBE` — observed N-D cubes (IFU MaNGA/SAMI/MUSE, HI
  WALLABY, 21cm CHIME/HERA). One row per cube; required columns
  `cube_id, ra, dec, shape (i4[3]), cube (list<f4>)`. WCS / axis
  metadata declared via `CubeSpec` on the Manifest (axes,
  axis_units, wavelength_convention).
- `DataGeometry.GW_SKYMAP` — per-event HEALPix probability maps
  after `mocpy` rasterisation. One row per event; required columns
  `event_id, event_name, map_nside, map_nest, prob (list<f4>)`.
  `GwSkymapSpec` declares NSIDE + ordering + has_distance_extras.
- `PARTICLE` / mock geometries are **owned by Pillar 3** — Pillar 1
  stays data-only.

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
- `ShearWeight(kind="metacal" | "lensfit", sigma_e_cols=…)` (Phase 19)
  — `w = shear_weight / (R_eff² + σ_e²)` with
  `R_eff = (R11+R22)/2 + R_S` (metacal) or `1 + m_bias` (lensfit).
  DES Y3 / KiDS-1000 / HSC-Y3 / Rubin.
- `PipBitweightWeight(mode="fraction" | "realisations")` (Phase 19)
  — expand DESI `BITWEIGHTS: i8[N]` to a fractional weight or a
  per-realisation 0/1 array (jackknife accumulator).
- `default_weight_for(survey_type, z_type, *, sub_kind=None)`
  (Phase 19) — registry key widens to
  `(survey_type, sub_kind, z_type)`; `sub_kind=None` matches the
  canonical pre-Phase-19 contract.
- New `shear` schema column group: `e1 / e2 / e1_err / e2_err /
  R11..R_S / m_bias / c1_bias / c2_bias / shear_weight`. All optional.

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
- 2026-05-29: Phases 16–20 complete; OUF 2.4.0; 487/487 tests green.
- **Phase 16–20** delivered observational metadata, variable-length
  columns, PDF polymorphism, shear + PIP, and map-based sub-object
  chains. Remaining Pillar 1 work: Phase 21 (deferred sub-object
  items — composite IDs, attribute filters, mocpy MOC), Phase 22
  (optional CUBE/PARTICLE/GW_SKYMAP geometries), Phase 23 (rolled-up
  real-survey loader writes). `MeasurementSet` and adapters were
  reassigned to Pillar 2 (separate `onemeasure` package).

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
