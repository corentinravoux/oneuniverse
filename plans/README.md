# oneuniverse Plans

Phase-level and pillar-level roadmaps for the oneuniverse project.
The plans here describe **why** and **how** each chunk of work lands;
detailed task lists live in dated phase files.

## Three-pillar structure (2026-05-28)

The whole stack is organised around three pillars:

- **Pillar 1 — Data + Combine.** Everything inside the `oneuniverse`
  package. Database construction from raw catalogs → cross-survey
  combination → typed parquet artefacts on disk. No cosmology, no
  estimators, no forward models, **no `MeasurementSet`** (that
  moved to Pillar 2 on 2026-05-29).
  See [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md).

- **Pillar 2 — `MeasurementSet` + External Scientific Tool
  Interfaces.** A new `onemeasure` package owns the `MeasurementSet`
  contract + builders (window, random, jackknife, n(z),
  multi-tracer) + adapters to `flip`, `pycorr`, `picca`, future
  `onecorr`. Pillar 2 is where cosmology enters.
  See [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md).

- **Pillar 3 — Simulation / Digital Twin.** Constrained Bayesian
  forward modelling of the actual Universe; per-survey observation
  models; mini-simulation zoom-ins; incremental updates.
  See [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md).

## Pillar 1 phase plans

### Stabilisation (Phases 1–15, complete 2026-05-22)

- [`2026-04-15-stabilisation-roadmap.md`](2026-04-15-stabilisation-roadmap.md) — master phase-level roadmap with rationale, design decisions, and scope per phase.
- [`2026-04-15-phase1-ouf-v2.md`](2026-04-15-phase1-ouf-v2.md) — Phase 1 (OUF 2.0 manifest + atomic writes + content hashing).
- [`2026-04-15-phase3-healpix-partitioning.md`](2026-04-15-phase3-healpix-partitioning.md) — Phase 3 (HEALPix spatial partitioning + cone / SkyPatch cell pruning).
- [`2026-04-15-phase4-unified-oneuid.md`](2026-04-15-phase4-unified-oneuid.md) — Phase 4 (CrossMatchRules policy, relocated cross-matcher, audit columns, named on-disk ONEUID indices, `WeightedCatalog.from_oneuid`).
- [`2026-04-16-phase5-streaming-hydration.md`](2026-04-16-phase5-streaming-hydration.md) — Phase 5 (row-level pushdown via `_original_row_index`, `OneuidQuery.iter_partial`).
- [`2026-04-16-phase6-housekeeping-combine.md`](2026-04-16-phase6-housekeeping-combine.md) — Phase 6 (`oneuniverse.weight` → `oneuniverse.combine` redesign, `default_weight_for`). Final `oneuniverse.weight` deprecation shim deleted 2026-05-28.
- [`2026-04-20-temporal-subobject-roadmap.md`](2026-04-20-temporal-subobject-roadmap.md) — joint roadmap for Phase 7 + 8.
- [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) — Phase 7 (temporal).
- [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) — Phase 8 (sub-object).
- [`2026-04-23-phase9-desi-dr1-onboarding.md`](2026-04-23-phase9-desi-dr1-onboarding.md) — Phase 9 (DESI DR1 QSO onboarding + fragility audit).
- [`2026-04-23-phase10-probabilistic-redshifts.md`](2026-04-23-phase10-probabilistic-redshifts.md) — Phase 10 (photo-z PDFs).
- [`2026-04-23-phase11-selection-weights.md`](2026-04-23-phase11-selection-weights.md) — Phase 11 (selection / completeness weight family).
- [`2026-05-22-phase12-carried-debt.md`](2026-05-22-phase12-carried-debt.md) — Phase 12 (carried-over debt cleanup).
- [`2026-05-22-phase14-performance-footprint.md`](2026-05-22-phase14-performance-footprint.md) — Phase 14 (performance + footprint).
- [`2026-05-22-phase15-docs-stability.md`](2026-05-22-phase15-docs-stability.md) — Phase 15 (docs + stability).

### Generalisation (Phases 16–22, planned 2026-05-28)

Driven by [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md)
and [`../research/schema_generalisation_audit.md`](../research/schema_generalisation_audit.md).
Per-phase detailed plans land as we start each.

| # | Name | Driver |
|---|------|--------|
| 16 | Observational metadata expansion (`CoordinateSpec`, `SpectrumSpec`, `z_type` registry, `ColumnDef.frame/epoch/λ-convention/nullable`) | GAIA epoch, SDSS air vs BOSS vacuum, multi-z columns |
| 17 | Variable-length columns + generic partition stats | Lyα δ, ZTF/Rubin lightcurves, GAIA XP, DESI BITWEIGHTS, multi-filter photometry, S/N pushdown |
| 18 | PDF polymorphism + tomographic n(z) + classification PDFs | RAIL / `qp` alignment, KiDS/DES/HSC tomographic bins |
| 19 | Shear group + `ShearWeight` + `PipBitweightWeight` | DES Y3/Y6, KiDS-1000, HSC-Y3, Rubin shapes, DESI PIP |
| 20 | Map-based ONEUID + multi-level sub-object chains | GW × galaxy, cluster→galaxy→spec, deblender trees |
| 21 | Cleanup of deferred sub-object items (composite-ID `galaxy_id`, `CrossMatchRules.attribute_filters`, `mocpy` multi-order MOC) | GWTC native ingest, composite-ID surveys, colour-aware cross-match |
| 22 | Data-driven geometry expansion: `CUBE` (observed IFU/HI/21cm) + `GW_SKYMAP` (event probability maps) — **no mocks** (`PARTICLE` is Pillar 3) | IFU cubes, HI cubes, 21cm intensity maps, GW skymaps |
| 23 | Real-survey loader writes (rolled-up Phase 13) | All loaders, depends on 16–20 |

## Phase status (Pillar 1)

| # | Name | Status |
|---|------|--------|
| 1 | OUF 2.0 (typed manifest, hashes, atomic writes) | **complete (2026-04-15, 130/130)** |
| 2 | DatasetView + pyarrow.dataset backend | **complete (2026-04-15, 145/145)** |
| 3 | HEALPix spatial partitioning | **complete (2026-04-15, 156/156)** |
| 4 | Unified ONEUID (z-type rules, subsets, named indices) | **complete (2026-04-16, 190/190)** |
| 5 | Streaming hydration | **complete (2026-04-16, 198/198)** |
| 6 | Housekeeping + `weight/` → `combine/` redesign | **complete (2026-04-20, 211/211)** |
| 7 | Temporal data (t_obs + LIGHTCURVE + bitemporal database + versioned ONEUID) | **complete (2026-04-21, 265/265)** |
| 8 | Sub-object hierarchy (bitemporal link sidecars) | **complete (2026-04-22, 292/292)** |
| 9 | DESI DR1 QSO onboarding end-to-end + fragility audit | **complete (2026-04-23, 299/299; F1+F2 fixed, F3 deferred)** |
| 10 | Probabilistic redshifts (photo-z PDFs) | **complete (2026-04-23, 326/326)** |
| 11 | Generic selection / completeness weight family | **complete (2026-04-23, 345/345)** |
| 12 | Carried-over debt cleanup | **complete (2026-05-22, 361/361)** |
| 14 | Performance + footprint | **complete (2026-05-22, 364/364; suite 277s → 205s)** |
| 15 | Docs + stability hardening | **complete (2026-05-22, 365/365)** |
| — | `oneuniverse.weight` deprecation shim deleted | **complete (2026-05-28, 365/365)** |
| 16 | Observational metadata expansion (`CoordinateSpec`, `SpectrumSpec`, extensible `z_type` registry, `ColumnDef.frame/epoch/wavelength_convention/nullable`, OUF → 2.2.0) | **complete (2026-05-28, 406/406 tests green)** |
| 17 | Variable-length columns + generic `PartitionStats.extra_ranges` (dtype mini-language, `column_dtypes` writer kwarg, `extra_stats_columns`, `DatasetView.extra_filters`, OUF → 2.3.0) | **complete (2026-05-29, 428/428 tests green)** |
| 18 | PDF polymorphism (`sample` / `hist`) + column aliases + `TomographicNzSpec` + `ClassificationPdfSpec` (OUF → 2.4.0) | **complete (2026-05-29, 450/450 tests green)** |
| 19 | Shear column group + `ShearWeight` + `PipBitweightWeight` + sub-species registry key | **complete (2026-05-29, 472/472 tests green)** |
| 20 | Map-based sub-object (point × HEALPix probability map) + multi-level chain walker + `relation_type` / `next_level` on `SubobjectRules` | **complete (2026-05-29, 487/487 tests green)** |
| 21 | Cleanup of deferred sub-object items (`CrossMatchRules.attribute_filters`, CORE `composite_id`, `mocpy` MOC rasteriser) | **complete (2026-05-29, 499/499 tests green; 2 skipped — mocpy optional)** |
| 22 | Data-driven geometry expansion: `CUBE` (observed IFU/HI/21cm) + `GW_SKYMAP` (event probability maps); OUF → 2.5.0 | **complete (2026-05-29, 522/522 tests green; 2 skipped — mocpy optional)** |
| 23 | Real-survey loader writes (rolled-up Phase 13) | planned (after 16–20) |

## Pillar 2 / Pillar 3 phase plans

Pillar 2 begins with Phase 0: standing up the new `onemeasure` package
(MeasurementSet + builders + adapters) against the current OUF 2.4
format. Per-phase detailed plans get written when work begins.

- Pillar 2 phases A–F: see
  [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md).

Pillar 3 (simulation storage + orchestration; digital-twin substrate)
started 2026-06-01 with an architecture proposal. Standalone package;
partial-access API is load-bearing; minimal cross-pillar coupling;
MPI/GPU reads first-class; mini-simulation runs deferred indefinitely
(see [[pillar3-partial-access-and-minimal-deps]] in memory).

- Pillar 3 large-scope roadmap:
  [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md).
- **Phase S1 — OUF-Sim architecture proposal** (doc only, no code):
  [`2026-06-01-phaseS1-oufsim-architecture.md`](2026-06-01-phaseS1-oufsim-architecture.md).
  Defines OUF-Sim format (input+output split), `SimDatabase`,
  `SimConverter`, `SimDatasetView` (partial access), region-selection
  orchestration → `SimulationRequest`, lineage + convertibility.
- **Phase S2 — `oneuniverse.simulation` skeleton + types** (executed):
  [`2026-06-01-phaseS2-oufsim-skeleton.md`](2026-06-01-phaseS2-oufsim-skeleton.md).
- Phase S3+ (AbacusSummit backend, orchestration) get detailed plans
  when work begins.

Backends-first was dropped 2026-06-01: real simulation formats
(AbacusSummit ASDF/pack9, Gadget HDF5, AMR/HACC/BORG/BigFile) are
**all deferred to the future bucket**. Instead the architecture is
finished against a **dummy linear simulation** — a pure-numpy
Eisenstein–Hu power spectrum + linear-theory LSS generator that emits
every product type (field/voxel, particles, halos, lightcone). Same
strategy as Pillar 1 starting on synthetic DR1 fixtures.

| Phase | Name | Status |
|---|---|---|
| S1 | OUF-Sim architecture proposal | **complete (2026-06-01, doc only)** |
| S2 | `oneuniverse.simulation` skeleton + types + no-Pillar-1-import lint guard | **complete (2026-06-01, 571/571 tests green; +49 sim)** |
| S3 | Dummy linear simulation generator (`oneuniverse.simulation.linear`): Eisenstein–Hu P(k), growth D(z), Gaussian field (mesh/voxel), Zel'dovich particles, toy halos | **complete (2026-06-02, 601/601 tests green; +30 linear)** |
| S4 | **OUF-Sim store format** (`manifest.json` + parquet + HEALPix-NEST + memmap `.npy` tiles, mirroring OUF) + Layer-1 index toolkit (`oufsim/index.py`) + `LinearSimConverter` real `convert()` + `SimStore` partial-access reads + lightcone product; demo dataset + cProfile/tracemalloc profiling at `/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation/linsim_demo` | **complete (2026-06-02, 606/606 tests green; +5 oufsim)** |
| **C1** | **Minimal data↔sim coupling — the mock challenge** (`oneuniverse/twin/`, the coupling layer that may import both pillars). truth → `mock_tracer_field` (biased Poisson) → `wiener_reconstruct` → `cross_correlation` r(k) vs truth. Closes the scientific loop on the dummy with a feasibility number (k where r=0.5 per n̄). Pulled **ahead of S5–S8** as the keystone | **complete (2026-06-02, +10 twin tests; demo at `…/oneuniverse_simulation/mock_challenge`)** — [`plans/2026-06-02-phaseC1-minimal-coupling-mock-challenge.md`](2026-06-02-phaseC1-minimal-coupling-mock-challenge.md) |
| S5 | **OUF-Sim write-path optimisation + full product coverage** (driven by [`research/2026-06-02-oufsim-optimization-findings.md`](../research/2026-06-02-oufsim-optimization-findings.md)): bounded-memory streaming bucket-chunker, parallel/MPI partition writes, `ExecutionPlan` enforcement, wrap-don't-re-encode projection, + `tree`/`phase_space`/`gr_fields`/`checkpoints`/`ic_posterior` dummy products, `SimDatasetView` (correct streaming reads; read *optimisation* is S6) | planned — [`plans/2026-06-02-phaseS5-oufsim-optimisation-and-coverage.md`](2026-06-02-phaseS5-oufsim-optimisation-and-coverage.md) |
| S6 | **OUF-Sim read-path optimisation** (benchmark harness + tests): column projection, predicate pushdown / row-group skipping, index LRU cache, threaded parallel reads, Morton row order, GPU-direct read hook; each lever benchmarked + regression-tested | planned — [`plans/2026-06-02-phaseS6-oufsim-read-optimisation.md`](2026-06-02-phaseS6-oufsim-read-optimisation.md) |
| S7 | **AMR octree layout + input/IC products**: toy 1-level refinement around peaks (`sim_kind=amr`, Morton octree-node index, `read_amr_box`) + initial-conditions product (`has_input=True`). After S5–S7 all 9 `PRODUCT_KINDS` + AMR + both input/output sides are exercised | planned — [`plans/2026-06-02-phaseS7-amr-and-input-products.md`](2026-06-02-phaseS7-amr-and-input-products.md) |
| S8 | **Resimulation orchestration — the digital-twin core** (sCOLA + zoom ICs + separate-universe tides). Six sub-phases: S8.0 research/feasibility (✅ doc), S8.1 fast-PM mini-simulator, S8.2 full-volume far-field φ(a) provider, S8.3 region IC extraction + **Gate 1** (pre-run large-scale match), S8.4 COLA-frame coupling + buffers/overlap, S8.5 merge + **Gate 2/3** (post-run match + error-budget go/no-go), S8.6 `SimDatabase` control plane driving extract→run→merge→verify. Critical verdict: feasible as controlled approximation, **not** a full-sim replacement | planned — [`plans/2026-06-02-phaseS8-resimulation-orchestration.md`](2026-06-02-phaseS8-resimulation-orchestration.md); research [`research/2026-06-02-resimulation-orchestration-feasibility.md`](../research/2026-06-02-resimulation-orchestration-feasibility.md) |
| future | **Real-format backends** (AbacusSummit ASDF/pack9, Gadget HDF5, AMR/HACC/BORG/BigFile) + heavy real-code mini-sim runs + IC samplers (deferred indefinitely) |
