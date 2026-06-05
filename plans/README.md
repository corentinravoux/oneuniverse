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

- **Pillar 2 — Estimators, Likelihoods, the DataProduct boundary.**
  The P1→P2 input contract is a **Universal DataProduct** (PointSet /
  Sightline / FieldMap). A new `onemeasure` package owns the contract
  + builders (window, random, jackknife, n(z)) + adapters; a new
  `onecorr` owns cross / multi-tracer. flip is **one of five** built
  estimator families (flip, p1desi, lyapower, lyavoid, lelantos).
  Pillar 2 is where cosmology enters.
  **Canonical:** [`2026-06-05-pillar2-definition.md`](2026-06-05-pillar2-definition.md)
  (supersedes [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md)).

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

- Pillar 2 phases P0–PF (DataProduct boundary → flip adopt → pycorr →
  Sightline/Lyα → FieldMap → onecorr → cobaya joint): see the canonical
  [`2026-06-05-pillar2-definition.md`](2026-06-05-pillar2-definition.md).

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
| **C1** | **Minimal data↔sim coupling — the mock challenge** (`oneuniverse/twin/`, the coupling layer that may import both pillars). truth → `mock_tracer_field` (biased Poisson) → `wiener_reconstruct` → `cross_correlation` r(k) vs truth. Closes the scientific loop on the dummy with a feasibility number (k where r=0.5 per n̄) | **complete (2026-06-02, +10 twin tests; demo at `…/oneuniverse_simulation/mock_challenge`)** — [`plans/2026-06-02-phaseC1-minimal-coupling-mock-challenge.md`](2026-06-02-phaseC1-minimal-coupling-mock-challenge.md) |
| **C2** | **Engine contracts + first plugins** — `ReconstructionEngine` (data→field) + `ForwardEngine` (field→products) ABCs + registry; `WienerReconstruction` + `LinearForwardEngine`. Generality demonstrated with one engine of each role | **complete (2026-06-02, +5 tests)** |
| **C3** | **Validation harness** (`twin/validation.py`) — `RecoveryMetrics` (r(k) / transfer / power-ratio / `k_half`); r(k) insensitive to filters, sensitive to noise; regression-tested | **complete (2026-06-02, +3 tests)** |
| **C4** | **Mock data / selection layer** (`twin/mock_survey.py`) — slab/ball masks + radial completeness (mock n(z)); masked reconstruction recovers inside the footprint | **complete (2026-06-02, +4 tests)** |
| **C5** | **Constrained realization** (Hoffman–Ribak) — Wiener mean → proper constrained IC (small-scale P(k) restored, ensemble mean = WF); also documents the linear-bias/clip caveat (b_eff drops at σ≳1) | **complete (2026-06-02, +3 tests)** |
| S5 | **OUF-Sim write-path optimisation + product coverage** — ✅ T1 streaming bucket-chunker (bounded mem, fused bbox), T2 parallel/MPI partition writes, T3 ExecutionPlan enforcement, T5 merger-tree, T6 phase_space + gr_fields + checkpoints (all 8 product kinds in the store), T7 `SimDatasetView` streaming reads. ⏳ remaining: T4 wrap-don't-re-encode (reference projection), T8 demo refresh | **mostly complete (2026-06-02; T4+demo remaining)** — [`plans/2026-06-02-phaseS5-oufsim-optimisation-and-coverage.md`](2026-06-02-phaseS5-oufsim-optimisation-and-coverage.md) |
| S6 | **OUF-Sim read-path optimisation** — ✅ T1 benchmark harness, T2 column projection, T3 predicate pushdown (pyarrow filters), T4 index LRU cache, T5 threaded parallel reads, T6 Morton row order (tighter 3D row-group boxes), T7 GPU-direct hook (cuDF fallback). ⏳ remaining: T8 benchmark-suite script (non-core) | **mostly complete (2026-06-02)** — [`plans/2026-06-02-phaseS6-oufsim-read-optimisation.md`](2026-06-02-phaseS6-oufsim-read-optimisation.md) |
| S7 | **AMR octree layout + input/IC products** — ✅ T1 `refine_field` (1-level octree, Morton node ids), T2 `fields_amr` store layout, T3 `read_amr_box` (cube-pruned nodes), T4 `white_noise_ic` IC product (`has_input=True`). **All 9 PRODUCT_KINDS + AMR + both input/output sides now exercised.** ⏳ T5 converter-decl + demo (cosmetic) | **mostly complete (2026-06-02)** — [`plans/2026-06-02-phaseS7-amr-and-input-products.md`](2026-06-02-phaseS7-amr-and-input-products.md) |
| **S8** | **Resimulation orchestration — the digital-twin core, the big stage** (sCOLA + zoom ICs + separate-universe tides). ✅ S8.0 research, ✅ **S8.1 fast-PM mini-sim** (`oneuniverse/simulation/pm/`: CIC+FFT-Poisson+KDK leapfrog; reproduces linear growth to ~few %; wired as the 2nd `ForwardEngine` `PMForwardEngine`). ✅ **S8.2** far-field φ(x;a), ✅ **S8.3** IC extraction + **Gate 1**, ✅ **S8.4** buffer-region resimulation (sCOLA-lite, `resim/coupling.py`), ✅ **S8.5 Gate 2/3 — feasibility DEMONSTRATED** (inner-region agreement 0.61→0.96 as buffer 16→64 Mpc/h; capstone fig at `…/resim_feasibility`), ✅ **S8.6** `SimDatabase` orchestration (catalog→region→request→dispatch resim→lineage). Verdict confirmed: selective resimulation feasible as a controlled approximation | **complete (2026-06-02)** — [`plans/2026-06-02-phaseS8-resimulation-orchestration.md`](2026-06-02-phaseS8-resimulation-orchestration.md); research [`research/2026-06-02-resimulation-orchestration-feasibility.md`](../research/2026-06-02-resimulation-orchestration-feasibility.md) |
| **S17** | **General storage, IO & optimisation — multi-backend substrate.** ✅ T1 native-adapter row reads + format registry (`get_adapter`/`register_adapter`), ✅ T2 `packed_npy` 2nd native format (chunk-sorted slab + header) + adapter, ✅ T3 format-agnostic `build_store` + `NativeProduct` (reuses per-product writers), ✅ T4 **`PackedSimConverter`** — 2nd backend reads identically to linear (generality proof; closes S5 "re-encode only"), ✅ T5 **particle wrap-in-place** (`projection="reference"`, index-only ≈14% of re-encode; closes S5 T4 + S15 particle gap), ✅ T6 `ExecutionPlan.batch_for` budget→batch bounded streaming, ✅ T7 MPI rank-partitioned reads + honest GPU hook, ✅ T8 scale sweep + wrap-vs-reencode figure. Real backends (ASDF/HDF5/BigFile) implement the same adapter+converter recipe | **complete (2026-06-04)** — [`plans/2026-06-04-phaseS17-general-storage-io-optimisation.md`](2026-06-04-phaseS17-general-storage-io-optimisation.md) |
| **measure** | **Pillar-2 connection in `oneuniverse.measure`** (owner: lives in oneuniverse, defines the general output format others adapt to). Universal DataProduct + MeasurementSet; build probe-by-probe. ✅ **galaxy clustering (spec)** — 9-step transform (select/clean/weight/randoms[ingest+generate]/window/n(z)/region/spec/assemble) → cosmology-free MeasurementSet; TDD on synthetic OUF; diagnostic figure. ✅ **all 5 probe connections built** (clustering, [WL](2026-06-05-measure-weak-lensing.md), [PV/SN](2026-06-05-measure-pv-sn.md), [Lyα](2026-06-05-measure-lya.md), [map×catalog](2026-06-05-measure-map-cross.md)) across **3 DataProduct subtypes** (PointSet/Sightline/FieldMap); 28 measure tests + diagnostic figures. ⏳ next: real DESI/eBOSS validation; estimator-side adapters (separate, later) | **all probes built (2026-06-05)** — [`plans/2026-06-05-measure-galaxy-clustering.md`](2026-06-05-measure-galaxy-clustering.md); design [`plans/2026-06-05-pillar2-definition.md`](2026-06-05-pillar2-definition.md) |
| **Track A** | **P1+P2 community product** (`onemeasure`/MeasurementSet → flip cross-correlation) | **deferred several months** (owner decision 2026-06-02) |
| future | **Real-format backends** (AbacusSummit ASDF/pack9, Gadget HDF5, AMR/HACC/BORG/BigFile) + heavy real-code mini-sim runs + IC samplers (deferred indefinitely) |
